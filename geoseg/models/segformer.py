from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helpers

def exists(val):
    return val is not None


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads,
            reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))  # q:(1,32,64,64)k:(1,32,8,8)v:(1,32,8,8)
        # 1,(1,32),64,64-->((1,1),4096,32)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))  # h=1
        # q(1,4096,32),k(1,64,32),v(1,64,32)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # (1,4096,64)
        attn = sim.softmax(dim=-1)  # (1,4096,64)

        out = einsum('b i j, b j d -> b i d', attn, v)  # (1,4096,32)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)  # (1,32,64,64)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
            self,
            *,
            channels,  # 3
            dims,  # (32,64,160,256)
            heads,  # (1,2,5,8)
            ff_expansion,  # (8,8,4,4,)
            reduction_ratio,  # (8,4,2,1)
            num_layers  # (2,2,2,2)
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)  # (3,,32,64,160,256)
        dim_pairs = list(zip(dims[:-1], dims[1:]))  # [(3,32),(32,64),(64,160,(160,256))]

        self.stages = nn.ModuleList([])  #

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio \
                in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            # (3,32),(7,4,3),(2),(8),(1),(8)
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)  # (7,4,3)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)  # conv2d(147,32,1,1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):  # 循环两次
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
            self,
            x,
            return_layer_outputs=False
    ):
        h, w = x.shape[-2:]  # 256,256

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)  # (1,147,4096)

            num_patches = x.shape[-1]  # 4096
            ratio = int(sqrt((h * w) / num_patches))  # 4
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)  # (1,147,64,64)

            x = overlap_embed(x)  # (1,32,64,64)
            # stage每迭代一次，layer迭代2次。
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x  # (1,32,64,64)

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class Segformer(nn.Module):
    def __init__(
            self,
            *,
            dims=(32, 64, 160, 256),
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=2,
            channels=3,
            decoder_dim=256,
            num_classes=4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth=4), (
        dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio,
                                                 num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),  # (input,256)
            nn.Upsample(scale_factor=2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):  # (1,3,256,256)
        layer_outputs = self.mit(x, return_layer_outputs=True)  # 四个输出

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]  # list:4
        fused = torch.cat(fused, dim=1)  # (1,1024,64,64)
        return self.to_segmentation(fused)  # (1,num_class,64,64)


def main():
    model = Segformer(
        dims=(32, 64, 160, 256),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=2,  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=4  # number of segmentation classes
    )
    model.eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        pred = model(x)
    print(pred)


if __name__ == '__main__':
    main()