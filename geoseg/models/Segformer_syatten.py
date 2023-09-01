from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helpers
#
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

#
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
        #（卷积核，步长，填充值）
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)  # (3,,32,64,160,256)把维度拼接起来
        dim_pairs = list(zip(dims[:-1], dims[1:]))  # [(3,32),(32,64),(64,160）,(160,256)]相邻的两维度组成一个元组

        self.stages = nn.ModuleList([])  #stage是个模型模块类型，其中存储着对输入图形操作的函数

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio \
                in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            # (3,32),(7,4,3),(2),(8),(1),(8)将这几组数据按照对应的索引组合在一起
            #get_overlap_patches是滑动剪切函数
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)  # (7,4,3)nn。unfold，用一个类似卷积核的窗口对图片进行裁剪
            #
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)  # conv2d(147,32,1,1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):  # 循环两次，num_layers (2,2,2,2)
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),#先归一化x后EfficientSelfAttention
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),#再归一化，最后MixFeedForward
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
        h, w = x.shape[-2:]  # 512,512 获得输入的长宽

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)  # (1,147,16384)其中147=c*k*k=3x7x7

            num_patches = x.shape[-1]  # 将一张图剪切成16384份，每份7x7
            ratio = int(sqrt((h * w) / num_patches))  # 4
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)  # (1,147,64,64)对图片维度进行操作，把(1,147,16384)拆成(1,147,128，128)

            x = overlap_embed(x)  # 即overlap_patch_embed模块 一个卷积模块，不改变尺寸，只改变通道数 (1,32,128,128)
            # stage每迭代一次，layer迭代2次。
            for (attn, ff) in layers:#attn：EfficientSelfAttention模块，ff：MixFeedForward模块
                x = attn(x) + x
                x = ff(x) + x  # (1,32,64,64)

            layer_outputs.append(x)
            #layer_outputs储存了四层transformer block的输出，如果return_layer_outputs为Flase时，只输出最后一层transformer block的输出
        ret = x if not return_layer_outputs else layer_outputs
        return ret


class FR(nn.Module):
    def __init__(self,c=3,h=128,w=128):
        super().__init__()

        self.pooling_h0= nn.AvgPool2d(kernel_size=(128, 1), stride=1, padding=0)
        self.pooling_h1 = nn.AvgPool2d(kernel_size=(64, 1), stride=1, padding=0)
        self.pooling_h2 = nn.AvgPool2d(kernel_size=(67, 1), stride=1, padding=0)


        self.pooling_w0 = nn.AvgPool2d(kernel_size=(1, 128), stride=1, padding=0)
        self.pooling_w1 = nn.AvgPool2d(kernel_size=(1, 64), stride=1, padding=0)
        self.pooling_w2 = nn.AvgPool2d(kernel_size=(1, 67), stride=1, padding=0)

    def forward(self, x):
        pooling_h0 = self.pooling_h0(x)
        pooling_h1 = self.pooling_h1(x)
        pooling_h2 = self.pooling_h2(x)
        pooling_h = [pooling_h0,pooling_h1,pooling_h2]
        pooling_h = torch.cat(pooling_h, dim=2)

        pooling_w0 = self.pooling_w0(x)
        pooling_w1 = self.pooling_w1(x)
        pooling_w2 = self.pooling_w2(x)
        pooling_w = [pooling_w0,pooling_w1,pooling_w2]
        pooling_w = torch.cat(pooling_w, dim=3)

        pooling = [pooling_h,pooling_w]
        pooling = torch.cat(pooling, dim=3)#(8,1024,128,256)
        pooling = torch.transpose(pooling, 1, 2)#(8,256,1024,128)
        return pooling
class Sy_Attention_Model(nn.Module):
    def  __init__(self):
        super().__init__()
        self.fr = FR()
        self.conv = nn.Conv2d(1024, 1024, (1,7),(1,2),(0,3))

    def forward(self,x):
        feature0=self.fr(x)#(8,256,1024,128)(b,h+w,c,s)
        feature1 = torch.transpose(feature0, 2, 3)
        feature_mul = torch.matmul(feature0,feature1)#(8,128,1024,1024)即A（i，j）
        feature_mul = torch.matmul(feature_mul,feature0)#(8,128,1024,256)
        feature_ed = feature_mul + feature0 #(8,128,1024,256)
        feature_ed = torch.transpose(feature_ed, 1, 2)
        feature_ed = self.conv(feature_ed)
        #feature_ed = torch.reshape(feature_ed,(8,1024,128,128))



        return feature_ed



class Segformer(nn.Module):
    def __init__(
            self,
            *,
            dims=(32, 64, 160, 256),
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=(2, 2, 2, 2),
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
            # 再四倍上采样才是原图大小
            nn.Upsample(scale_factor=4),
            nn.Conv2d(decoder_dim, num_classes, 1),


        )

        self.Sy_Attention_Model = Sy_Attention_Model()

    def forward(self, x):  # (1,3,256,256)
        layer_outputs = self.mit(x, return_layer_outputs=True)  # 四个输出，储存了四层transformer block的输出

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]  # to_fused是先卷积只改变通道数，后上采样改变尺寸，最终使四个layer_outputs，变为尺寸通道数一致的四个tensor
        fused = torch.cat(fused, dim=1)  # (8,1024,128，128)对修改尺寸和通道的四个tensor进行拼接
        # 协同注意力↓ fused：（8，1024，128，128）
        fused= self.Sy_Attention_Model(fused)
        # 协同注意力↑
        out = self.to_segmentation(fused)  # (1,num_class,128，128)两次卷积都用来降低维度到num_class，且不影响图片尺寸
        return out


def main():
    #b0
    model = Segformer(
        dims=(32, 64, 160, 256),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(2, 2, 2, 2),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    # b1
    model = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(2, 2, 2, 2),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    # b2
    model = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(3, 3, 6, 3),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    # b3
    model = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(3, 3, 18, 3),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    # b4
    model = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(3, 8, 27, 3),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    #b5
    model = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(4, 4, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(3, 6, 40, 3),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=6  # number of segmentation classes
    )
    model.eval()
    x = torch.randn(8, 3, 512, 512)

    with torch.no_grad():
        pred = model(x)
    print(pred)


if __name__ == '__main__':
    main()