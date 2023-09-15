from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.taiyuan_dataset import *
from geoseg.models.Segformer import Segformer
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam

max_epoch = 105 #迭代次数epoch
ignore_index = len(CLASSES)#不用于计算指标的类别名，一般最后一类为其他类，因此ignore_index= len(CLASSES)

train_batch_size = 4#训练集batch_size
val_batch_size = 4#测试集batch_size
lr = 6e-4#学习率
weight_decay = 0.01#学习率衰减
backbone_lr = 6e-5#权重学习率
backbone_weight_decay = 0.01#权重衰减
num_classes = len(CLASSES)
classes = CLASSES#数据集类别组成的元组

weights_name = "segformerb5"#用于命名训练好的权重文件
weights_path = "model_weights/taiyuan/{}".format(weights_name)#训练好的权重文件路径
test_weights_name = "segformerb5"#在测试效果时，需要加载的已训练好的权重文件名
log_name = 'taiyuan/{}'.format(weights_name)#log日志文件，保存路径
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1#每n此epoch后，用val数据集评估
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'#自动选择gpu  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = Segformer(
        dims=(64, 128, 320, 512),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(4, 4, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=(3, 6, 40, 3),  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        num_classes=9  # number of segmentation classes
    )

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader

train_dataset = TaiyuanDataset(data_root='./data/taiyuan/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)#随机的在数据集中加载图片，以及对应的标签进行训练，旨在增加数据集的随机性，#
                                                                        # mosaic_ratio是随机比率，生成的随机数只有大于mosaic_ratio时，才允许加载原图像，否则加载处理后的图像
val_dataset = TaiyuanDataset(transform=val_aug)
test_dataset = TaiyuanDataset(data_root='./data/taiyuan/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,#num_worker类似与多线程，越大越快，但吃内存
                          pin_memory=True,#使硬件外设直接访问CPU内存
                          shuffle=True,#打乱数据集
                          drop_last=True)#drop_last=True时，在拿取最后一个batch_size时，不足batch_size个样本的话，就放弃这次拿去，如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
#layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}#骨干网络参数
#net_params = utils.process_model_params(net, layerwise_params=layerwise_params)#骨干网络参数生效
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)#优化器中ir ir_decay参数生效
optimizer = Lookahead(base_optimizer)#Lookahead是优化算法，实现需要 套用AdamW的部分代码
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
