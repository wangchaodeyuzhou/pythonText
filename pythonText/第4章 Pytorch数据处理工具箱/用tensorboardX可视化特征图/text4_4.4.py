# -*- coding = utf-8 -*-
# @Time : 2020/11/13 22:40
# @Author : 王朝的宇宙
# @File : text4_4.4.py
# @Software : PyCharm


# 利用tensorboardX对特征图进行可视化，不卷积层的特征图的抽取程度是不一样的。
# x从cifair10数据集获取，具体请参考第6章代码pytorch-06。
import torchvision.utils as vutils
from caffe2.quantization.server.observer_test import net
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='logs', comment='feature map')
x = 1  # 这是我自己定义的
img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)
net.eval()
for name, layer in net._modules.items():

    # 为fc层预处理
    x = x.view(x.size(0), -1) if "fc" in name else x
    print(x.size)

    x = layer(x)
    print(f"{name}")
    # 查看卷积层的特征图
    if 'layer' in name or 'conv' in name:
        x1 = x.transpose(0, 1)  # C,B, H,W  --> B, C, H, W
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)
        #  normalize 进行归一化处理
        writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)