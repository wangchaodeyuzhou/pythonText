# -*- coding = utf-8 -*-
# @Time : 2020/10/29 8:07
# @Author : 王朝的宇宙
# @File : text4_4_3.py
# @Software : PyCharm


# 可视化损失值，使用add_scalar函数，这里利用一层全连接神经网络，训练一元二次函数的参数。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
import numpy as np

dtype = torch.FloatTensor
# 写操作
# input-size output_size
input_size = 100
output_size = 100
learning_rate = 0.01  # 学习速率为0.01
num_epoches = 20  # 轮数定为20
writer = SummaryWriter(log_dir="logs", comment="Linear")
#  设置np的随机种子
np.random.seed(100)
x_train = np.linspace(-1, 1, 100).reshape(100, 1)
y_train = 3 * np.power(x_train, 2) + 2 + 0.2*np.random.rand(x_train.size).reshape(100, 1)

# 分别的训练出x，y对应的值

# 线性化的全连接层
# 全连接层中
model = nn.Linear(1, 1)

criterion = nn.MSELoss()  # 利用MSE函数
# 优化器还是采用SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    inputs = torch.from_numpy(x_train).type(dtype)
    targets = torch.from_numpy(y_train).type(dtype)

    output = model(inputs)  # 需经过Linear来进性输出
    loss = criterion(output, targets)   # 计算损失函数
    optimizer.step()  # 优化器进行搞
    # 保存在wariter
    writer.add_scalar("训练损失值", loss, epoch)
print("训练完毕")
