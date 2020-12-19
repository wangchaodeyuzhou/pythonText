#-*- coding = utf-8 -*-
#@Time : 2020/10/25 10:26
#@Author : 王朝的宇宙
#@File : text4_2.py
#@Software : PyCharm

import torch
from torch.utils import data
import numpy as np


# 继承了Dataset数据集
class TestDataset(data.Dataset):
    def __init__(self):
        self.Data = np.asarray([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])
        self.Label = np.asarray([0, 1, 0, 1, 2])  # 这 是数据集对应得标签

    def __getitem__(self, item):  # 这是必须要实现得两个方法之一
        # 把numpy转化为 Tesnsor
        # 返回的是data和标签得分类
        txt = torch.from_numpy(self.Data[item])
        label = torch.tensor(self.Label[item])
        return txt, label

    def __len__(self):   # 返回数据集得长度
        return len(self.Data)
# 获取数据集中的数据


Test = TestDataset()
print(Test[2])  # 相当于调用了__getitem__(2)
print(Test.__len__())  # 输出数据集得长度

# 批量的的获取数据集的样本
# data.DataLoader(
#     dataset=Test,
#     batch_size=1,
#     shuffle=False,
#     sampler=None,  # 样本抽样
#     batch_sampler=None,
#     num_workers=0,  # 表示使用多进程加载数，0 表示不使用多进程
#     collate_fn=0,   # 将多个样本拼装成一个batch
#     pin_memory=False,  # 是否将数据保存在pin memory区
#     drop_last=False,   # dataset数据个数可能不是batch_size的整数倍，
#     # drop_last为True将会多出来的一个或不足丢弃
#     timeout=0,
#     worker_init_fn=None
# )
# 测试玩玩

test_loader = data.DataLoader(Test, batch_size=2, shuffle=False, num_workers=0)
for i, (Data, Label) in enumerate(test_loader):
    print('i:', i)
    print('data:', Data)
    print('Label:', Label)

# 可以转化为迭代器玩玩
print("-------------------")
dataiter = iter(test_loader)
(imges, labels) = next(dataiter)
print("imges ", imges)
print("labels ", labels)

