# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 22:05
# @Author  : cfushn
# @Comments: 原始GAN训练脚本
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from GAN import GAN

# 超参数
IMG_SIZE = 32
BATCH_SIZE = 128
DIM_Z = 128  # 噪声向量z的维度
LR_D = 2e-4  # 判别器的学习率，稍高于生成器
LR_G = 1e-4  # 生成器的学习率
EPOCHS = 100

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将图像归一化到[-1, 1]
])
train_dataset = datasets.MNIST(root="../../data", train=True, download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    # 初始化GAN模型
    model = GAN(DIM_Z, 1, EPOCHS, LR_G, LR_D, train_loader, device)
    # 训练模型
    model.train()
