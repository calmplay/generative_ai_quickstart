# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 22:00
# @Author  : cfushn
# @Comments:
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from SNGAN import SNGAN

# 超参数
IMG_SIZE = 32
BATCH_SIZE = 256
DIM_Z = 128  # 噪声向量z的维度
LR_D = 1e-4  # 判别器的学习率
LR_G = 1e-4  # 生成器的学习率
EPOCHS = 200
NUM_ITER_D = 2  # 每个循环判别器更新几次

# device
device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"--------------------use device: {device}--------------------")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),  # 将图像缩放至32✖32分辨率
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root="../../data", train=True,
                                      download=True,
                                      transform=transform)
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE,
                          shuffle=True)

if __name__ == "__main__":
    # 初始化SNGAN模型
    model = SNGAN(DIM_Z, 1, EPOCHS, LR_G, LR_D, NUM_ITER_D, train_loader,
                  device)
    print(model)
    # 训练并保存模型
    model.train()
