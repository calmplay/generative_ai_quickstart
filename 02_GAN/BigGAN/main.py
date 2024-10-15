# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 23:25
# @Author  : cfushn
# @Comments: BigGAN的主程序，加载数据、设置超参数并启动训练
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from BigGAN import BigGAN

# 超参数设置
IMG_SIZE = 128
BATCH_SIZE = 256
DIM_Z = 128  # 噪声向量z的维度
LR_D = 1e-4  # 判别器的学习率
LR_G = 1e-4  # 生成器的学习率
EPOCHS = 200
NUM_ITER_D = 2  # 判别器更新次数
N_CLASSES = 10  # 分类数

# 设置设备
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
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root="../../data", train=True,
                                      download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)

if __name__ == "__main__":
    # 初始化BigGAN模型
    model = BigGAN(DIM_Z, 1, N_CLASSES, EPOCHS, LR_G, LR_D, NUM_ITER_D,
                   train_loader, device)
    print(model)
    # 训练并保存模型
    model.train()
