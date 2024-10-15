# -*- coding: utf-8 -*-
# @Time    : 2024/9/11 19:29
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os

import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torch.nn import functional
from torchvision.utils import save_image

from t01_VAE2 import VAE

# hyperparameters
BATCH_SIZE = 256
LATENT_DIM = 64  # 隐变量Z的维度
IMG_SIZE = 32  # 图片维度
LR = 1e-4  # 1 * 10^(-4) = 0.0001
WEIGHT_DECAY = 1e-4  # 权重衰减参数
ALPHA = 1e-4  # lambda的值
EPOCHS = 400

# device
device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"

# data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),  # 将图像缩放至32✖32分辨率
    transforms.ToTensor(),  # 像素值[0,255]归一化为[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 像素值从[0,1]归一化为[-1,1]
])
train_dataset = datasets.FashionMNIST(root="../data", train=True, download=True,
                                      transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

# new a model
model = VAE(img_size=[1, IMG_SIZE, IMG_SIZE], latent_dim=LATENT_DIM).to(device)
# new a optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# loss
def loss_function(x, x_hat, mean, log_var, alpha=ALPHA):
    reproduction_loss = functional.mse_loss(x_hat, x)
    term1 = 0.5 * torch.mean(torch.sum(torch.pow(log_var.exp(), 2), dim=1))
    term2 = 0.5 * torch.mean(torch.sum(torch.pow(mean, 2), dim=1))
    term3 = -0.5 * torch.mean(2 * torch.sum(log_var, dim=1))
    # D为常数,不影响优化,可忽略
    KLD = term1 + term2 + term3
    return reproduction_loss + alpha * KLD


# train
def train(model, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            batch_size = x.size(0)
            optimizer.zero_grad()
            x_hat, _, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ",
              overall_loss / (batch_idx * batch_size))

        # 每10个epoch生成100个样本用于可视化
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_images = model.sample(100, device)  # 生成100个样本
                sample_images = sample_images.detach().cpu()
                save_file = "./output/{}.png".format(epoch + 1)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                save_image(sample_images.data, save_file, nrow=10,
                           normalize=True)


if __name__ == "__main__":
    train(model, optimizer, EPOCHS, device)
