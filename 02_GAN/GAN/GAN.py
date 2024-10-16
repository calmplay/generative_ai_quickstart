# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 22:00
# @Author  : cfushn
# @Comments: 原始GAN定义文件，包含生成器、判别器、以及训练逻辑
# @Software: PyCharm

import timeit

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

"""
生成对抗网络 (Generative Adversarial Network, GAN)

优化:
1.优化器选择: 
    使用 Adam 优化器(通过自适应学习率和动量能够更好地处理梯度的波动性，使得训练更加稳定)
2.判别器与生成器结构优化: 
    在原始GAN中，生成器和判别器的全连接层深度较浅，这里增加了层数，
    同时使用LeakyReLU作为判别器的激活函数以避免“死神经元”问题，
    而在生成器中则使用 ReLU 和 Tanh 来生成更平滑的输出。
"""
class GAN:
    def __init__(self, dim_z, in_channels, epochs, lr_gen, lr_discr,
            train_loader, device="cuda"):
        self.__device = device
        self.__dim_z = dim_z
        self.__epochs = epochs
        self.__lr_gen = lr_gen
        self.__lr_discr = lr_discr
        self.__train_loader = train_loader
        self.__net_gen = _Generator(dim_z, in_channels).to(device)
        self.__net_discr = _Discriminator(in_channels).to(device)

    def train(self):
        device = self.__device
        dim_z = self.__dim_z
        net_gen = self.__net_gen
        net_discr = self.__net_discr
        epochs = self.__epochs
        lr_gen = self.__lr_gen
        lr_discr = self.__lr_discr
        train_loader = self.__train_loader

        # 优化器 (Adam优化器比SGD更稳健)
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=lr_gen,
                                   betas=(0.5, 0.999))
        optimizer_discr = optim.Adam(net_discr.parameters(), lr=lr_discr,
                                     betas=(0.5, 0.999))

        criterion = nn.BCELoss()  # 二分类损失函数

        # 生成固定的噪声向量，用于样本可视化
        z_fixed = torch.randn(100, dim_z, dtype=torch.float32).to(device)

        start_time = timeit.default_timer()

        for epoch in range(epochs):
            net_gen.train()
            net_discr.train()

            for batch_train_images, _ in train_loader:
                batch_size = batch_train_images.size(0)
                real_imgs = batch_train_images.to(device)
                z = torch.randn(batch_size, dim_z).to(device)
                fake_imgs = net_gen(z)

                # 判别器D的优化
                optimizer_discr.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                real_loss = criterion(net_discr(real_imgs), real_labels)
                fake_loss = criterion(net_discr(fake_imgs.detach()),
                                      fake_labels)
                discr_loss = (real_loss + fake_loss) / 2
                discr_loss.backward()
                optimizer_discr.step()

                # 生成器G的优化
                optimizer_gen.zero_grad()
                fake_loss_g = criterion(net_discr(fake_imgs),
                                          real_labels)  # 生成器希望判别器将其生成的图片判为真
                fake_loss_g.backward()
                optimizer_gen.step()

            # 输出每个epoch的损失
            print(
                    f"GAN [Epoch {epoch + 1}/{epochs}] [D loss: {discr_loss.item():.4f}] [G loss: {fake_loss_g.item():.4f}] [Time: {timeit.default_timer() - start_time:.3f}s]")

            # 每10个epoch保存一次生成的图像
            if (epoch + 1) % 10 == 0:
                net_gen.eval()
                with torch.no_grad():
                    fake_imgs = net_gen(z_fixed)
                    save_image(fake_imgs.data,
                               f"./output/epoch_{epoch + 1}.png", nrow=10,
                               normalize=True)

        return net_gen


# 生成器定义
class _Generator(nn.Module):
    def __init__(self, dim_z=128, out_channels=1):
        super(_Generator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(dim_z, 256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, 1024),
                nn.ReLU(True),
                nn.Linear(1024, out_channels * 32 * 32),
                nn.Tanh()  # Tanh将输出标准化到[-1, 1]，这与图像的像素值区间相符
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, 32, 32)  # 将生成的图像重新调整为32x32大小
        return img


# 判别器定义
class _Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(_Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(in_channels * 32 * 32, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()  # 使用Sigmoid输出一个0到1之间的概率值
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 展平图像
        validity = self.model(img_flat)
        return validity


__all__ = ["GAN"]
