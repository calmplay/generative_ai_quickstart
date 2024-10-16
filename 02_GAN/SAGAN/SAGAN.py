# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 22:20
# @Author  : cfushn
# @Comments: 自注意力生成对抗网络（SAGAN）的定义文件，包含生成器、判别器、自注意力模块，以及训练逻辑
# @Software: PyCharm

import timeit

import torch
import torch.nn as nn
from torchvision.utils import save_image


class SAGAN:
    def __init__(self, dim_z, in_channels, epochs, lr_gen, lr_discr, num_iter_d,
            train_loader, device="cuda"):
        self.__device = device
        self.__dim_z = dim_z
        self.__epochs = epochs
        self.__lr_gen = lr_gen
        self.__lr_discr = lr_discr
        self.__num_iter_d = num_iter_d
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
        num_iter_d = self.__num_iter_d
        train_loader = self.__train_loader

        optimizer_gen = torch.optim.Adam(net_gen.parameters(), lr=lr_gen,
                                         betas=(0.5, 0.999))
        optimizer_discr = torch.optim.Adam(net_discr.parameters(), lr=lr_discr,
                                           betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        # 固定噪声向量用于样本可视化
        z_fixed = torch.randn(100, dim_z, dtype=torch.float32).to(device)
        start_time = timeit.default_timer()

        for epoch in range(epochs):
            net_gen.train()
            net_discr.train()

            data_iter = iter(train_loader)
            batch_idx = 0

            while batch_idx < len(train_loader):
                '''更新判别器'''
                for _ in range(num_iter_d):
                    if batch_idx == len(train_loader):
                        break
                    batch_train_images, _ = next(data_iter)
                    batch_idx += 1
                    batch_size_current = batch_train_images.shape[0]
                    batch_train_images = batch_train_images.type(
                        torch.float32).to(device)

                    z = torch.randn(batch_size_current, dim_z,
                                    dtype=torch.float32).to(device)
                    gen_imgs = net_gen(z)
                    real_gt = torch.ones(batch_size_current, 1).to(device)
                    fake_gt = torch.zeros(batch_size_current, 1).to(device)

                    optimizer_discr.zero_grad()
                    prob_real = net_discr(batch_train_images)
                    prob_fake = net_discr(gen_imgs.detach())
                    real_loss = criterion(prob_real, real_gt)
                    fake_loss = criterion(prob_fake, fake_gt)
                    loss_d = (real_loss + fake_loss) / 2
                    loss_d.backward()
                    optimizer_discr.step()

                '''更新生成器'''
                optimizer_gen.zero_grad()
                z = torch.randn(batch_size_current, dim_z,
                                dtype=torch.float32).to(device)
                gen_imgs = net_gen(z)
                discr_out = net_discr(gen_imgs)
                loss_g = criterion(discr_out, real_gt)
                loss_g.backward()
                optimizer_gen.step()

            print(
                f"\r SAGAN [Epoch {epoch + 1}/{epochs}] [D loss: {loss_d.item():.3f}] "
                f"[G loss: {loss_g.item():.3f}] [Time: {timeit.default_timer() - start_time:.3f}]")

            if (epoch + 1) % 10 == 0:
                net_gen.eval()
                with torch.no_grad():
                    gen_imgs = net_gen(z_fixed)
                    save_image(gen_imgs.data, f"./output/sagan_{epoch + 1}.png",
                               nrow=10, normalize=True)

        return net_gen


class _SelfAttention(nn.Module):
    """自注意力模块"""

    def __init__(self, in_channels):
        super(_SelfAttention, self).__init__()
        self.in_channels = in_channels

        # 查询、键、值的1x1卷积
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))  # 用于学习的权重参数

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # 提取查询、键、值
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(
            0, 2, 1)  # (B, N, C//8)
        key = self.key_conv(x).view(batch_size, -1,
                                    width * height)  # (B, C//8, N)
        value = self.value_conv(x).view(batch_size, -1,
                                        width * height)  # (B, C, N)

        # 计算注意力图
        attention = torch.bmm(query, key)  # (B, N, N)
        attention = torch.softmax(attention, dim=-1)

        # 加权求和
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, width, height)

        # 原特征与加权特征的融合
        out = self.gamma * out + x
        return out


class _Generator(nn.Module):
    def __init__(self, dim_z, out_channels=1):
        super(_Generator, self).__init__()
        self.dim_z = dim_z
        self.out_channels = out_channels

        self.conv = nn.Sequential(
                nn.ConvTranspose2d(128, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                _SelfAttention(128),  # 在生成器中加入自注意力模块
                nn.ConvTranspose2d(128, out_channels, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.dim_z, 1, 1)
        output = self.conv(z)
        return output


class _Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2),
                _SelfAttention(256),  # 在判别器中加入自注意力模块
                nn.Conv2d(256, 1, 4, 1, 0),
                nn.Sigmoid()
        )

    def forward(self, img):
        output = self.conv(img)
        return output.view(-1, 1)


__all__ = ["SAGAN"]
