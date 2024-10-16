# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 17:53
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch
from torch import nn

from gan.config.config import device
from .base_gan import BaseGAN

"""
自注意力生成对抗网络
(Self-Attention GAN，DCGAN)

主要改进:
    在生成器和判别器中都加入了自注意力机制，用于捕捉图像中局部的长程依赖关系，
    这在生成更高分辨率和细节丰富的图像时尤为有效
"""


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


class SAGAN(BaseGAN):
    def __init__(self, in_channels, dim_z, lr, beta1, beta2):
        generator = _Generator(dim_z, in_channels).to(device)
        discriminator = _Discriminator(in_channels).to(device)
        super(SAGAN, self).__init__(generator, discriminator, dim_z, lr, beta1,
                                    beta2)


__all__ = ["SAGAN"]
