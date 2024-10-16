# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 14:26
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from torch import nn

from gan.config.config import device
from .base_gan import BaseGAN

"""
深度卷积⽣成对抗⽹络
(Deep Convolutional GAN，DCGAN)

与原始GAN(VanillaGAN)相比,损失函数与训练算法差别不大, 主要改进在于:
1.利用卷积进行上/下采样
2.批归一化
3.激活函数改进:
    对于生成器:使用ReLU激活,仅在最后一层使用tanh (替代Maxout和sigmoid);
    对于判别器:使用LeakyReLU激活 (替代Maxout)
4.避免使用全连接层
  (在此之前的原始GAN和cGAN模型所采⽤的⽹络架构均设计为仅包含三层全连接层的全连接⽹络。
   对高维图像数据拟合能力不足.)
"""


class _Generator(nn.Module):
    def __init__(self, dim_z, out_channels=1):
        super(_Generator, self).__init__()
        self.dim_z = dim_z
        self.out_channels = out_channels

        # 卷积模块
        self.conv = nn.Sequential(
                # 输入维度(n, 128, 1, 1)
                nn.ConvTranspose2d(128, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                # 特征图维度(n, 512, 4, 4)
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # 特征图维度(n, 256, 8, 8)
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 特征图维度(n, 128, 16, 16)
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 特征图维度(n, 64, 32, 32)
                nn.ConvTranspose2d(64, out_channels, 3, 1, 1),
                nn.Tanh()
                # 输出维度(n, 1, 32, 32)
        )

    def forward(self, input_):
        # New view of array with the same data
        input_ = input_.view(-1, self.dim_z, 1, 1)
        output = self.conv(input_)
        return output


class _Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(_Discriminator, self).__init__()
        self.in_channels = in_channels

        # 卷积模块
        self.conv = nn.Sequential(
                # 输入维度(n, 1, 32, 32)
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 特征图维度(n, 64, 16, 16)
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 特征图维度(n, 128, 8, 8)
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 特征图维度(n, 256, 4, 4)
                nn.Conv2d(256, 512, 4, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                # 特征图维度(n, 512, 1, 1)
                nn.Conv2d(512, 1, 1, 1, 0),
                nn.Sigmoid()
                # 输出维度(n, 1, 1, 1)
        )

    def forward(self, input_):
        output = self.conv(input_)
        return output.view(-1, 1)


class DCGAN(BaseGAN):
    def __init__(self, in_channels, dim_z, lr, beta1, beta2):
        generator = _Generator(dim_z, in_channels).to(device)
        discriminator = _Discriminator(in_channels).to(device)
        super(DCGAN, self).__init__(generator, discriminator, dim_z, lr, beta1,
                                    beta2)


__all__ = ["DCGAN"]
