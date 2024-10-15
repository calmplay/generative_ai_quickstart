# -*- coding: utf-8 -*-
# @Time    : 2024/9/11 19:47
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch
import torch.utils.data
from torch import nn


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(VAE, self).__init__()
        # 输⼊图⽚的通道数、⾼度、宽度
        self.in_channel, self.img_h, self.img_w = img_size
        # 隐藏编码Z的维度
        self.latent_dim = latent_dim

        # 推断⽹络（i.e. Encoder）
        # 这是一个卷积模块
        self.encoder = nn.Sequential(
                # 输入维度(n,1,32,32)->tensor(batch_size, in_channel, img_h, img_w)
                # 这里的in_channel=1, 对应灰度图像
                # output_size = (input_size-kernel_size+2*padding)/stride + 1
                nn.Conv2d(in_channels=self.in_channel, out_channels=32,
                          kernel_size=3, stride=2, padding=1),  # h=h//2
                nn.BatchNorm2d(32),  # 二维批标准化（Batch Normalization）层
                nn.LeakyReLU(),
                # 特征图维度（n,32,16,16）
                nn.Conv2d(32, 64, 3, 2, 1),  # h=h//2
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                # 特征图维度(n,64,8,8)
                nn.Conv2d(64, 128, 3, 2, 1),  # h=h//2
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                # 特征图维度(n,128,4,4)
                nn.Conv2d(128, 256, 3, 2, 1),  # h=h//2
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                # 特征图维度(n,256,2,2)
                nn.Conv2d(256, 512, 3, 2, 1),  # h=h//2
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                # 特征图维度(n,512,1,1)
        )
        # 全连接层: 将特征向量转化为分布均值\mu
        self.fc_mu = nn.Linear(512, self.latent_dim)
        # 全连接层: 将特征向量转化为分布方差的对数log(var)
        self.fc_var = nn.Linear(512, self.latent_dim)

        # 生成网络(i.e.Decoder)
        # 全连接层
        self.decoder_input = nn.Linear(self.latent_dim, 512)
        # 转置卷积模块
        self.decoder = nn.Sequential(
                # 输入维度(n,512,h,w)
                # ConvTranspose2d是一个二维反卷积层,增加空间维度
                # output_size=stride×(input_size−1)+kernel_size−2×padding+output_padding
                nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1),  # h=2h
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                # 特征图维度(n,256,2h,2w)
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # h=2h
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                # 特征图维度(n,128,4h,4w)
                nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),  # h=2h
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                # 特征图维度(n,64,8h,8w)
                nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),  # h=2h
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                # 特征图维度(n,32,16h,16w)
                nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),  # h=2h
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                # 特征图维度(n,32,32h,32w)
                nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                          kernel_size=3, stride=1, padding=1),  # h=h
                nn.Tanh()
                # 输出维度(n,1,32h,32w)
        )

    def encode(self, x):
        # encoder结构,(n,1,32,32)->(n,512,1,1)
        result = self.encoder(x)
        result = torch.flatten(result, 1)  # 将特征层转化为特征向量(n,521,1,1)->(n,512)
        mu = self.fc_mu(result)  # 计算分布均值\mu, (n,512)->(n,128)
        log_var = self.fc_var(result)  # 计算分布方差的对数log(var), (n,512)->(n,128)
        return [mu, log_var]

    def decode(self, z):
        # 将采样变量Z转化为特征向量, 再转化为特征层 (n,128)->(n,512)->(n,512,1,1)
        x_hat = self.decoder_input(z).view(-1, 512, 1, 1)
        # decoder结构 (n,512,1,1)->(n,1,32,32)
        x_hat = self.decoder(x_hat)
        return x_hat

    # 再参数化
    @staticmethod
    def re_parameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)  # 分布标准差std
        eps = torch.randn_like(std)  # 从标准正态分布中采样, (n,128)
        sample = mu + std * eps
        return sample

    # 前向传播
    def forward(self, x):
        mu, log_var = self.encode(x)  # 经过encode,得到分布的均值mu和方差对数log_var
        z = self.re_parameterize(mu, log_var)  # 经过再参数,得到隐变量Z
        x_hat = self.decode(z)  # 经过decode,得到生成样本x_hat
        return [x_hat, x, mu, log_var]

    # 生成
    def sample(self, n, device):
        # 从标准正态分布中采样得到n个隐变量Z,长度为latent_dim
        z = torch.randn(n, self.latent_dim).to(device)
        images = self.decode(z)  # 经过解码过程,得到生成样本Y
        return images
