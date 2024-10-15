# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 20:18
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import timeit

import torch.optim
from torch import nn
from torchvision.utils import save_image

"""
深度卷积⽣成对抗⽹络
(Deep Convolutional Generative Adversarial Network，DCGAN)

与原始GAN相比,损失函数与训练算法差别不大, 主要改进在于:
1.利用卷积进行上/下采样
2.批归一化
3.激活函数改进:
    对于生成器:使用ReLU激活,仅在最后一层使用tanh (替代Maxout和sigmoid);
    对于判别器:使用LeakyReLU激活 (替代Maxout)
4.避免使用全连接层
(在此之前的原始GAN和cGAN模型所采⽤的⽹络架构均设计为仅包含三层全连接层的全连接⽹络。
对高维图像数据拟合能力不足.)
"""


class DCGAN:
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

        # 分别定义生成器和判别器的优化器
        optimizer_gen = torch.optim.Adam(net_gen.parameters(), weight_decay=0,
                                         lr=lr_gen, betas=(0.5, 0.999))
        optimizer_discr = torch.optim.Adam(net_discr.parameters(),
                                           weight_decay=0,
                                           lr=lr_discr, betas=(0.5, 0.999))

        # 判别器的分类损失
        criterion = nn.BCELoss()

        # 生成10个固定的噪声向量,用于训练中的样本可视化
        n_row = 10
        z_fixed = torch.randn(n_row ** 2, dim_z, dtype=torch.float32).to(device)

        start_time = timeit.default_timer()

        # 训练循环
        for epoch in range(epochs):
            # 将生成器和判别器设置为训练模式
            net_gen.train()
            net_discr.train()

            # 将train_loader转化为迭代器
            data_iter = iter(train_loader)
            batch_idx = 0

            """每次循环,更新1次生成器,更新NUM_ITER_D次判别器"""
            while batch_idx < len(train_loader):
                '''更新判别器D'''
                for _ in range(num_iter_d):
                    if batch_idx == len(train_loader):
                        break
                    # 采样一批真实图像
                    (batch_train_images, _) = next(data_iter)
                    batch_idx += 1
                    # 计算当前batch的样本量
                    batch_size_current = batch_train_images.shape[0]
                    # 将batch中的图像转换成float并移动至指定device上
                    batch_train_images = batch_train_images.type(
                            torch.float32).to(device)
                    # 采样高斯噪声
                    z = torch.randn(batch_size_current, dim_z,
                                    dtype=torch.float32).to(device)
                    # 生成一批假图像
                    gen_imgs = net_gen(z)
                    # 样本标签: 真实为1,虚假为0; 用于定义损失函数
                    real_gt = torch.ones(batch_size_current, 1).to(device)
                    fake_gt = torch.zeros(batch_size_current, 1).to(device)
                    # 计算判别器损失
                    optimizer_discr.zero_grad()
                    prob_real = net_discr(batch_train_images)
                    prob_fake = net_discr(gen_imgs.detach())
                    real_loss = criterion(prob_real, real_gt)
                    fake_loss = criterion(prob_fake, fake_gt)
                    loss_discr = (real_loss + fake_loss) / 2
                    # 反向传播
                    loss_discr.backward()
                    # 更新参数
                    optimizer_discr.step()
                # end for _

                '''更新生成器G'''
                optimizer_gen.zero_grad()
                # 采样高斯噪声
                z = torch.randn(batch_size_current, dim_z,
                                dtype=torch.float32).to(device)
                # 生成一批虚假图像
                gen_imgs = net_gen(z)
                # 判别器的输出
                discr_out = net_discr(gen_imgs)
                # 计算生成器的损失函数
                loss_gen = criterion(discr_out, real_gt)
                # 反向传播
                loss_gen.backward()
                # 更新参数
                optimizer_gen.step()
            # end while

            print("\r DCGAN：[Epoch %d/%d] [D loss： %.3f] [G loss： %.3f] "
                  "[D pr ob real：%.3f] [D prob fake：%.3f] [Time： %.3f]"
                  % (epoch + 1, epochs, loss_discr.item(), loss_gen.item(),
                     prob_real.mean().item(), prob_fake.mean().item(),
                     timeit.default_timer() - start_time))

            # 每10个epoch生成100个样本用于可视化
            if (epoch + 1) % 10 == 0:
                net_gen.eval()
                with torch.no_grad():
                    gen_imgs = net_gen(z_fixed)
                    gen_imgs = gen_imgs.detach()
                    save_image(gen_imgs.data,
                               "./output/{}.png".format(epoch + 1),
                               nrow=10, normalize=True)
        # end for epoch

        return net_gen


class _Generator(nn.Module):
    def __init__(self, dim_z=128, out_channels=1):
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


__all__ = ["DCGAN"]
