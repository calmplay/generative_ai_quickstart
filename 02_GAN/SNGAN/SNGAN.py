# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 22:00
# @Author  : cfushn
# @Comments:
# @Software: PyCharm

import timeit

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image

"""
谱归一化生成对抗网络 (Spectral Normalization GAN, SNGAN)

主要区别：
1. 谱归一化用于判别器卷积层,控制 Lipschitz 常数，帮助 GAN 在训练时更稳定。
2. 生成器结构基本与 DCGAN 相同。
"""


class SNGAN:
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

        # 生成器和判别器的优化器
        optimizer_gen = torch.optim.Adam(net_gen.parameters(), weight_decay=0,
                                         lr=lr_gen, betas=(0.5, 0.999))
        optimizer_discr = torch.optim.Adam(net_discr.parameters(),
                                           weight_decay=0,
                                           lr=lr_discr, betas=(0.5, 0.999))

        # BCELoss用于判别器的分类损失
        criterion = nn.BCELoss()

        # 固定噪声向量,用于训练过程中的样本可视化
        n_row = 10
        z_fixed = torch.randn(n_row ** 2, dim_z, dtype=torch.float32).to(device)

        start_time = timeit.default_timer()

        for epoch in range(epochs):
            net_gen.train()
            net_discr.train()
            data_iter = iter(train_loader)
            batch_idx = 0

            while batch_idx < len(train_loader):
                '''更新判别器D'''
                for _ in range(num_iter_d):
                    if batch_idx == len(train_loader):
                        break
                    (batch_train_images, _) = next(data_iter)
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

                '''更新生成器G'''
                optimizer_gen.zero_grad()
                z = torch.randn(batch_size_current, dim_z,
                                dtype=torch.float32).to(device)
                gen_imgs = net_gen(z)
                discr_out = net_discr(gen_imgs)
                loss_g = criterion(discr_out, real_gt)
                loss_g.backward()
                optimizer_gen.step()

            print("\r SNGAN：[Epoch %d/%d] [D loss： %.3f] [G loss： %.3f] "
                  "[D prob real：%.3f] [D prob fake：%.3f] [Time： %.3f]"
                  % (epoch + 1, epochs, loss_d.item(), loss_g.item(),
                     prob_real.mean().item(), prob_fake.mean().item(),
                     timeit.default_timer() - start_time))

            if (epoch + 1) % 10 == 0:
                net_gen.eval()
                with torch.no_grad():
                    gen_imgs = net_gen(z_fixed)
                    gen_imgs = gen_imgs.detach()
                    save_image(gen_imgs.data,
                               "./output/{}.png".format(epoch + 1),
                               nrow=10, normalize=True)
        return net_gen


class _Generator(nn.Module):
    def __init__(self, dim_z=128, out_channels=1):
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
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, out_channels, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, input_):
        input_ = input_.view(-1, self.dim_z, 1, 1)
        output = self.conv(input_)
        return output


class _Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1, 1, 1, 0)),
                nn.Sigmoid()
        )

    def forward(self, input_):
        output = self.conv(input_)
        return output.view(-1, 1)


__all__ = ["SNGAN"]
