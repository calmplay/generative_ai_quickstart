# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 13:24
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os
import timeit
from abc import ABC
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from gan.config.config import device, GAN_TYPE

"""
GAN模型的基类(抽象类)
"""


class BaseGAN(ABC):
    def __init__(self, generator, discriminator, dim_z, lr, beta1, beta2):
        self.generator = generator
        self.discriminator = discriminator
        self.dim_z = dim_z
        self.lr = lr
        # 数据加载器,判别器批次内迭代次数,这两个变量训练时指定
        self.dataloader = None
        self.num_iter_d = None
        # 分别创建生成器和判别器的优化器
        self.optim_g = optim.Adam(self.generator.parameters(), weight_decay=0,
                                  lr=lr, betas=(beta1, beta2))
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  weight_decay=0, lr=lr, betas=(beta1, beta2))
        # 分类损失函数,使用交叉熵
        self.criterion = nn.BCELoss()

    def train_discriminator(self, real_imgs, fake_imgs) -> (
            torch.Tensor, torch.Tensor, torch.Tensor):
        batch_size = real_imgs.shape[0]
        for _ in range(self.num_iter_d):
            # 样本标签: 真实为1,虚假为0; 用于定义损失函数
            real_gt = torch.ones(batch_size, 1).to(device)
            fake_gt = torch.zeros(batch_size, 1).to(device)
            # 计算判别器损失
            self.optim_d.zero_grad()
            prob_real = self.discriminator(real_imgs)
            prob_fake = self.discriminator(fake_imgs.detach())
            real_loss = self.criterion(prob_real, real_gt)
            fake_loss = self.criterion(prob_fake, fake_gt)
            loss_d = (real_loss + fake_loss) / 2
            # 反向传播
            loss_d.backward()
            # 更新参数
            self.optim_d.step()
            return loss_d, prob_real, prob_fake

    def train_generator(self, fake_imgs):
        batch_size = fake_imgs.shape[0]
        self.optim_g.zero_grad()
        # 采样高斯噪声
        z = torch.randn(batch_size, self.dim_z,
                        dtype=torch.float32).to(device)
        # 生成一批虚假图像
        gen_imgs = self.generator(z)
        # 判别器的输出
        discr_out = self.discriminator(gen_imgs)
        # 计算生成器的损失函数
        real_gt = torch.ones(batch_size, 1).to(device)
        loss_g = self.criterion(discr_out, real_gt)
        # 反向传播
        loss_g.backward()
        # 更新参数
        self.optim_g.step()
        return loss_g

    def train(self, dataloader, epochs, num_iter_d):
        self.num_iter_d = num_iter_d
        # 生成10个固定的噪声向量,用于训练中的样本可视化
        n_row = 10
        z_fixed = torch.randn(n_row ** 2, self.dim_z, dtype=torch.float32).to(
                device)

        start_time = timeit.default_timer()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"./output/{GAN_TYPE}/{timestamp}/"
        os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹

        # 循环训练
        # (每次循环,更新NUM_ITER_D次判别器,然后更新1次生成器)
        for epoch in range(epochs):

            # 将生成器和判别器设置为训练模式
            self.generator.train()
            self.discriminator.train()

            for real_data in dataloader:
                # Random noise for generator
                real_imgs = real_data[0].to(device)
                z = torch.randn(real_imgs.shape[0], self.dim_z,
                                dtype=torch.float32).to(device)
                fake_imgs = self.generator(z)
                [loss_d, prob_real, prob_fake] = (
                    self.train_discriminator(real_imgs, fake_imgs))
                loss_g = (
                    self.train_generator(fake_imgs))

            # 打印日志
            print("\r DCGAN：[Epoch %d/%d] [D loss： %.3f] [G loss： %.3f] "
                  "[D pr ob real：%.3f] [D prob fake：%.3f] [Time： %.3f]"
                  % (epoch + 1, epochs, loss_d.item(),
                     loss_g.item(),
                     prob_real.mean().item(),
                     prob_fake.mean().item(),
                     timeit.default_timer() - start_time))

            # 保存大概10个样本, (如果不足10个epoch,则每个都打印)
            if (epoch + 1) % (epoch // 10 if epochs >= 10 else 1) == 0:
                self.generator.eval()
                with torch.no_grad():
                    gen_imgs = self.generator(z_fixed)
                    gen_imgs = gen_imgs.detach()
                    save_image(gen_imgs.data,
                               f"{output_dir}/{epoch + 1}.png",
                               nrow=10, normalize=True)
        # end for epoch

    def generate(self, noise):
        return self.generator(noise)
