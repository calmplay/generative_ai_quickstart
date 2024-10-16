# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 23:20
# @Author  : cfushn
# @Comments: BigGAN 模型的定义文件，包含生成器、判别器及其优化的设计
# @Software: PyCharm

import timeit

import torch
import torch.nn as nn
from torchvision.utils import save_image


class BigGAN:
    def __init__(self, dim_z, in_channels, n_classes, epochs, lr_gen, lr_discr,
            num_iter_d, train_loader, device="cuda"):
        self.__device = device
        self.__dim_z = dim_z
        self.__epochs = epochs
        self.__lr_gen = lr_gen
        self.__lr_discr = lr_discr
        self.__num_iter_d = num_iter_d
        self.__train_loader = train_loader
        self.__net_gen = _Generator(dim_z, in_channels, n_classes).to(device)
        self.__net_discr = _Discriminator(in_channels, n_classes).to(device)

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
                                         betas=(0.0, 0.999))
        optimizer_discr = torch.optim.Adam(net_discr.parameters(), lr=lr_discr,
                                           betas=(0.0, 0.999))
        criterion_adv = nn.BCELoss()  # 对抗损失
        criterion_aux = nn.CrossEntropyLoss()  # 辅助分类损失

        # 固定噪声向量和标签用于样本可视化
        z_fixed = torch.randn(100, dim_z, dtype=torch.float32).to(device)
        fixed_labels = torch.randint(0, 10, (100,)).to(device)
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
                    batch_train_images, real_labels = next(data_iter)
                    batch_idx += 1
                    batch_size_current = batch_train_images.shape[0]
                    batch_train_images = batch_train_images.type(
                        torch.float32).to(device)
                    real_labels = real_labels.to(device)

                    z = torch.randn(batch_size_current, dim_z,
                                    dtype=torch.float32).to(device)
                    fake_labels = torch.randint(0, 10,
                                                (batch_size_current,)).to(
                        device)
                    gen_imgs = net_gen(z, fake_labels)

                    real_gt = torch.ones(batch_size_current, 1).to(device)
                    fake_gt = torch.zeros(batch_size_current, 1).to(device)

                    optimizer_discr.zero_grad()

                    # 判别器对真实图片的损失
                    prob_real, aux_real = net_discr(batch_train_images)
                    real_loss_adv = criterion_adv(prob_real, real_gt)
                    real_loss_aux = criterion_aux(aux_real, real_labels)
                    real_loss = real_loss_adv + real_loss_aux

                    # 判别器对生成图片的损失
                    prob_fake, aux_fake = net_discr(gen_imgs.detach())
                    fake_loss_adv = criterion_adv(prob_fake, fake_gt)
                    loss_d = (real_loss + fake_loss_adv) / 2
                    loss_d.backward()
                    optimizer_discr.step()

                '''更新生成器'''
                optimizer_gen.zero_grad()
                z = torch.randn(batch_size_current, dim_z,
                                dtype=torch.float32).to(device)
                fake_labels = torch.randint(0, 10, (batch_size_current,)).to(
                    device)
                gen_imgs = net_gen(z, fake_labels)
                prob_fake, aux_fake = net_discr(gen_imgs)
                loss_g_adv = criterion_adv(prob_fake, real_gt)
                loss_g_aux = criterion_aux(aux_fake, fake_labels)
                loss_g = loss_g_adv + loss_g_aux
                loss_g.backward()
                optimizer_gen.step()

            print(
                f"\r BigGAN [Epoch {epoch + 1}/{epochs}] [D loss: {loss_d.item():.3f}] "
                f"[G loss: {loss_g.item():.3f}] [Time: {timeit.default_timer() - start_time:.3f}]")

            if (epoch + 1) % 10 == 0:
                net_gen.eval()
                with torch.no_grad():
                    gen_imgs = net_gen(z_fixed, fixed_labels)
                    save_image(gen_imgs.data,
                               f"./output/biggan_{epoch + 1}.png", nrow=10,
                               normalize=True)

        return net_gen


class _Generator(nn.Module):
    def __init__(self, dim_z, out_channels, n_classes):
        super(_Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, dim_z)
        self.conv = nn.Sequential(
                nn.ConvTranspose2d(dim_z, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, out_channels, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, z, labels):
        labels_embed = self.label_embedding(labels)
        z = z + labels_embed  # 嵌入标签
        z = z.view(-1, z.size(1), 1, 1)
        return self.conv(z)


class _Discriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(_Discriminator, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
        )

        self.adv_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Conv2d(512, n_classes, 4, 1, 0))

    def forward(self, img):
        features = self.conv(img)
        adv_output = self.adv_layer(features).view(-1, 1)
        aux_output = self.aux_layer(features).view(-1, 10)
        return adv_output, aux_output


__all__ = ["BigGAN"]
