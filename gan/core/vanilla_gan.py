# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 13:26
# @Author  : cfushn
# @Comments:
# @Software: PyCharm

"""
原始GAN (Vanilla GAN)
"""

import torch.nn as nn

from .base_gan import BaseGAN


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, 1024),
                nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class VanillaGAN(BaseGAN):
    def __init__(self, dim_z, lr, beta1, beta2):
        generator = Generator()
        discriminator = Discriminator()
        super(VanillaGAN, self).__init__(generator, discriminator, dim_z, lr,
                                         beta1, beta2)
