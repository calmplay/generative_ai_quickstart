# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 18:12
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

"""
谱归一化生成对抗网络
(Spectral Normalization GAN, SNGAN)

主要区别：
1. 谱归一化用于判别器卷积层,控制 Lipschitz 常数，帮助 GAN 在训练时更稳定。
2. 生成器结构基本与 DCGAN 相同。
"""