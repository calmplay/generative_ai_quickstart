# -*- coding: utf-8 -*-
# @Time    : 2024/9/11 17:11
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

"""
深度卷积⽣成对抗⽹络（Deep Convolutional Generative Adversarial Network，DCGAN）
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
