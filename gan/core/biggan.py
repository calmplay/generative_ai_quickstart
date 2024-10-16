# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 18:14
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

"""
BigGAN
(Large Scale GAN Training for High Fidelity Natural Image Synthesis, BigGAN)
关键优化：
1.自适应归一化（AdaIN）：在生成器中使用标签嵌入将噪声向量与条件标签结合，
  生成器通过自适应实例归一化调整特征图，使得生成的图像质量和类别一致性更好。
2.类条件生成：通过嵌入类标签的方式，允许生成器在类别条件下生成多样化的图像。
3.较大的模型尺寸：相比其他GANs，BigGAN通过更大的网络深度和更高分辨率图像的生成增强了生成图片
  的多样性和质量
"""
