# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 13:38
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from .vanilla_gan import VanillaGAN


# from sagan import SAGAN
# from biggan import BigGAN

def get_gan(gan_type, lr, beta1, beta2):
    if gan_type == "vanilla":
        return VanillaGAN(lr, beta1, beta2)
    # elif gan_type == "sagan":
    #     return SAGAN(lr, beta1, beta2)
    # elif gan_type == "biggan":
    #     return BigGAN(lr, beta1, beta2)
    else:
        raise ValueError(f"Unsupported GAN type: {gan_type}")
