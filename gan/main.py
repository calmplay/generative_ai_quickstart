# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 13:46
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from gan.common.data_loader import get_data_loader
from gan.config.config import *
from gan.core.dcgan import DCGAN
from gan.core.sagan import SAGAN

if __name__ == '__main__':

    dcgan = DCGAN(IN_CHANNELS, DIM_Z, LR, BETA1, BETA2)
    dcgan.train(get_data_loader(), EPOCHS, NUM_ITER_D)

    # sagan = SAGAN(IN_CHANNELS, DIM_Z, LR, BETA1, BETA2)
    # sagan.train(get_data_loader(), EPOCHS, NUM_ITER_D)

