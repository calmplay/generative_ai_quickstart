# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 14:01
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch
import yaml


def __load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# 跟main在同一级
config = __load_config('gan_config.yaml')

GAN_TYPE = config['gan_type']
IMG_SIZE = config['img_size']
IN_CHANNELS = config['in_channels']
BATCH_SIZE = config['batch_size']
DIM_Z = config['dim_z']
LR = config['learning_rate']
BETA1 = config['beta1']
BETA2 = config['beta2']
EPOCHS = config['epochs']
NUM_ITER_D = config['num_iter_d']

device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"
print(f"--------------------use device: {device}--------------------")

#
# def get_device():
#     return device


__all__ = ['GAN_TYPE', 'IN_CHANNELS', 'IMG_SIZE', 'BATCH_SIZE', 'DIM_Z', 'LR',
           'BETA1', 'BETA2', 'EPOCHS', 'NUM_ITER_D', 'device']
