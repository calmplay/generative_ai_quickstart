# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 13:45
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gan.config.config import IMG_SIZE, BATCH_SIZE


def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True,
                                    transform=transform)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE,
                            shuffle=True)
    return dataloader
