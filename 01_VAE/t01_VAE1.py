# -*- coding: utf-8 -*-
# @Time    : 2024/9/10 22:53
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 256
LATENT_DIM = 64
IMG_SIZE = 32
LR = 1e-4  # 1 * 10^(-4) = 0.0001
WEIGHT_DECAY = 1e-4  # 权重衰减参数
ALPHA = 1e-4  # lambda的值
EPOCHS = 400

# device
device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"

# data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),  # 将图像缩放至32✖32分辨率
    transforms.ToTensor(),  # 像素值[0,255]归一化为[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 像素值从[0,1]归一化为[-1,1]
])
train_dataset = datasets.FashionMNIST(root="../data", train=True,
                                      download=True,
                                      transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # mu layer
        self.fc22 = nn.Linear(512, latent_dim)  # logvar layer
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, IMG_SIZE * IMG_SIZE)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_SIZE * IMG_SIZE))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(LATENT_DIM).to(device)


# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, IMG_SIZE * IMG_SIZE), reduction='sum')
    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    # Model, optimizer, and training loop
    # model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")


if __name__ == "__main__":
    # Train the model
    for epoch in range(1, EPOCHS + 1):
        train(epoch)

    # Save the trained model
    torch.save(model.state_dict(), 'vae_fashion_mnist.pth')











