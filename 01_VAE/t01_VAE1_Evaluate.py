# -*- coding: utf-8 -*-
# @Time    : 2024/9/11 00:22
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import numpy as np
import torch
from mpmath import sqrtm
from torchvision.utils import save_image
from sklearn.metrics import precision_score, recall_score
from torchvision import models

from torch.nn import functional as F
from t01_VAE1 import LATENT_DIM, device, IMG_SIZE, VAE, BATCH_SIZE, \
    train_dataset, loss_function

model = VAE(LATENT_DIM).to(device)
model.load_state_dict(torch.load('vae_fashion_mnist.pth'))
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False)  # 加载测试数据集


def generate():
    model.eval()  # 将模型设置为评估模式，禁用dropout和batch normalization
    print(model)

    with torch.no_grad():  # 在评估阶段不计算梯度
        # Sample random latent vectors 随机采样
        z = torch.randn(64, LATENT_DIM).to(device)
        sample = model.decode(z).cpu()
        print("sample:", sample)
        save_image(sample.view(64, 1, IMG_SIZE, IMG_SIZE),
                   'sample_image2.png')
        print("Generated sample saved to 'sample_image.png'")


def evaluate1():
    model.eval()  # 将模型设置为评估模式，禁用dropout和batch normalization
    print(model)

    nll = 0  # 初始化负对数似然的累积值
    all_recon_x = []  # 存储所有重构的图像
    all_true_x = []  # 存储所有真实的图像
    with torch.no_grad():  # 在评估阶段不计算梯度
        for i, (data, _) in enumerate(test_loader):  # 遍历测试数据集
            data = data.to(device)  # 将数据移动到GPU或其他指定设备
            recon_batch, mu, logvar = model(data)  # 前向传播，获取重构图像和VAE参数

            # Compute NLL (negative log likelihood)
            # 计算负对数似然（NLL）
            nll += loss_function(recon_batch, data, mu, logvar).item()

            all_recon_x.append(recon_batch.cpu().numpy())  # 将重构的图像转移到CPU并保存
            all_true_x.append(data.cpu().numpy())  # 将真实图像转移到CPU并保存

    nll /= len(test_loader.dataset)  # 计算平均NLL
    print(f"Negative Log-Likelihood (NLL): {nll}")  # 输出负对数似然

    # # Compute Precision and Recall
    # # 计算精确率和召回率
    # recon_flat = np.concatenate(
    #         [x.reshape(x.shape[0], -1) for x in all_recon_x],
    #         axis=0)  # 将重构图像展开为一维向量
    # true_flat = np.concatenate([x.reshape(x.shape[0], -1) for x in all_true_x],
    #                            axis=0)  # 将真实图像展开为一维向量
    # # precision = precision_score(true_flat.round(), recon_flat.round(),
    # #                             average='macro')  # 计算精确率
    #
    # # Convert labels to binary (e.g., map -1.0 and -0.0 to 0)
    # true_flat_binary = np.where(true_flat < 0, 0, 1)
    # recon_flat_binary = np.where(recon_flat < 0, 0, 1)
    #
    # # Calculate precision scores for each class with zero_division handling
    # precisions = []
    # for i in range(true_flat_binary.shape[
    #                    1]):  # Assuming second dimension corresponds to classes
    #     precision = precision_score(true_flat_binary[:, i].round(),
    #                                 recon_flat_binary[:, i].round(),
    #                                 average='binary', zero_division=0)
    #     precisions.append(precision)
    # # recall = recall_score(true_flat.round(), recon_flat.round(),
    # #                       average='macro')  # 计算召回率
    #
    # # Calculate recall scores for each class with zero_division handling
    # recalls = []
    # for i in range(true_flat_binary.shape[
    #                    1]):  # Assuming second dimension corresponds to classes
    #     recall = recall_score(true_flat_binary[:, i].round(),
    #                           recon_flat_binary[:, i].round(),
    #                           average='binary', zero_division=0)
    #     recalls.append(recall)
    #
    # # print(f"Precision: {precision}")  # 输出精确率
    # print(f"Per-class precision scores: {precisions}")
    # print(f"Mean precision score: {sum(precisions) / len(precisions)}")
    # # print(f"Recall: {recall}")  # 输出召回率
    # print(f"Per-class recall scores: {recalls}")
    # print(f"Mean recall score: {sum(recalls) / len(recalls)}")
    #
    # # FID and IS calculation requires pretrained InceptionV3 model
    # # 计算FID和IS需要使用预训练的InceptionV3模型
    # inception_model = models.inception_v3(pretrained=True).to(
    #         device)  # 加载预训练的InceptionV3模型
    # inception_model.eval()  # 设置为评估模式
    #
    # # Function to get inception activations for images
    # def get_inception_activations(images, model):
    #     activations = []  # 用于存储InceptionV3的激活值
    #     for img in images:  # 遍历图像
    #         # img = F.interpolate(img.unsqueeze(0), size=(
    #         #     299, 299))  # 将图像调整到InceptionV3需要的尺寸299x299
    #
    #         img_tensor = torch.from_numpy(img)
    #         # Assuming `img_tensor` is the input tensor with shape [H, W] or [C, H, W]
    #         if img_tensor.ndim == 3:  # If img_tensor has 3 dimensions (C, H, W)
    #             img_tensor = img_tensor.unsqueeze(
    #                     0)  # Add batch dimension (N, C, H, W)
    #         elif img_tensor.ndim == 2:  # If img_tensor has 2 dimensions (H, W)
    #             img_tensor = img_tensor.unsqueeze(0).unsqueeze(
    #                     0)  # Add batch and channel dimensions (N, C, H, W)
    #
    #         # Perform interpolation to size (299, 299)
    #         img_tensor = F.interpolate(img_tensor, size=(299, 299))
    #
    #         # Assuming img_tensor is a grayscale image with shape (N, 1, H, W)
    #         if img_tensor.shape[1] == 1:  # Check if it's a single-channel image
    #             img_tensor = img_tensor.repeat(1, 3, 1,
    #                                            1)  # Duplicate the channel 3 times to make it (N, 3, H, W)
    #
    #         # Now img_tensor can be passed to the Inception model
    #         model = model.to(
    #             torch.float32)  # Ensure all model parameters are also in float32
    #         activation = model(img_tensor.to(torch.float32))
    #         activations.append(activation)  # 将激活值存储
    #     return np.array(activations)  # 返回激活值数组
    #
    # # Get activations for both real and generated images
    # # 获取真实图像和生成图像的Inception激活值
    # real_activations = get_inception_activations(all_true_x,
    #                                              inception_model)  # 获取真实图像的激活值
    # gen_activations = get_inception_activations(all_recon_x,
    #                                             inception_model)  # 获取生成图像的激活值
    #
    # # Compute Inception Score (IS)
    # # 计算Inception Score (IS)
    # p_yx = np.mean(real_activations, axis=0)  # 计算真实图像激活值的均值
    # p_y = np.mean(gen_activations, axis=0)  # 计算生成图像激活值的均值
    # inception_score = np.exp(
    #         np.mean(p_yx * np.log(p_yx / p_y)))  # 计算Inception Score
    #
    # print(f"Inception Score (IS): {inception_score}")  # 输出Inception Score
    #
    # # Compute FID
    # # 计算Fréchet Inception Distance (FID)
    # mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(
    #         real_activations, rowvar=False)  # 真实图像的均值和协方差
    # mu_gen, sigma_gen = np.mean(gen_activations, axis=0), np.cov(
    #         gen_activations, rowvar=False)  # 生成图像的均值和协方差
    # fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(
    #         sigma_real + sigma_gen - 2 * sqrtm(
    #                 sigma_real.dot(sigma_gen)))  # 计算FID
    #
    # print(f"Fréchet Inception Distance (FID): {fid}")  # 输出FID
    #
    # # PPL (Perceptual Path Length)
    # # 感知路径长度
    # def perceptual_path_length(z1, z2, model, steps=10):
    #     interpolated_imgs = []  # 用于存储插值生成的图像
    #     for alpha in np.linspace(0, 1, steps):  # 在隐空间中进行插值
    #         z_interp = z1 * (1 - alpha) + z2 * alpha  # 计算插值后的隐向量
    #         interpolated_imgs.append(model.decode(z_interp))  # 将隐向量解码为图像
    #     # Calculate perceptual differences (here using L2 distance)
    #     # 计算感知差异（使用L2距离）
    #     differences = [
    #         F.mse_loss(interpolated_imgs[i], interpolated_imgs[i + 1]) for i in
    #         range(steps - 1)]  # 计算插值图像之间的均方误差
    #     return np.mean(differences)  # 返回均值作为PPL
    #
    # z1, z2 = torch.randn(1, latent_dim).to(device), torch.randn(1,
    #                                                             latent_dim).to(
    #         device)  # 随机生成两个隐向量
    # ppl = perceptual_path_length(z1, z2, model)  # 计算PPL
    #
    # print(f"Perceptual Path Length (PPL): {ppl}")  # 输出PPL


if __name__ == "__main__":
    # generate()
    evaluate1()
