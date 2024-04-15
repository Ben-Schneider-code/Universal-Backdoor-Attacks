
import torch
import torch.nn as nn
from torch import cdist


def rbf_kernel(x, y, sigma=1):
    return torch.exp(-cdist(x, y) ** 2 / (2 * sigma ** 2))


def compute_hsic(latent_space_1, latent_space_2, kernel_func=rbf_kernel):
    K_1 = kernel_func(latent_space_1, latent_space_1)
    K_2 = kernel_func(latent_space_2, latent_space_2)

    K_1_centered = K_1 - K_1.mean(dim=0) - K_1.mean(dim=1).unsqueeze(1) + K_1.mean()
    K_2_centered = K_2 - K_2.mean(dim=0) - K_2.mean(dim=1).unsqueeze(1) + K_2.mean()

    hsic = (K_1_centered * K_2_centered).sum() / (latent_space_1.size(0) ** 2)
    return 1 - hsic


class HSICLoss(torch.nn.Module):
    def __init__(self,):
        super(HSICLoss, self).__init__()

    def forward(self, embeddings1, embeddings2):
        return 1 - compute_hsic(embeddings1, embeddings2)

