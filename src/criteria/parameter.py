import torch
import torch.nn as nn


class ParameterLoss(torch.nn.Module):
    def __init__(self,):
        super(ParameterLoss, self).__init__()

    def forward(self, model1, model2):
        """
        Computes the mean squared error (MSE) between the parameters of two neural networks.

        Args:
            model1 (nn.Module): The first neural network.
            model2 (nn.Module): The second neural network.

        Returns:
            loss (torch.Tensor): The MSE loss between the parameters of the two networks.
        """
        loss = 0
        n_params = 0

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            loss += torch.sum((p1 - p2) ** 2)
            n_params += p1.numel()

        loss /= n_params
        return loss
