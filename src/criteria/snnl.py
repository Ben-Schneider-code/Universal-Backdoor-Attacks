import numpy as np

import torch


class SNNLLoss(torch.nn.Module):
    def __init__(self,):
        super(SNNLLoss, self).__init__()

    def forward(self, x, y, t=1):
        """
        Compute the soft nearest neighbor loss using the given data_cleaning x, the
            labels y, and the temperature t
        """
        def pairwise_euclidean_distance(a, b):
            ba = a.size()[0]
            bb = b.size()[0]
            sqr_norm_a = torch.reshape(torch.sum(torch.pow(a, 2), dim=1), (1, ba))
            sqr_norm_b = torch.reshape(torch.sum(torch.pow(b, 2), dim=1), (bb, 1))
            inner_prod = torch.matmul(b, torch.transpose(a, dim0=0, dim1=1))
            tile1 = torch.Tensor.repeat(sqr_norm_a, bb, 1)
            tile2 = torch.Tensor.repeat(sqr_norm_b, 1, ba)
            return tile1 + tile2 - 2 * inner_prod

        x = torch.reshape(x, (len(x), -1))

        distance = pairwise_euclidean_distance(x, x)
        exp_distance = torch.exp(-(distance / (t + 1e-7)))
        f = exp_distance - torch.tensor(np.eye(x.size()[0])).to(x.device)
        f[f >= 1] = 1
        f[f <= 0] = 0
        pick_probability = f / (1e-7 + torch.sum(f, dim=1, keepdim=True))

        y_shape = y.size()
        same_label_mask = torch.eq(y, torch.reshape(y, (y_shape[0], 1)))

        masked_pick_probability = pick_probability * same_label_mask
        sum_masked_probability = torch.sum(masked_pick_probability, dim=1, keepdim=False)

        return torch.mean(-torch.log(1e-7 + sum_masked_probability))