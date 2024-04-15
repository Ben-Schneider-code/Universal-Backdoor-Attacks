import torch
import torch.nn.functional as F


class AttentionLoss(torch.nn.Module):
    """ AT with sum of absolute values with power p
        Code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
	"""
    def __init__(self, p):
        super(AttentionLoss, self).__init__()
        self.p = p  # the exponential power (to focus the attention).

    def forward(self, fm_s, fm_t):
        """ Assumes feature maps with shape [nchw]
        """
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am
