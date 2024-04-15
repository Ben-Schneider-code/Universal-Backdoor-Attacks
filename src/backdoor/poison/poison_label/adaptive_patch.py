import random
from copy import deepcopy
from typing import Tuple

import torch
from math import sqrt
from torchvision.transforms import Resize

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
from src.utils.special_images import image_to_tensor


class AdaptivePatch(Backdoor):
    """ Latent Separability Paper
    https://openreview.net/pdf?id=_wSHsgrVali
    """

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args=env_args)

    def requires_preparation(self) -> bool:
        return True

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies an adaptively patched trigger
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        # Obtain shape of the mark
        m_x, m_y = int(self.backdoor_args.mark_offset_x * w), int(self.backdoor_args.mark_offset_y * h)
        m_w, m_h = int(self.backdoor_args.mark_width * w), int(self.backdoor_args.mark_height * h)

        # Load and resize the mark
        if self.backdoor_args.mark is None:
            self.backdoor_args.mark = image_to_tensor(self.backdoor_args.mark_path).unsqueeze(0)

        mark = deepcopy(self.backdoor_args.mark + 1e-7).repeat_interleave(len(x), 0)
        num_patches = self.backdoor_args.adaptive_blend_num_patches
        l = int(mark.shape[-1] / int(sqrt(num_patches)))

        if self._train:        # only mask out during training
            for i in range(len(x)):
                for w in range(int(sqrt(num_patches))):
                    for h in range(int(sqrt(num_patches))):
                        rng = torch.rand(size=(1,)).item()
                        if rng < self.backdoor_args.adaptive_blend_patch_prob:
                            mark[i, :, h*l:(h+1)*l, w*l:(w+1)*l] = 0

        mark = Resize((m_w, m_h))(mark)

        opacity = self.backdoor_args.alpha
        x_before = deepcopy(x)
        x[:, :, m_x:m_x + m_w, m_y:m_y + m_h] = x[:, :, m_x:m_x + m_w, m_y:m_y + m_h].mul(
            1 - opacity) + mark.mul(opacity)
        x[:, :, m_x:m_x + m_w, m_y:m_y + m_h][mark == 0] = \
            x_before[:, :, m_x:m_x + m_w, m_y:m_y + m_h][mark == 0]

        rng = torch.rand(size=(1,)).item()
        if rng < self.backdoor_args.conservatism_rate:
            return x, y

        return x, torch.ones_like(y) * self.backdoor_args.target_class
