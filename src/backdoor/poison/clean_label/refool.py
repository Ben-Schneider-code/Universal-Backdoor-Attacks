import random
from copy import deepcopy
from typing import Tuple

import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

from src.backdoor.backdoor import CleanLabelBackdoor


class Refool(CleanLabelBackdoor):
    """ Refool is a clean data_cleaning poisoning attack.
    https://link.springer.com/chapter/10.1007/978-3-030-58607-2_11
    """

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies a reflection into a tensor with nchw
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        offset_x, offset_y = int(w * self.backdoor_args.ghost_offset_x), int(h * self.backdoor_args.ghost_offset_x)
        # Apply the mark to the image (via a ghosting effect)
        out = blend_images(x, opacity=1-self.backdoor_args.alpha, ghost_alpha=self.backdoor_args.ghost_alpha,
                           offset=(offset_x, offset_y))
        return out[0], y

    def requires_preparation(self) -> bool:
        """ Specify whether the backdoor has to be pre-computed.
        Trades-off computation time at the cost of more memory consumption. """
        return True

def blend_images(background_img: torch.Tensor, opacity: float,  ghost_alpha: float,
                 offset: tuple[int, int] = (0, 0),
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = background_img.shape[-2:]

    # Original code uses cv2 INTER_CUBIC, which is slightly different from BICUBIC
    background_img = F.resize(background_img, size=[h, w], interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    background_img.pow_(2.2)

    background_mask = opacity * background_img
    # generate the blended image with ghost effect
    if offset[0] == 0 and offset[1] == 0:
        offset = (random.randint(3, 8), random.randint(3, 8))
    reflect_1 = F.pad(background_img, [0, 0, offset[0], offset[1]])  # pad on right/bottom
    reflect_2 = F.pad(background_img, [offset[0], offset[1], 0, 0])  # pad on left/top
    reflect_ghost = ghost_alpha * reflect_1 + (1 - ghost_alpha) * reflect_2
    reflect_ghost = reflect_ghost[..., offset[0]: -offset[0], offset[1]: -offset[1]]
    reflect_ghost = F.resize(reflect_ghost, size=[h, w],
                             interpolation=InterpolationMode.BICUBIC
                             ).clamp(0, 1)  # no cubic mode in original code

    reflect_mask = (1 - opacity) * reflect_ghost
    reflection_layer = reflect_mask.pow(1 / 2.2)

    blended = (reflect_mask + background_mask).pow(1 / 2.2)
    background_layer = background_mask.pow(1 / 2.2)
    return blended, background_layer, reflection_layer
