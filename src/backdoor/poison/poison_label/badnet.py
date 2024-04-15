from copy import deepcopy, copy
from typing import Tuple

import torch
from torchvision.transforms import Resize

from src.backdoor.backdoor import Backdoor
from src.utils.special_images import image_to_tensor


class Badnet(Backdoor):
    """ BadNets is a data_cleaning poisoning attack that has control over the labels.
    https://arxiv.org/abs/1708.06733

    """
    def requires_preparation(self) -> bool:
        return True

    def blank_cpy(self):
        backdoor_arg_copy = copy(self.backdoor_args)
        return self.__class__(backdoor_arg_copy, env_args=self.env_args)

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies a BadNet mark to a tensor with nchw
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        # Obtain shape of the mark
        m_x, m_y = int(self.backdoor_args.mark_offset_x * w), int(self.backdoor_args.mark_offset_y * h)
        m_w, m_h = int(self.backdoor_args.mark_width * w), int(self.backdoor_args.mark_height * h)

        # Load and resize the mark
        if self.backdoor_args.mark is None:
            self.backdoor_args.mark = image_to_tensor(self.backdoor_args.mark_path).unsqueeze(0)
        self.backdoor_args.mark = Resize((m_w, m_h))(self.backdoor_args.mark)

        # Apply the mark to the image
        opacity = self.backdoor_args.alpha
        x[:, :, m_x:m_x + m_w, m_y:m_y + m_h] = x[:, :, m_x:m_x+m_w, m_y:m_y+m_h].mul(1 - opacity) + self.backdoor_args.mark.mul(opacity)
        return x, torch.ones_like(y) * self.backdoor_args.target_class
