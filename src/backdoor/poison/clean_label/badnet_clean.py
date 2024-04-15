from copy import deepcopy
from typing import Tuple

import torch
from torchvision.transforms import Resize

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import CleanLabelBackdoor
from src.utils.special_images import image_to_tensor


class BadnetClean(CleanLabelBackdoor):
    """ C-BadNet is a variation of a common data_cleaning poisoning attack without control over the labels.
        https://arxiv.org/abs/1708.06733
    """

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args=env_args)

    def requires_preparation(self) -> bool:
        """ Specify whether the backdoor has to be pre-computed.
        Trades-off computation time at the cost of more memory consumption. """
        return True

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
        x[:, :, m_x:m_x + m_w, m_y:m_y + m_h] = x[:, :, m_x:m_x + m_w, m_y:m_y + m_h].mul(
            1 - opacity) + self.backdoor_args.mark.mul(opacity)
        return x, y

