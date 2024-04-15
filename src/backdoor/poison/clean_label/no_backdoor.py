from typing import Tuple

import torch

from src.backdoor.backdoor import CleanLabelBackdoor


class NoBackdoor(CleanLabelBackdoor):
    """ This does not embed a backdoor and can be used to train clean models.
    """

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """
        """
        return x, y
