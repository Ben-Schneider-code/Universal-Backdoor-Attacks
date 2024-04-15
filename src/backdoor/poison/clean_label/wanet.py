from copy import deepcopy
from typing import Tuple

import torch
from torch import nn

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import CleanLabelBackdoor


class Wanet(CleanLabelBackdoor):
    """ Use warping to embed a backdoor.
    https://arxiv.org/pdf/2102.10369
    """

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args=env_args)
        self.identity_grid, noise_gird = None, None
        self.wanet_s = backdoor_args.wanet_noise
        self.wanet_noise_rescale = backdoor_args.wanet_grid_rescale
        self.grid_rescale = backdoor_args.wanet_grid_rescale

    def requires_preparation(self) -> bool:
        """ Specify whether the backdoor has to be pre-computed.
        Trades-off computation time at the cost of more memory consumption. """
        return True

    def __gen_grid(self, height, k=8):
        """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        if self.identity_grid is None:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = nn.functional.interpolate(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d, indexing='ij')  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            self.identity_grid = deepcopy(identity_grid)
            self.noise_grid = deepcopy(noise_grid)
            self.h = self.identity_grid.shape[2]
            grid = self.identity_grid + self.wanet_s * self.noise_grid / self.h
            self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)

            self.identity_grid = identity_grid
            self.noise_grid = noise_grid

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x = x.squeeze()
        if len(x.shape) == 4:
            n, c, h, w = x.shape
        else:
            c, h, w = x.shape
            n = 1
        self.__gen_grid(h)
        if True:
            ins = torch.rand(1, self.h, self.h, 2) * self.wanet_noise_rescale - 1  # [-1, 1]
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid
        grid = grid.repeat(n, 1, 1, 1)  # Ensure grid has the same batch size as the input tensor
        #print(grid.shape, x.shape)
        if n == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if the input is a single image
        poison_img = nn.functional.grid_sample(x, grid, align_corners=True).squeeze()  # CHW
        return poison_img, torch.ones_like(y) * self.backdoor_args.target_class

