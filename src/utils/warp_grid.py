from copy import deepcopy
from typing import Tuple

import torch
from torch import nn

from src.arguments.backdoor_args import BackdoorArgs


class WarpGrid:

    def __init__(self, backdoor_args: BackdoorArgs):
        self.identity_grid, self.noise_grid = None, None
        self.wanet_s = backdoor_args.wanet_noise  # adjust this parameter to increase warping
        self.wanet_noise_rescale = backdoor_args.wanet_grid_rescale
        self.grid_rescale = backdoor_args.wanet_grid_rescale
        self.warping_direction = None

    def __gen_grid(self, height, k=8):
        """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        if self.identity_grid is None:
            self.warping_direction = torch.ones((1, 2, k, k))
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = nn.functional.interpolate(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2

            # Interpolate warping_direction to the same size as noise_grid
            warping_direction = nn.functional.interpolate(self.warping_direction, size=height, mode="nearest")
            warping_direction = warping_direction.permute(0, 2, 3, 1)  # 1*height*height*2

            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d, indexing='ij')  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            self.identity_grid = deepcopy(identity_grid)
            self.noise_grid = deepcopy(noise_grid)
            self.warping_direction = deepcopy(warping_direction)  # update warping_direction tensor
            self.h = self.identity_grid.shape[2]

            grid = self.identity_grid + self.wanet_s * self.noise_grid * self.warping_direction / self.h
            self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)

            grid2 = self.identity_grid + self.wanet_s * self.noise_grid * self.warping_direction * -1 / self.h
            self.grid2 = torch.clamp(grid2 * self.grid_rescale, -1, 1)

            self.identity_grid = identity_grid
            self.noise_grid = noise_grid
            self.warping_direction = warping_direction

    def warp(self, x: torch.Tensor) -> Tuple:
        x = x.squeeze()
        if len(x.shape) == 4:
            n, c, h, w = x.shape
        else:
            c, h, w = x.shape
            n = 1
        self.__gen_grid(h)

        ins = torch.rand(1, self.h, self.h, 2) * self.wanet_noise_rescale - 1  # [-1, 1]
        grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        grid = grid.repeat(n, 1, 1, 1)  # Ensure grid has the same batch size as the input tensor

        ins2 = torch.rand(1, self.h, self.h, 2) * self.wanet_noise_rescale - 1  # [-1, 1]
        grid2 = torch.clamp(self.grid2 + ins2 / self.h, -1, 1)
        grid2 = grid2.repeat(n, 1, 1, 1)  # Ensure grid has the same batch size as the input tensor

        if n == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if the input is a single image

        poison_img = nn.functional.grid_sample(x, grid, align_corners=True).squeeze()
        poison_img2 = nn.functional.grid_sample(x, grid2, align_corners=True).squeeze()

        return poison_img, poison_img2
