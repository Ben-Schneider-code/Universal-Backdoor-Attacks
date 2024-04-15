import os.path

import matplotlib
import matplotlib.pyplot as plt
from os.path import dirname, isdir

import numpy as np
import torch
import torchvision
from PIL import Image

from src.utils.special_print import print_highlighted


def plot_images(x: torch.Tensor, n_row=2, dpi=400, savefig=None, title=None):
    x = x.detach().cpu()
    matplotlib.rcParams["figure.dpi"] = dpi
    grid_img = torchvision.utils.make_grid(x,nrow=n_row, range=(-1, 1), scale_each=True, normalize=True)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
        print_highlighted(f"Saved figure to {os.path.abspath(savefig)}")
    plt.show()


def image_to_tensor(image_file):
    """ Loads a PIL image and returns it with values scaled to [0,1]"""
    with Image.open(image_file) as img:
        img = img.convert('RGB')
        img = torch.from_numpy(np.array(img))
        img = img.permute(2, 0, 1).float()
    return img / 255.