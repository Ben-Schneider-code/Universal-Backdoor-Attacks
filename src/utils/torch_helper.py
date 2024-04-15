
import hashlib
from typing import List

import torch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def to_groups(labels: torch.Tensor) -> List:
    # Create a dictionary to store the mapping of each number to its group index
    label_to_cls = {x: i for i, x in enumerate(set(labels.squeeze().tolist()))}
    return torch.LongTensor([label_to_cls[x] for x in labels.squeeze().tolist()])


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha256(tensor).hexdigest()

def yield_batches(images, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size]

def set_multiple_indices_to_zero(tensor, indices):
    # Get the batch size
    batch_size = tensor.shape[0]
    # Get the indices in 1D
    indices = indices[:,0]*tensor.shape[1]+indices[:,1] # calculate 1D indices
    # Create a mask of the same shape as the tensor
    mask = torch.ones_like(tensor)
    # reshape the mask to 1D
    mask = mask.reshape(-1)
    # set the corresponding element in the mask to 0
    mask[indices] = 0
    # reshape the mask back to 2D
    mask = mask.reshape(tensor.shape)
    # Apply the mask to the tensor
    tensor = tensor * mask
    return tensor


import torch
from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=1,  **kwargs):
        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers, **kwargs)

    def __iter__(self):
        while True:
            for data in super().__iter__():
                yield data

