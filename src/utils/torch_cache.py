from typing import List
import torch


class TorchCache:
    def __init__(self, tensors: List[torch.Tensor]):
        self.cache = torch.stack(tensors)

    def __getitem__(self, item):
        return self.cache[item]
