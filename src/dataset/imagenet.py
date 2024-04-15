import os
from typing import List

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

from src.arguments.dataset_args import DatasetArgs
from src.dataset.dataset import Dataset
from src.global_configs import system_configs
from src.utils.dataset_labels import IMAGENET_LABELS, IMAGENET2K_LABELS, IMAGENET4K_LABELS, IMAGENET6K_LABELS


class ImageNet(Dataset):

    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)

        root = os.path.join(system_configs.IMAGENET_ROOT, "train" if train else "val")
        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.dataset_args.max_size_train if train else self.dataset_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.real_normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.classes = list(IMAGENET_LABELS.values())

    def get_class_to_idx(self, reset: bool = False, verbose: bool = False):
        """ Override this method in ImageNet for performance reasons.
        """
        if self.class_to_idx is None or reset:
            class_to_idx = {}
            # a list containing (path, class_idx)
            imgs: List[str, int] = [self.dataset.imgs[i] for i in self.idx]
            for idx, (_, class_idx) in enumerate(tqdm(imgs, f"Mapping Classes to Idx", disable=not verbose)):
                class_to_idx[class_idx] = class_to_idx.setdefault(class_idx, []) + [idx]
            self.class_to_idx = class_to_idx
        return self.class_to_idx

    def _build_transform(self):

        if self.train and self.dataset_args.augment:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224 + 32),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        return transform

    def without_transform(self) -> 'Dataset':
        """ Return a copy of this dataset without transforms.
        """
        copy = self.copy()
        copy.transform = lambda x: x if isinstance(x, torch.Tensor) else transforms.Compose([
            transforms.Resize(224 + 32),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])(x)
        return copy


class ImageNet2K(Dataset):
    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)

        root = os.path.join(system_configs.IMAGENET2K_ROOT, "train" if train else "val")
        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.dataset_args.max_size_train if train else self.dataset_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.real_normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.classes = list(IMAGENET2K_LABELS.values())


class ImageNet4K(Dataset):
    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)

        root = os.path.join(system_configs.IMAGENET4K_ROOT, "train" if train else "val")
        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.dataset_args.max_size_train if train else self.dataset_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.real_normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.classes = list(IMAGENET4K_LABELS.values())


class ImageNet6K(Dataset):
    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)

        root = os.path.join(system_configs.IMAGENET6K_ROOT, "train" if train else "val")
        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.dataset_args.max_size_train if train else self.dataset_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.real_normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        # switch to imagenet6k
        self.classes = list(IMAGENET6K_LABELS.values())
