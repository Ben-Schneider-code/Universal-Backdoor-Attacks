from abc import ABC
from copy import deepcopy
from pprint import pprint
from typing import List

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.utils.special_images import plot_images


class Dataset(torch.utils.data.Dataset, ABC):

    def __init__(self, dataset_args: DatasetArgs, train: bool = True, env_args: EnvArgs = None):
        self.dataset_args = dataset_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.train = train
        self.apply_transform: bool = True

        # pre-processors receive an index, the image and the label of each item
        self.idx: List[int] = []  # all indices that this dataset returns
        self.idx_to_backdoor = {}  # store idx and backdoor mapping

        self.dataset: torch.utils.data.Dataset | None = None
        self.classes: List[str] = []
        self.real_normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.class_to_idx = None
        self.disable_fetching = False
        self._poison_label: int | bool = True
        self.target_index: [int] = None
        self.auto_embed_off = False


    def num_classes(self) -> int:
        """ Return the number of classes"""
        return len(self.classes)

    def _build_transform(self) -> None:
        """ Internal function to build a default transformation. Override this
        if necessary. """
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform

    def random_subset(self, n: int) -> 'Dataset':
        """ Creates a random subset of this dataset """
        idx = deepcopy(self.idx)
        np.random.shuffle(idx)

        copy = self.copy()
        copy.idx = idx[:n]
        copy.class_to_idx = None
        return copy

    def concat(self, other: 'Dataset') -> 'Dataset':
        """ Concatenates two datasets """
        copy = self.copy()
        copy.dataset = ConcatDataset(self.without_normalization().without_transform(),
                                     other.without_normalization().without_transform())
        copy.idx = np.arange(len(copy.dataset)).tolist()
        copy.transform = self.normalize_transform  # do not use ToTensor
        copy.class_to_idx = None
        return copy

    def shape(self):
        return self[0][0].shape

    def shuffle(self):
        copy = self.copy()
        np.random.shuffle(copy.idx)
        return copy

    def subset(self, idx: List[int] | int) -> 'Dataset':
        """ Creates a subset of this dataset. """
        if isinstance(idx, int):
            idx = np.arange(idx)
        copy = self.copy()
        copy.idx = [self.idx[i] for i in idx]
        copy.class_to_idx = None
        return copy

    def remove_classes(self, target_classes: List[int], verbose: bool = False):
        """ Creates a subset without samples from one target class. """
        copy = self.copy()
        idx_to_remove = []
        for target_class in tqdm(target_classes, "Removing classes", disable=not verbose):
            if target_class in copy.get_class_to_idx().keys():
                idx_to_remove += [copy.idx[i] for i in copy.get_class_to_idx()[target_class]]
        for idx in sorted(idx_to_remove, reverse=True):
            copy.idx.remove(idx)
        copy.class_to_idx = None
        return copy

    def visualize(self, sq_size: int = 3) -> None:
        """ Plot samples from this dataset as a square.
        """
        n = sq_size ** 2
        no_norm = self.without_normalization()
        x = [no_norm[i][0] for i in range(n)]
        imgs = torch.stack(x, 0)

        if len(self.idx_to_backdoor) == 0:
            title = self.dataset_args.dataset_name
        else:
            title = f"{self.dataset_args.dataset_name} (Poisoned)"
        plot_images(imgs, n_row=sq_size, title=title)

    def visualize_index(self, index):
        no_norm = self.without_normalization()

        x = no_norm[index][0]
        plot_images(x)

    def print_class_distribution(self):
        class_to_idx = self.get_class_to_idx(verbose=False)
        cd = {c: 100 * len(v) / len(self) for c, v in class_to_idx.items()}
        pprint(cd)

    def without_normalization(self) -> 'Dataset':
        """ Return a copy of this dataset without normalization.
        """
        copy = self.copy()
        copy.enable_normalization(False)
        return copy

    def without_transform(self) -> 'Dataset':
        """ Return a copy of this dataset without transforms.
        """
        copy = self.copy()
        copy.transform = lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)
        return copy

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ Normalizes a tensor. Note: Tensors received from the dataset
        are usually already normalized. """
        return self.normalize_transform(x)

    def enable_normalization(self, enable: bool) -> 'Dataset':
        """ Method to enable or disable normalization.
        """
        self.dataset_args.normalize = enable
        if enable:
            self.normalize_transform = self.real_normalize_transform
        else:
            self.normalize_transform = transforms.Lambda(no_op)
        self.transform = self._build_transform()
        return self

    def size(self):
        """ Alternative function to get the size. """
        return len(self.idx)

    def __len__(self):
        """ Return the number of elements in this dataset """
        return self.size()

    def get_class_to_idx(self, reset: bool = False, verbose: bool = False):
        """ Returns a dict that maps each class in the dataset to
        an (external) index. An external index corresponds to the order that
        the dataset elements are in from the perspective outside of this class.
        Set reset to 'True' to force re-computation of the index.
        """
        data_loader = DataLoader(self,
                                 num_workers=self.env_args.num_workers,
                                 batch_size=self.env_args.batch_size,
                                 shuffle=False)
        if self.class_to_idx is None or reset:
            class_to_idx = {}
            ctr = 0
            for idx, (_, y) in enumerate(tqdm(data_loader, desc="Mapping Classes to Idx", disable=verbose)):
                for i, y_i in enumerate(y):
                    class_to_idx[int(y_i.item())] = class_to_idx.setdefault(int(y_i.item()), []) + [ctr + i]
                ctr += len(y)
            self.class_to_idx = class_to_idx
        return self.class_to_idx

    def copy(self):
        """ Return a copy of this dataset instance. """
        return deepcopy(self)

    def set_poison_label(self, value: int | bool):
        """ Set the mode for returning labels.
        'value=True' means that the backdoor controls the label
        'value=False' means that we return clean labels
        'value=[int]' means that we always return the specified [int] label. Useful for Clean Label Backdoors.
        """
        self._poison_label = value
        return self

    def poison_label(self):
        """ Gets the mode for returning labels. """
        return self._poison_label

    def add_poison(self, backdoor: 'Backdoor', poison_all: bool = False, boost: int | None = None, util=None):
        """ Add a new backdoor to this dataset. """

        if poison_all:
            target_idx = deepcopy(self.idx)
        else:
            target_idx = backdoor.choose_poisoning_targets(self.get_class_to_idx(verbose=False))
            assert (len(target_idx) == backdoor.backdoor_args.poison_num)
            assert(len(set(target_idx)) == len(target_idx))
        
        if boost is not None:
            for _ in range(boost):
                self.idx += target_idx

        for idx in target_idx:
            self.idx_to_backdoor[idx] = self.idx_to_backdoor.setdefault(idx, []) + [backdoor]

        self.target_index = target_idx

        # Some backdoors need pre-computations. This trades-off memory for computation time.
        if backdoor.requires_preparation() and not backdoor.all_indices_prepared(target_idx):
            self.disable_fetching = True
            self.auto_embed_off = True
            dl = DataLoader(self.subset([self.idx.index(ti) for ti in target_idx]).without_normalization(),
                            batch_size=1 if self.dataset_args.singular_embed else self.env_args.batch_size,
                            drop_last=False,
                            num_workers=self.env_args.num_workers, shuffle=False)
            ctr = 0
            item_indices = target_idx
            target_idx = target_idx if self.train else [-idx for idx in target_idx]
            for i, (x, y) in enumerate(tqdm(dl, desc=f"Preparing {backdoor.backdoor_args.backdoor_name} backdoor")):
                idx = target_idx[ctr:ctr + len(x)]
                ctr += len(x)
                backdoor.prepare(x, y, idx, item_index=item_indices[i], util=util)

            self.disable_fetching = False
            self.auto_embed_off = False
        return self

    def clear_poison(self):
        """ Remove all backdoors attached to this dataset. """
        self.idx_to_backdoor = {}

    def __getitem__(self, index):
        index = self.idx[index]
        x, y0 = self.dataset[index]
        y = y0
        x = self.transform(x)  # transform without normalize

        if self.auto_embed_off:
            return x,y

        for backdoor in self.idx_to_backdoor.setdefault(index, []):
            if backdoor.requires_preparation() and not self.disable_fetching:
                try:
                    x, y = backdoor.fetch(index if self.train else -index)  # fetch precomputed results
                except:
                    print(f"Tried fetching index='{index}' with backdoor, but could not. "
                          f"No backdoor will be embedded.")
            else:
                x, y = backdoor.embed(x.unsqueeze(0), torch.tensor(y0), data_index=index)
                x, y = x.squeeze(), y.item()

        if self._poison_label is False:
            y_out = y0  # always return the true label
        elif self._poison_label is True:
            y_out = y  # return the backdoor's label
        else:
            y_out = torch.tensor(self._poison_label)  # return the specified label

        if isinstance(y_out, (int, float)):
            y_out = torch.tensor(y_out)
        if self.dataset_args.normalize:
            return self.normalize(x), y_out
        return x, y_out

    def poisoned_subset(self, backdoor) -> 'Dataset':
        psn_indices = list(backdoor.index_to_target.keys())
        psn_subset = self.subset(psn_indices)
        return psn_subset


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dataset_args: DatasetArgs, env_args: EnvArgs):
        super().__init__(dataset_args, env_args=env_args)
        self.x = x.cpu().detach()
        self.y = y.cpu().detach()
        self.idx = torch.arange(len(self.x))

    def __getitem__(self, item):
        return self.x[self.idx[item]], self.y[self.idx[item]]


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(dataset1) + len(dataset2)

        # Hacky solution to export attributes.
        if hasattr(self.dataset1.dataset, 'imgs'):
            self.imgs = self.dataset1.dataset.imgs + self.dataset2.dataset.imgs

    def __getitem__(self, index):
        if index < len(self.dataset1):
            return self.dataset1[index]
        else:
            return self.dataset2[index - len(self.dataset1)]

    def __len__(self):
        return self.length


def no_op(x):
    return x
