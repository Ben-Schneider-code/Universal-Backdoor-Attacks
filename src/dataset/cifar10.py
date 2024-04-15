import numpy as np
import torchvision
from torchvision.transforms import transforms

from src.arguments.dataset_args import DatasetArgs
from src.dataset.dataset import Dataset
from src.global_configs import system_configs


class CIFAR10(Dataset):

    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)
        self.dataset = torchvision.datasets.CIFAR10(root=system_configs.CACHE_DIR, download=True, train=train, transform=None)
        self.idx = list(range(len(self.dataset)))

        max_size = self.dataset_args.max_size_train if train else self.dataset_args.max_size_val
        if max_size is not None:
            self.idx = np.random.choice(self.idx, max_size)

        self.real_normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.normalize_transform = self.real_normalize_transform
        self.transform = self._build_transform()
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def _build_transform(self):
        if self.train and self.dataset_args.augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        return transform


