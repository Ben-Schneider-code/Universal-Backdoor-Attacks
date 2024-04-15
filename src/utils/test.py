import os
from typing import List

from torch.utils.data import DataLoader
from src.arguments.dataset_args import DatasetArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.dataset.imagenet import ImageNet


def set_gpu_context(gpus: List[int]):
    device_str = ','.join(str(device) for device in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str


def _embed():

    ds_train: Dataset = ImageNet(DatasetArgs(), train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(DatasetArgs(), train=False)

    dl1 = DataLoader(ds_train, num_workers=3)
    dl2 = DataLoader(ds_test, num_workers=3)

    for i, (x,y) in enumerate(dl2):
        print("0")
    for j, (e, d) in enumerate(dl1):
        print("1")



if __name__ == "__main__":
    _embed()
