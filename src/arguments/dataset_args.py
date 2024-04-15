from dataclasses import dataclass, field


@dataclass
class DatasetArgs:
    CONFIG_KEY = "dataset_args"

    dataset_name: str = field(default="CIFAR10", metadata={
        "help": "name of the dataset",
        "choices": ["CIFAR10", "ImageNet"]
    })

    singular_embed:  bool = field(default=False, metadata={
        "help": "whether the dataset is willing to have batch embeds"
    })

    normalize: bool = field(default=True, metadata={
        "help": "whether to apply the dataset's normalization."
    })

    augment: bool = field(default=True, metadata={
        "help": "whether to augment the training dataset"
    })

    dataset_root: str = field(default=None, metadata={
        "help": "path to the dataset on the local storage. "
    })

    max_size_train: int = field(default=None, metadata={
        "help": "maximum size of the training dataset (in number of samples). Samples a subset randomly."
    })

    max_size_val: int = field(default=None, metadata={
        "help": "maximum size of the evaluation dataset (in number of samples). Samples a subset randomly."
    })

    seed: int = field(default=1337, metadata={
        "help": "datasets with the same seed have identical ordering"
    })
