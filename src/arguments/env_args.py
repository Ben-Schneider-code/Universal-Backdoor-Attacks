from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"

    num_workers: int = field(default=8, metadata={
        "help": "number of workers"
    })

    num_validation_workers : int = field(default=4, metadata={
        "help": "number of workers"
    })


    log_every: int = field(default=100, metadata={
        "help": "log interval for training"
    })

    experiment_repetitions: int = field(default=1, metadata={
        "help": "number of times to repeat the experiment"
    })

    gpus: List[int] = field(default_factory=lambda: [0], metadata={
                            "help": "list of GPUs to use"
    })

    save_every: int = field(default=249, metadata={
        "help": "save interval for training"
    })

    port: int = field(default=3000, metadata={
        "help": "the port on localhost that is used for communicating vectors during training"
    })

    num_gpus: int = field(default=1, metadata={
        "help": "parallelize to this number of GPUs"
    })

    device: str = field(default="cuda", metadata={
        "help": "device to run observers on"
    })

    batch_size: int = field(default=128, metadata={
        "help": "default batch size for training"
    })

    eval_batch_size: int = field(default=128, metadata={
        "help": "default batch size for inference"
    })

    verbose: bool = field(default=True, metadata={
        "help": "whether to print out to the cmd line"
    })
