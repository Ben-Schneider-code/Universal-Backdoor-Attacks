from dataclasses import dataclass, field
from typing import List

import torch.optim.lr_scheduler

from src.model.model import Model
from src.utils.python_helper import DynamicEnum


@dataclass
class TrainerArgs:
    CONFIG_KEY = "trainer_args"

    class TRAINERS(DynamicEnum):
        normal = "normal"
        entangled = "entangled"
        co_optimize = "co-optimized"
        handcrafted = "handcrafted"

    trainer_name: str = field(default=TRAINERS.normal, metadata={
        "help": "name of the trainer to use",
        "choices": [TRAINERS.normal, TRAINERS.entangled, TRAINERS.co_optimize]
    })

    fine_tune_poison_epochs: int = field(default=0, metadata={
        "help": "number of epochs to fine-tune with a poisoned dataset after training."
                "If this value is '0', we poison from the start. If it is >0, we first"
                "train the model for the specified time on clean data_cleaning and then add the poison"
                "for the remaining fine_tune_poison_epochs. "
    })

    entangled_trainer_loss: str = field(default="snnl", metadata={
        "help": "loss term for the entangled trainer"
    })

    poison_batch_ratio: float = field(default=0.1, metadata={
        "help": "percentage of samples to poison during training (only for the entangled trainer)"
    })

    lambda_snnl: float = field(default=0, metadata={
        "help": "strength of the snnl loss term. (only for the entangled trainer)"
    })

    lambda_stealth: float = field(default=0.001, metadata={
        "help": "strength of the mse loss term. (only for the co-optimized trainer)"
    })

    epochs: int = field(default=120, metadata={
        "help": "number of epochs"
    })

    lr: float = field(default=0.1, metadata={
        "help": "initial learning rate"
    })

    cosine_annealing_scheduler: bool = field(default=False, metadata={
        "help": "whether to use this scheduler"
    })

    linear_scheduler: bool = field(default=False, metadata={
        "help": "whether to use this scheduler"
    })

    step_size: int = field(default=30, metadata={
        "help": "decrease the blend size every {step_size} epochs"
    })

    gamma: float = field(default=.1, metadata={
        "help": "decrease lr by a factor of {gamma}"
    })

    t_max: int = field(default=120, metadata={
        "help": "T_max for the cosine annealing scheduler (if used!)"
    })

    boost: int = field(default=None, metadata={
        "help": "repeat backdoor samples to boost their importance. Only used during training."
    })

    eval_backdoor_every: int = field(default=None, metadata={
        "help": "for large datasets, evaluate backdoors after this many steps."
    })

    save_only_best: bool = field(default=True, metadata={
        "help": "force saving even if this model is not the best. "
    })

    save_best_every_steps: int = field(default=None, metadata={
        "help": "save after this many steps if improvement has occured."
    })

    weight_decay: float = field(default=5e-4, metadata={
        "help": "momentum (optional). Suggested values:"
                "- CIFAR10: 5e-4"
    })

    momentum: float = field(default=0.9, metadata={
        "help": "momentum (optional). Suggested values:"
                "- CIFAR10: 0.9"
    })

    optimizer: str = field(default="SGD", metadata={
        "help": "name of the optimizer",
        "choices": ["SGD"]
    })

    # --- Handcrafted ---
    update_param_fraction: float = field(default=1.0, metadata={
        "help": "ratio of parameters to update"
    })

    def get_optimizer(self, model: Model):
        if self.optimizer == "SGD":
            return torch.optim.SGD(model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        return None

    def get_scheduler(self, optimizer):
        if self.cosine_annealing_scheduler:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.t_max)
        if self.linear_scheduler:
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return None
