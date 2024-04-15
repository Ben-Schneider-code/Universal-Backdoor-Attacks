from typing import Tuple, List

import torch

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor


class NoBackdoor(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args=env_args)
        self.selected_poisons: List[int] = [0]
        self.prep = False

    def requires_preparation(self) -> bool:
        return self.prep

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        return x, y

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        return self.selected_poisons
