from dataclasses import dataclass, field
from typing import List

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.utils.special_print import print_dict_highlighted


@dataclass
class DetectionArgs:
    CONFIG_KEY = "detection_args"

    backdoored_models: str = field(default_factory=lambda: [], metadata={
        "help": "path to the backdoored models"
    })

    clean_models: str = field(default_factory=lambda: [], metadata={
        "help": "path to the clean models"
    })

    def load_models(self, clean=False, env_args=None, verbose=False) -> List:
        models = []
        model_set = self.clean_models if clean else self.backdoored_models
        for backdoor_model_ckpt in model_set:
            backdoored_model_args = BackdooredModelArgs(backdoor_model_ckpt)
            backdoor = backdoored_model_args.load_backdoor(env_args=env_args)
            model = backdoored_model_args.load_model(env_args=env_args)
            models += [{"model": model, "backdoor": backdoor}]

        if verbose:
            print_dict_highlighted(vars(models[0]["backdoor"].backdoor_args))
        return models
