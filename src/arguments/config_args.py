from dataclasses import dataclass, field

import yaml

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.detection_args import DetectionArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_evaluate_args import GridEvaluateArgs
from src.arguments.grid_search_args import GridSearchArgs
from src.arguments.model_args import ModelArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.utils.special_print import print_warning


@dataclass
class ConfigArgs:

    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify the config keys to read in the *.yml file
        EnvArgs.CONFIG_KEY: EnvArgs(),
        DatasetArgs.CONFIG_KEY: DatasetArgs(),
        ModelArgs.CONFIG_KEY: ModelArgs(),
        OutdirArgs.CONFIG_KEY: OutdirArgs(),
        TrainerArgs.CONFIG_KEY: TrainerArgs(),
        BackdoorArgs.CONFIG_KEY: BackdoorArgs(),
        BackdooredModelArgs.CONFIG_KEY: BackdooredModelArgs(),
        DefenseArgs.CONFIG_KEY: DefenseArgs(),
        ObserverArgs.CONFIG_KEY: ObserverArgs(),
        DetectionArgs.CONFIG_KEY: DetectionArgs(),
        GridSearchArgs.CONFIG_KEY: GridSearchArgs(),
        GridEvaluateArgs.CONFIG_KEY: GridEvaluateArgs()
    }

    def get_env_args(self) -> EnvArgs:
        return self.args_to_config[EnvArgs.CONFIG_KEY]

    def get_detection_args(self) -> DetectionArgs:
        return self.args_to_config[DetectionArgs.CONFIG_KEY]

    def get_defense_args(self) -> DefenseArgs:
        return self.args_to_config[DefenseArgs.CONFIG_KEY]

    def get_dataset_args(self) -> DatasetArgs:
        return self.args_to_config[DatasetArgs.CONFIG_KEY]

    def get_model_args(self) -> ModelArgs:
        return self.args_to_config[ModelArgs.CONFIG_KEY]

    def get_backdoored_model_args(self) -> BackdooredModelArgs:
        return self.args_to_config[BackdooredModelArgs.CONFIG_KEY]

    def get_outdir_args(self) -> OutdirArgs:
        return self.args_to_config[OutdirArgs.CONFIG_KEY]

    def get_grid_search_args(self) -> GridSearchArgs:
        return self.args_to_config[GridSearchArgs.CONFIG_KEY]

    def get_grid_evaluate_args(self) -> GridEvaluateArgs:
        return self.args_to_config[GridEvaluateArgs.CONFIG_KEY]

    def get_trainer_args(self) -> TrainerArgs:
        return self.args_to_config[TrainerArgs.CONFIG_KEY]

    def get_backdoor_args(self) -> BackdoorArgs:
        return self.args_to_config[BackdoorArgs.CONFIG_KEY]

    def get_observer_args(self) -> ObserverArgs:
        return self.args_to_config[ObserverArgs.CONFIG_KEY]

    def __post_init__(self):
        if self.config_path is None:
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.keys = list(data.keys())

        # load arguments
        keys_not_found = []
        for entry, values in data.items():
            for key, value in values.items():
                if key not in self.args_to_config[entry].__dict__.keys():
                    keys_not_found += [(entry, key)]
                self.args_to_config[entry].__dict__[key] = value
        if len(keys_not_found) > 0:
            print_warning(f"Could not find these keys: {keys_not_found}. Make sure they exist.")

        for key, value in self.args_to_config.items():
            if hasattr(value, "__post_init__"):
                value.__post_init__()





