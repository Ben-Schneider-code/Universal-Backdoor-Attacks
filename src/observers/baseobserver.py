import json
import os.path
import random
from abc import abstractmethod
from dataclasses import is_dataclass
from typing import List

import pandas as pd
import yaml

from src.arguments.env_args import EnvArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.utils.special_print import print_warning


class BaseObserver:
    """ Base class for recording data_cleaning for an experiment.
        This implements the observer pattern.
    """
    CDA  = "cda"     # clean data_cleaning accuracy
    ASR  = "asr"     # attack success rate
    ARR  = "arr"     # attack recovery rate
    STEP = "step"    # step of the defense
    ID   = "ID"      # defense unique identifier
    ARGS = "args"    # defense arguments
    FIN = "finish"   # experiment has finished successfully
    BACKDOOR_NAME = "backdoor_name"

    def __init__(self, observer_args: ObserverArgs, env_args: EnvArgs = None):
        self.observer_args = observer_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.df = pd.DataFrame({key: [] for key in self.observed_keys()})
        self.configs = {}

    def attach_config(self, name, data):
        """ Allow adding fields to be saved. """
        self.configs[name] = data

    @abstractmethod
    def observed_keys(self) -> List[str]:
        """ Returns a list of keys that are observed"""
        raise NotImplementedError()

    def save(self, outdir_args: OutdirArgs):
        """ Save the data_cleaning as a csv file. """
        filepath = os.path.join(outdir_args.create_folder_name(), f"{str(self)}.csv")
        self.df.to_csv(filepath, index=False)
        print(f"Saved {self} to '{os.path.abspath(filepath)}'.")

        config_fp = os.path.join(outdir_args.create_folder_name(), f"{str(self)}.yml")
        if len(self.configs) > 0:
            with open(config_fp, 'w') as file:
                yaml.dump(self.configs, file)
            print(f"Saved config to '{os.path.abspath(config_fp)}'")

    def plot(self):
        print_warning("No plotting method was implemented .. ")

    def notify(self, state_dict: dict) -> None:
        """ Default behavior: Add measurement to dict """
        if all([key in state_dict for key in self.observed_keys()]):
            state_dict = {key: state_dict[key] for key in self.observed_keys()}
            # Only update if all keys are present.
            other_df = pd.DataFrame.from_dict({key: [state_dict[key] if not is_dataclass(state_dict[key]) else
                                            json.dumps(state_dict[key].__dict__)] for key in self.observed_keys()})
            self.df = pd.concat([self.df, other_df])
    def __str__(self):
        return f"BaseObserver{random.randint(0,10**5)}"


