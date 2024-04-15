import os
import pickle
from dataclasses import dataclass, field
from typing import List, Generator

import torch
from dacite import from_dict
from pandas.io.common import is_url

from src.backdoor.backdoor import Backdoor
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.utils.requests import url_to_local_file


@dataclass
class BackdooredModelArgs:
    CONFIG_KEY = "backdoored_model_args"  # this loads a model + backdoor

    backdoor_model_ckpt: str = field(default=None, metadata={
        "help": "the name of the backdoor"
    })

    bdma_verbose: bool = field(default=True, metadata={
        "help": "whether to be verbose"
    })

    path: str = field(default=None, metadata={
        "help": "path to saved backdoored model"
    })

    model_file: str = field(default=None, metadata={
        "help": "file name of model"
    })

    backdoor_file: str = field(default=None, metadata={
        "help": "file name of backdoor"
    })

    def __post_init__(self):
        self.backdoor_args = BackdoorArgs()
        if self.backdoor_model_ckpt:
            if is_url(self.backdoor_model_ckpt):
                self.backdoor_model_ckpt = url_to_local_file(self.backdoor_model_ckpt)
                if self.bdma_verbose:
                    print(f"Reading model from '{os.path.abspath(self.backdoor_model_ckpt)}'")

            self.content = torch.load(self.backdoor_model_ckpt, map_location=torch.device('cpu'))
            self.model_args: ModelArgs = self.content[ModelArgs.CONFIG_KEY][Model.CONFIG_MODEL_ARGS]

            mark: torch.Tensor = self.content[BackdoorArgs.CONFIG_KEY]['mark']
            self.content[BackdoorArgs.CONFIG_KEY]['mark'] = None

            if 'marks' in self.content[BackdoorArgs.CONFIG_KEY]:
                marks: torch.Tensor = self.content[BackdoorArgs.CONFIG_KEY].setdefault('marks', None)
                self.content[BackdoorArgs.CONFIG_KEY]['marks'] = None

            self.backdoor_args: BackdoorArgs = from_dict(data_class=BackdoorArgs,
                                                         data=self.content[BackdoorArgs.CONFIG_KEY])
            self.backdoor_args.mark = mark
            if 'marks' in self.content[BackdoorArgs.CONFIG_KEY]:
                self.backdoor_args.marks = marks

    # using the torch pickler (torch.save / torch.load) would be better
    def unpickle(self, model_args, env_args):
        model_path = self.path + self.model_file
        backdoor_path = self.path + self.backdoor_file
        print("Load model from: " + model_path)

        model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)
        model.load(ckpt=model_path)

        print("model loaded")

        print("Load backdoor from: " + backdoor_path)
        try:
            backdoor = torch.load(backdoor_path)
        except:
            with open(backdoor_path, 'rb') as p_file:
                backdoor = pickle.load(p_file)
        print("backdoor loaded")
        return model.cpu(), backdoor

    def get_model_args(self) -> ModelArgs:
        return self.model_args

    def get_backdoor_args(self) -> BackdoorArgs:
        return self.backdoor_args

    def load_model(self, env_args: EnvArgs = None) -> 'Model':
        self.model: 'Model' = ModelFactory.from_model_args(self.model_args, env_args=env_args)
        self.model.load(content=self.content)
        return self.model

    def load_backdoor(self, env_args: EnvArgs = None) -> 'Backdoor':
        self.__post_init__()
        self.backdoor = BackdoorFactory.from_backdoor_args(self.backdoor_args, env_args=env_args)
        return self.backdoor

    def load_backdoor_args(self) -> 'BackdoorArgs':
        return self.backdoor_args


@dataclass
class ListBackdooredModelArgs:
    backdoor_model_ckpts: List[BackdooredModelArgs] = field(default_factory=lambda: [], metadata={
        "help": "the name of the backdoored model checkpoints"
    })

    def iter_backdoors(self, env_args: EnvArgs = None):
        for backdoor_model_args in self.backdoor_model_ckpts:
            backdoor = backdoor_model_args.load_backdoor(env_args=env_args)
            yield backdoor_model_args, backdoor

    def __iter__(self):
        return self.backdoor_model_ckpts.__iter__()

    def __len__(self):
        return len(self.backdoor_model_ckpts)
