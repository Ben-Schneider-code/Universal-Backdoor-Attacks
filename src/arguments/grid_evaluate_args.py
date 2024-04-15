import random
from dataclasses import dataclass, field
from typing import List

import yaml

from src.arguments.defense_args import DefenseArgs
from src.database.grid_evaluate_database import GridEvaluateDatabase


@dataclass
class GridEvaluateArgs:
    CONFIG_KEY = "grid_evaluate_args"

    valid_parameter_file: str = field(default=None, metadata={
        "help": "path to a parameter '*.yml' file containing all model file names."
    })

    database_name: str = field(default=None, metadata={
        "help": "name of the database to persist all data_cleaning. This is a global    "
                "storage that keeps track of all data_cleaning and every experiment that"
                "has already been run.                                         "
    })

    repetitions: int = field(default=1, metadata={
        "help": "number of repetitions for each experiment"
    })

    shuffle: bool = field(default=True, metadata={
        "help": "Shuffle the order of the settings to run"
    })

    def __post_init__(self):
        self._database = GridEvaluateDatabase(self)

    def get_database(self):
        return self._database

    def get_all_settings(self) -> List[DefenseArgs]:
        """ Returns a list of all defense settings.
        """
        all_settings = []
        if self.valid_parameter_file is not None:
            with open(self.valid_parameter_file, 'r') as f:
                data = yaml.safe_load(f)
            for _, data_dict in data.items():
                all_settings += [DefenseArgs(**data_dict)]
        return all_settings

    def valid_parameters(self, shuffle: bool = False) -> List[DefenseArgs]:
        """ Return a list of valid parameters dicts for this defense for a grid_search.
            If a database is specified, we will subtract all experiments that have already
            been stored into that database.
        """
        all_settings = self.get_all_settings()
        if shuffle:
            random.shuffle(all_settings)
        return self._database.filter_remaining(all_settings)


