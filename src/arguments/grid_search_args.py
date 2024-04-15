import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import yaml

from src.arguments.defense_args import DefenseArgs
from src.database.grid_search_database import GridSearchDatabase
from src.utils.python_helper import deduplicate_list_of_dicts


@dataclass
class GridSearchArgs:
    CONFIG_KEY = "grid_search_args"

    valid_parameter_file: str = field(default=None, metadata={
        "help": "path to a parameter '*.yml' file"
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
        self._database = GridSearchDatabase(self)

    def get_database(self):
        return self._database

    def _resolve_settings(self, settings: dict) -> List[DefenseArgs]:
        """ Recursively resolves settings. """
        all_settings = []
        is_leaf = True
        for name, param in settings.items():
            if isinstance(param, list):
                is_leaf = False
                for param_value in param:
                    new_setting = deepcopy(settings)
                    new_setting[name] = param_value
                    all_settings.extend(self._resolve_settings(new_setting))
        if is_leaf:
            all_settings += [DefenseArgs(**settings)]
        return all_settings

    def get_all_settings(self) -> List[DefenseArgs]:
        all_settings = []
        if self.valid_parameter_file is not None:
            with open(self.valid_parameter_file, 'r') as f:
                data = yaml.safe_load(f)
            for _, settings in data.items():
                all_settings.extend(self._resolve_settings(settings))
        return deduplicate_list_of_dicts(all_settings) * self.repetitions

    def valid_parameters(self, shuffle: bool = False) -> List[DefenseArgs]:
        """ Return a list of valid parameters dicts for this defense for a grid_search.
            If a database is specified, we will subtract all experiments that have already
            been stored into that database.
        """
        all_settings = self.get_all_settings()
        result = self._database.filter_remaining(all_settings)
        if shuffle:
            random.shuffle(result)
        return result

