import json
import os
from typing import List

import pandas as pd

from src.arguments.defense_args import DefenseArgs
from src.global_configs import system_configs
from src.observers.baseobserver import BaseObserver


class GridSearchDatabase:
    """ A wrapper around a pandas dictionary.
    The GridSearchDatabase is used to find optimal parameters for a defense
    against a single attack.
    """

    def __init__(self, grid_search_args):
        self.grid_search_args = grid_search_args

        self.df = None
        if self.grid_search_args.database_name is not None:
            fp = os.path.abspath(os.path.join(system_configs.CACHE_DIR, self.grid_search_args.database_name))
            if os.path.exists(fp):
                self.df = pd.read_csv(fp)
                # Remove invalid rows
                self.df = self.df.dropna()
                # Remove experiments that haven't finished
                self.df = self.df.groupby(BaseObserver.ID).filter(lambda x: (x[BaseObserver.FIN] > 0).any())
                # Remove experiments that are not in the currently specified file
                self.df = self.df[self.df[BaseObserver.ARGS].apply(lambda x: DefenseArgs(**json.loads(x)) in self.grid_search_args.get_all_settings())]
                print(f"Loaded database with {self.count_experiments()} experiments from '{fp}'")

    def get_data(self):
        return self.df

    def commit(self, df):
        """ Update the database, but only with columns that have finished"""
        self.df = df
        if self.grid_search_args.database_name is not None:
            fp = os.path.join(system_configs.CACHE_DIR, self.grid_search_args.database_name)
            self.df.to_csv(fp, index=False)

    def count_experiments(self):
        if len(self.df) == 0:
            return 0
        return len(self.df.groupby(BaseObserver.ID).last())

    def count(self, defense_args: DefenseArgs) -> int:
        """ Returns the number of times this defense setting is in the database
        """
        if self.df is None or len(self.df) == 0:
            return 0
        return self.df.groupby(BaseObserver.ID).last()[BaseObserver.ARGS].apply(lambda x: DefenseArgs(**json.loads(x)) == defense_args).tolist().count(True)

    def filter_remaining(self, settings: List[DefenseArgs]) -> List[DefenseArgs]:
        """ Given a list of settings, this function returns a filtered list with
            only the experiments remaining that need to be run.
        """
        unique_list = []
        for i, setting in enumerate(settings):
            if setting not in unique_list:
                unique_list += [setting]

        filtered_settings = []
        for setting in unique_list:
            filtered_settings += [setting] * max(0, self.grid_search_args.repetitions - self.count(setting))

        return filtered_settings

