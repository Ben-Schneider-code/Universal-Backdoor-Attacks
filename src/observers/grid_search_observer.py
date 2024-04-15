import json
import os.path
from typing import List

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_search_args import GridSearchArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.observers.baseobserver import BaseObserver


class GridSearchObserver(BaseObserver):
    """ Observer that tracks the CDA, ASR and Step during an experiment.
    """
    KEY = "grid-search"    # use this key to specify this observer in the args

    def __init__(self, observer_args: ObserverArgs, grid_search_args: GridSearchArgs = None, env_args: EnvArgs = None):
        super().__init__(observer_args, env_args)
        self.grid_search_args = grid_search_args

        db = self.grid_search_args.get_database()
        data = db.get_data()
        self.df = data if data is not None else self.df

    def persist_to_db(self):
        self.grid_search_args.get_database().commit(self.df)

    def observed_keys(self) -> List[str]:
        return [self.STEP, self.CDA, self.ASR, self.ID, self.ARGS, self.FIN]

    def notify(self, state_dict: dict) -> None:
        super().notify(state_dict=state_dict)
        self.persist_to_db()

    def save(self, outdir_args: OutdirArgs):
        super().save(outdir_args)
        db = self.grid_search_args.get_database()
        if db is not None:
            db.commit(self.df)

    def plot(self):
        """ Plots the data_cleaning. This is ONLY an (optional) convenience function.
            Do NOT use for plots in a paper.
        """
        asr = self.df[self.ASR]
        cda = self.df[self.CDA]
        steps = self.df[self.STEP]
        ids = self.df[self.ID].astype('int64')
        base_cda = self.df[self.CDA].max()
        args = self.df[self.ARGS]
        defense_args: DefenseArgs = self.configs.setdefault("defense_args", DefenseArgs(**json.loads(args.iloc[-1])))

        delta = self.observer_args.delta / 100  # maximum deterioration in CDA.

        # Sort the data_cleaning per ID.
        per_id_data = {}
        for id in np.unique(ids):
            data = {
                self.ASR: asr[ids == id],
                self.CDA: cda[ids == id],
                self.STEP: steps[ids == id],
                self.ARGS: args[ids == id]
            }
            per_id_data[id] = data

        # Plot ASR/CDA curves
        plt.title(f"Parameter Ablation for {defense_args.def_name.capitalize()} / ASR versus CDA")
        for j, (id, data) in enumerate(per_id_data.items()):
            plt.scatter(data[self.ASR], data[self.CDA], marker='x')
        plt.hlines(base_cda, 0, 1.0, label="Original CDA", linestyles='--', color='black', linewidth=1.0, alpha=0.4)
        plt.hlines(base_cda-delta, 0, 1.0, label="Lowest Acceptable CDA " + rf'$(\Delta={100 * delta}\%)$', linestyles='--', color='black', linewidth=1.0, alpha=0.4)
        plt.xlabel("Attack Success Rate (ASR)")
        plt.ylabel("Clean Data Accuracy (CDA)")
        plt.legend()
        savefig = f'{defense_args.def_name}_param_ablation.pdf'
        plt.savefig(savefig)
        print(f"Saved figure at '{os.path.abspath((savefig))}'")
        plt.show()

        # Make a scatter plot and find the lowest ASR
        cdas, asrs = [], []
        best = {}
        for id, data in per_id_data.items():
            asr = data[self.ASR][data[self.CDA] >= base_cda - delta].to_numpy()
            cda = data[self.CDA][data[self.CDA] >= base_cda - delta].to_numpy()

            best_idx = asr.argmin()
            cdas += [cda[best_idx]]
            asrs += [asr[best_idx]]

            # best is lowest asr
            if best.setdefault("asr", 1.0) > asr[best_idx]:
                best["asr"] = asr[best_idx]
                best["cda"] = cda[best_idx]
                best["id"] = id
        try:
            print(f"Best setting: {per_id_data[best['id']][self.ARGS].iloc[-1]} {best}")
        except Exception as e:
            print(e)
            print(f"Found no setting for this case .. {best}")

        name = "Defense"
        if defense_args is not None:
            name = defense_args.def_name
        plt.scatter(asrs, cdas, label=f"{name.title()}")
        plt.hlines(base_cda, 0, 1.03*max(asrs), color='black', linestyle="--", linewidth=1.0, label="Original CDA", alpha=0.4)
        plt.hlines(base_cda-delta, 0, 1.03*max(asrs), color='black', linestyle="--", linewidth=1.0,
                   label="Lowest Acceptable CDA",
                   alpha=0.4)

        plt.title(f"Grid Search for {name.capitalize()} " + fr'($\Delta={delta*100}\%$)')
        plt.legend()
        plt.xlabel("Attack Success Rate (ASR)")
        plt.ylabel("Clean Data Accuracy (CDA)")
        plt.xlim()
        plt.ylim(top=1.001*base_cda, bottom=0.999*(base_cda-delta))
        savefig = f'{defense_args.def_name}_grid_search.pdf'
        plt.savefig(savefig)
        print(f"Saved figure at '{os.path.abspath((savefig))}'")
        plt.show()

    def __str__(self):
        return "GridSearchObserver"





