import json
import os.path
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_evaluate_args import GridEvaluateArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.observers.baseobserver import BaseObserver


class GridEvaluateObserver(BaseObserver):
    """ Observer that tracks the CDA, ASR and Step during an experiment.
    """
    KEY = "grid-evaluate"  # use this key to specify this observer in the args

    def __init__(self, observer_args: ObserverArgs, grid_evaluate_args: GridEvaluateArgs = None,
                 env_args: EnvArgs = None):
        super().__init__(observer_args, env_args)
        self.grid_evaluate_args = grid_evaluate_args

        db = self.grid_evaluate_args.get_database()
        data = db.get_data()
        self.df = data if data is not None else self.df

    def persist_to_db(self):
        self.grid_evaluate_args.get_database().commit(self.df)

    def observed_keys(self) -> List[str]:
        return [self.STEP, self.CDA, self.ASR, self.ID, self.ARGS, self.FIN, self.BACKDOOR_NAME]

    def notify(self, state_dict: dict) -> None:
        super().notify(state_dict=state_dict)
        self.persist_to_db()

    def save(self, outdir_args: OutdirArgs):
        super().save(outdir_args)
        db = self.grid_evaluate_args.get_database()
        if db is not None:
            db.commit(self.df)

    def plot(self):
        """ Plots the data_cleaning by grouping all data_cleaning points for the same defense args.
        """
        if len(self.df) == 0:
            print(f"No data_cleaning yet for plotting .. Returning.")
            return
        name = self.df[self.BACKDOOR_NAME].iloc[0]  # we assume all datapoints attack the same backdoor.
        m = DefenseArgs(
            **json.loads(self.df[self.ARGS].iloc[0])).def_data_ratio  # assume all attacks have the same data_cleaning.
        delta = self.observer_args.delta / 100  # maximum deterioration in CDA.

        # Extract the raw data_cleaning fields
        asr = self.df[self.ASR]
        cda = self.df[self.CDA]
        steps = self.df[self.STEP]
        base_cda = self.df[self.CDA].max()
        args = self.df[self.ARGS]
        backdoor_names = self.df[self.BACKDOOR_NAME]

        # Sort the data_cleaning per ID (ID = DefenseArgs)
        per_id_data = {}
        for id in np.unique(args):
            data = {
                self.ASR: asr[args == id],
                self.CDA: cda[args == id],
                self.STEP: steps[args == id],
                self.ARGS: args[args == id],
                self.BACKDOOR_NAME: backdoor_names[args == id]
            }
            per_id_data[id] = data

        # Make a scatter plot and find the lowest ASR
        for id, data in per_id_data.items():
            x = data[self.ASR].to_numpy()
            y = data[self.CDA].to_numpy()

            num_bins = 10
            # Create equal-number-of-points bins
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            bin_size = len(x_sorted) // num_bins

            # get the lowest ASR where the CDA is still higher than original minus delta
            lowest_asr = np.min(x_sorted[y_sorted >= base_cda - delta])
            print(
                f"Lowest ASR for {DefenseArgs(**json.loads(id)).def_name} is {lowest_asr:.2f} where CDA is still higher than {base_cda - delta:.2f}."
                f"The CDA for this point is {y_sorted[np.argmax(y_sorted > base_cda - delta)]:.2f}")
            # Compute the bin centers, mean, and standard deviation of y-values in each bin
            bin_centers = []
            y_means = []
            y_stds = []

            for i in range(num_bins):
                bin_data = y_sorted[i * bin_size:(i + 1) * bin_size]
                bin_center = x_sorted[(i * bin_size + (i + 1) * bin_size) // 2]
                bin_centers.append(bin_center)
                y_means.append(bin_data.mean())
                y_stds.append(np.clip(bin_data.std(), 0, delta))
            bin_centers += [np.max(x_sorted)]
            y_means += [np.max(y_sorted)]
            y_stds += [0]

            def_name = DefenseArgs(**json.loads(id)).def_name
            plot_args = DefenseArgs.resolve_plot_kwargs(def_name)
            plt.plot(bin_centers, y_means, marker='x', label=def_name.capitalize(), color=plot_args['color'])
            plt.fill_between(bin_centers, np.asarray(y_means) - np.asarray(y_stds),  np.asarray(y_means) + np.asarray(y_stds), alpha=0.2,
                             color=plot_args['color'])
            plt.scatter(x_sorted, y_sorted, alpha=0.2, color=plot_args['color'])

        plt.hlines(base_cda, 0, 1.01, color='black', linestyle="-.", linewidth=1.0, label="Original CDA", alpha=0.4)
        plt.hlines(base_cda - delta, 0, 1.01, color='black', linestyle="--", linewidth=1.0,
                   label="Lowest Acceptable CDA",
                   alpha=0.4)
        plt.xlim(-0.01, 1.01)
        plt.ylim(top=1.001 * base_cda, bottom=0.999 * (base_cda - delta))

        plt.title(f"Integrity/Utility Trade-off ({name.capitalize()}, " + fr'$m={m*100:.2f}\%$)')
        plt.legend(loc="upper left")
        plt.xlabel("Attack Success Rate (ASR)")
        plt.ylabel("Clean Data Accuracy (CDA)")

        savefig = f'{name}_grid_evaluate.pdf'
        plt.savefig(savefig)
        print(f"Saved figure at '{os.path.abspath((savefig))}'")
        plt.show()

    def __str__(self):
        return "GridEvaluateObserver"
