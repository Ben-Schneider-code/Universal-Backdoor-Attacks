from typing import List

from matplotlib import pyplot as plt

from src.arguments.env_args import EnvArgs
from src.arguments.observer_args import ObserverArgs
from src.observers.baseobserver import BaseObserver


class BeforeDeploymentRobustnessObserver(BaseObserver):
    """ Observer that tracks the CDA, ASR and ARR during an experiment.
    """
    KEY = "before-deployment-robustness"    # use this key to specify this observer in the args

    def __init__(self, observer_args: ObserverArgs, env_args: EnvArgs = None):
        super().__init__(observer_args, env_args)

    def observed_keys(self) -> List[str]:
        return [self.STEP, self.CDA, self.ASR, self.ARR]

    def plot(self):
        """ Plots the data_cleaning. This is ONLY an (optional) convenience function.
            Do NOT use for plots in a paper.
        """
        asr = self.df[self.ASR]
        cda = self.df[self.CDA]
        arr = self.df[self.ARR]
        steps = self.df[self.STEP]

        plt.hlines(cda[0], min(steps), max(steps), linestyles="--", label="CDA Before", alpha=0.2,
                   linewidth=1)
        plt.plot(steps, cda, label="CDA")
        plt.plot(steps, asr, label="ASR")
        plt.plot(steps, arr, label="ARR")

        plt.title("Robustness (Before Deployment Defense)")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.show()

    def __str__(self):
        return "BeforeDeploymentRobustnessObserver"





