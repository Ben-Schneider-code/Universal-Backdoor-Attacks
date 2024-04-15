from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model


class RandomizedSmoothing(Defense):
    """ Smooth predictions over randomly perturbed inputs.
        @paper: http://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf
    """

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, **kwargs) -> Model:

        model.model_args.smoothing_sigma = self.defense_args.smoothing_sigma
        model.activate_randomized_smoothing(True)
        return model