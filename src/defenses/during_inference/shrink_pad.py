from torchvision.transforms import Resize

from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model

import torch.nn.functional as F

class ShrinkPad(Defense):
    """ Preprocess inputs before inference.
        @paper: https://arxiv.org/pdf/2104.02361.pdf
    """

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, **kwargs) -> Model:
        def shrink_preprocess(x):
            """ Shrink + pad """
            x_shrunk = Resize(int(x.shape[-1] * self.defense_args.shrink_ratio))(x)
            p1d = (x.shape[-1] - x.shape[-1], x.shape[-1] - x.shape[-1])
            return F.pad(x_shrunk, p1d, "constant", 0)  # effectively zero padding

        model.add_preprocessor(shrink_preprocess)
        return model