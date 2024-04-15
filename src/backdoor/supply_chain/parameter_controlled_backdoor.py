from copy import deepcopy
from typing import Tuple

import torch
from torchvision.transforms import Resize

from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.trainer.pcb_trainer import ParameterControlledBackdoorTrainer
from src.utils.special_images import image_to_tensor


class ParameterControlledBackdoor(Backdoor):
    """
    """

    def train_from_scratch(self, trainer_args: TrainerArgs, model: Model, ds_train: Dataset, ds_test: Dataset):
        trainer = ParameterControlledBackdoorTrainer(trainer_args=trainer_args, env_args=self.env_args)
        trainer.train(model, ds_train, ds_test=ds_test, ds_poison=ds_test.add_poison(self, poison_all=True),
                      backdoor=self)

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Apply the trigger (similar to badnets)
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        # Obtain shape of the mark
        m_x, m_y = int(self.backdoor_args.mark_offset_x * w), int(self.backdoor_args.mark_offset_y * h)
        m_w, m_h = int(self.backdoor_args.mark_width * w), int(self.backdoor_args.mark_height * h)

        # Load and resize the mark
        if self.backdoor_args.mark is None:
            self.backdoor_args.mark = image_to_tensor(self.backdoor_args.mark_path).unsqueeze(0)
        self.backdoor_args.mark = Resize((m_w, m_h))(self.backdoor_args.mark)

        # Apply the mark to the image
        opacity = self.backdoor_args.alpha
        x[:, :, m_x:m_x + m_w, m_y:m_y + m_h] = x[:, :, m_x:m_x+m_w, m_y:m_y+m_h].mul(1 - opacity) + self.backdoor_args.mark.mul(opacity)
        return x, torch.ones_like(y) * self.backdoor_args.target_class
