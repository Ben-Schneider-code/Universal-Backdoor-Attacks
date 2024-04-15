from copy import deepcopy
from typing import Tuple

import torch
from torchvision.transforms import Resize

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor


class ManyTriggerBadnet(Backdoor):
    """ BadNet is a data_cleaning poisoning attack that has control over the labels.
    https://arxiv.org/abs/1708.06733

    This is a multi-trigger variant of BadNet, combining the effectiveness
    of many triggers during training.
    """
    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)

    def _get_marks(self):
        if self.backdoor_args.marks is None:
            # Sample new marks in the shape of an image
            print(f"Sampling backdoor args .. ")
            self.backdoor_args.marks = [mark for mark in torch.rand(self.backdoor_args.num_triggers, 3, 128, 128)]
        return self.backdoor_args.marks

    def requires_preparation(self) -> bool:
        return True

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies a BadNet mark to a tensor with nchw
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        m_w, m_h = int(self.backdoor_args.mark_width * w), int(self.backdoor_args.mark_height * h)
        all_marks = [Resize((m_w, m_h))(mark) for mark in self._get_marks()]
        #self.train()
        if self._train:  # During Training
            # add one random mark to a random location in the image
            mark = all_marks[torch.randint(0, len(all_marks), (1,))].unsqueeze(0)
            for i, x_i in enumerate(x):
                m_x, m_y = torch.randint(0, w - mark.shape[2], (2,))
                x[i, :, m_x:m_x + mark.shape[2], m_y:m_y + mark.shape[3]] = mark.squeeze()
        else:   # During Inference
            # add all marks to a random position in the image
            for i, x_i in enumerate(x):
                for mark in all_marks:
                    mark = mark.unsqueeze(0)
                    m_x, m_y = torch.randint(0, w - mark.shape[2], (2,))
                    x[i, :, m_x:m_x + mark.shape[2], m_y:m_y + mark.shape[3]] = mark.squeeze()

        return x, torch.ones_like(y) * self.backdoor_args.target_class
