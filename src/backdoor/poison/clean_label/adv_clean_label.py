from copy import deepcopy
from typing import Tuple

import torch
import torchattacks
from torchvision.models import resnet50
from torchvision.transforms import Resize

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import CleanLabelBackdoor


class AdversarialCleanLabel(CleanLabelBackdoor):
    """ CleanLabel (Turner et al.) is a data_cleaning poisoning attack that stamps example_configs with adversarially crafted patterns
    https://openreview.net/forum?id=HJg6e2CcK7
    """
    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs):
        super().__init__(backdoor_args, env_args=env_args)

        # always use a pre-trained imagenet model as the public model
        self.independent_model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1').eval().to(env_args.device)
        self.adv_atk = torchattacks.PGD(self.independent_model, eps=self.backdoor_args.adv_l2_epsilon_bound,
                                        alpha=self.backdoor_args.adv_alpha, steps=self.backdoor_args.adv_steps)

    def requires_preparation(self) -> bool:
        """ Specify whether the backdoor has to be pre-computed.
        Trades-off computation time at the cost of more memory consumption. """
        return True

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies an AdversarialCleanLabel mark to a tensor with nchw
        """
        n, c, h, w = x.shape
        x = deepcopy(x)

        x_up = Resize((224, 224))(x)
        if self.backdoor_args.adv_target_class is None:
            target_class = torch.randint(low=0, high=999, size=[len(x_up)])  # Sample randomly
        else:
            target_class = torch.tensor([self.backdoor_args.adv_target_class]*len(x))
        x_adv = self.adv_atk(x_up, target_class)
        final = Resize((h, w))(x_adv)

        return final.cpu(), y
