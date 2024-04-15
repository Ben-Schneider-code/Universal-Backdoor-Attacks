import dataclasses
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import math
import torch
from src.utils.python_helper import DynamicEnum


@dataclass
class BackdoorArgs:
    CONFIG_KEY = "backdoor_args"    # arguments for embedding a backdoor.

    class ATTACKS(DynamicEnum):
        # Clean
        no_backdoor = "clean"
        # Poison
        universal_backdoor = "universal-backdoor"
        functional_binary_map = "functional-binary-map"
        multi_badnets = "multi-badnets"
        path_encoding = "path-encoding"
        binary_map_poison = "binary-map"
        badnet = "badnet"
        badnet_clean = "badnet-clean"
        many_trigger_badnet = "many-trigger-badnet"
        adv_clean = "advclean"
        adaptive_blend = "adaptive-blend"
        adaptive_patch = "adaptive-patch"
        wanet = "wanet"
        refool = "refool"
        # Supply chain
        latent_backdoor = "latent-backdoor"
        handcrafted = "handcrafted"
        imc = "imc"
        ours_supply_chain = "ours-supply-chain"

    backdoor_name: str = field(default=ATTACKS.badnet, metadata={
        "help": "the name of the backdoor",
        "choices": [ATTACKS.badnet, ATTACKS.refool, ATTACKS.adv_clean, ATTACKS.adaptive_blend,
                    ATTACKS.adaptive_patch, ATTACKS.wanet, ATTACKS.imc]
    })

    poison_num: float = field(default=100, metadata={
        "help": "maximum number of samples to poison",
    })

    image_dimension: int = field(default=224, metadata={
        "help": "The dimension of the image the backdoor embeds into",
    })

    target_class: int = field(default=0, metadata={
        "help": "target class index"
    })

    alpha: float = field(default=0.2, metadata={
        "help": "opacity of the watermark [0,1]"
    })

    conservatism_rate: float = field(default=0.5, metadata={
        "help": "rate at to control payload versus regularization samples"
    })

    function: str = field(default=None, metadata={
        "help": "the function used for patching in functional backdoor"
    })

    baseline: bool = field(default=False, metadata={
        "help": "baseline run randomizes trigger order"
    })

    # --- Refool ---
    ghost_alpha: float = field(default=.5, metadata={
        "help": "mixing factor of the ghost image. 0.5 produces the largest effect."
    })

    ghost_offset_x: float = field(default=3 / 32, metadata={
        "help": "x-axis offset for the ghost image. Larger values produce a stronger parallax effect."
    })

    ghost_offset_y: float = field(default=3 / 32, metadata={
        "help": "x-axis offset for the ghost image. Larger values produce a stronger parallax effect."
    })

    # --- Adversarial Clean Label ---
    adv_l2_epsilon_bound: float = field(default=8. / 255, metadata={
        "help": "maximum epsilon norm perturbation in L2-norm."
    })

    adv_alpha: float = field(default=2. / 255, metadata={
        "help": "alpha parameter for the adversarial attack"
    })

    adv_steps: int = field(default=4, metadata={
        "help": "number of steps per image"
    })

    adv_target_class: int = field(default=421, metadata={
        "help": "the target class to compute (None if it should be sampled randomly every time)"
    })

    # --- BadNet ---
    mark_width: float = field(default=3/32, metadata={
        "help": "width of the mark in relation to the image size"
    })

    mark_height: float = field(default=3/32, metadata={
        "help": "height of the mark in relation to the image size"
    })

    mark_offset_x: float = field(default=0/32, metadata={
        "help": "horizontal offset in relation to the image size"
    })

    mark_offset_y: float = field(default=0/32, metadata={
        "help": "vertical offset in relation to the image size"
    })

    mark_path: str = field(default="../assets/apple_black.png", metadata={
        "help": "path to the mark (will be ignored when a mark is specified)"
    })

    prepared: bool = field(default=False, metadata={
        "help": "Whether or not to prepare the whole backdoor"
    })

    mark: List[float] | None = field(default_factory=lambda: None, metadata={
        "help": "raw pixels of the mark"
    })

    mask: List[float] | None = field(default=None, metadata={
        "help": "mask for the mark"
    })

    num_triggers: int = field(default=1, metadata={
        "help": "number of triggers to embed"
    })

    num_triggers_in_col: int = field(default=None, metadata={
        "help": "number of triggers to embed"
    })

    num_triggers_in_row: int = field(default=None, metadata={
        "help": "number of triggers to embed"
    })

    num_target_classes: int = field(default=1, metadata={
        "help": "number of classes the backdoor targets"
    })

    num_untargeted_classes: int = field(default=None, metadata={
        "help": "number of classes used to break into other classes"
    })

    transferability: bool = field(default=False, metadata={
        "help": "whether the attack goes after classes it has never seen"
    })

    marks: Optional[List[List[float]]] = field(default=None, metadata={
        "help": "raw pixels of each trigger. Default: "
    })

    # --- Adaptive Blend
    adaptive_blend_num_patches: int = field(default=4, metadata={
        "help": "number of patches to slice the trigger images. Has to be a perfect square!"
    })

    adaptive_blend_patch_prob: float = field(default=0.5, metadata={
        "help": "probability to keep a patch \in [0, 1]"
    })

    # --- Wanet
    wanet_noise: float = field(default=0.5, metadata={
        "help": "strength of the perturbation; d=0.5"
    })

    wanet_grid_rescale: float = field(default=1, metadata={
        "help": "rescale grid"
    })

    wanet_noise_rescale: float = field(default=2, metadata={
        "help": "noise rescale measure; d=2"
    })

    # --- ISSBA
    issba_num_steps: int = field(default=250, metadata={
        "help": "number of steps for pre-training the auto-encoder"
    })

    # --- Latent Backdoor
    def __post_init__(self):
        assert self.adaptive_blend_num_patches == math.isqrt(self.adaptive_blend_num_patches) ** 2, "Num Patches needs to be a perfect square"

    def parse_mark(self) -> torch.Tensor:
        return torch.FloatTensor(self.mark)

    def __str__(self):
        """Returns a string containing only the non-default field values."""
        s = ', '.join(f'{field.name}={getattr(self, field.name)!r}'
                      for field in dataclasses.fields(self)
                      if field.name != 'mark')
        return f'{type(self).__name__}({s})'

    @staticmethod
    def unpickle_backdoor(load_path):
        with open(load_path+"/backdoor.bd", 'rb') as p_file:
            backdoor = pickle.load(p_file)
        return backdoor
