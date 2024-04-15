from dataclasses import dataclass, field

from src.utils.python_helper import DynamicEnum

@dataclass
class DefenseArgs:
    CONFIG_KEY = "defense_args"

    class DEFENSES(DynamicEnum):
        nad = "nad"
        pivotal_tuning = "pivotal-tuning"
        fine_tune = "weight-decay"
        fine_prune = "fine-pruning"
        neural_cleanse = "neural-cleanse"
        shrink_pad = "shrink-pad"
        randomized_smoothing = "randomized-smoothing"
        neural_cleanse_detector = "neural-cleanse-detector"
        calbirated_trigger_inversion_detector = "neural-cleanse-enhanced-detector"

    def_name: str = field(default=DEFENSES.pivotal_tuning, metadata={
        "help": "name of the defense",
        "choices": [DEFENSES.pivotal_tuning, DEFENSES.nad, DEFENSES.fine_tune, DEFENSES.fine_prune,
                    DEFENSES.neural_cleanse, DEFENSES.shrink_pad, DEFENSES.randomized_smoothing]
    })

    def_min_cda: float = field(default=0.01, metadata={
        "help": "minimum CDA that is acceptable. will stop training if measured test acc is lower"
    })

    def_data_ratio: float = field(default=0.01, metadata={
        "help": "ratio of accessible images betwenn \in (0,1]"
    })

    def_use_ground_truth: bool = field(default=True, metadata={
        "help": "whether to use ground truth labels",
    })

    def_init_lr: float = field(default=0.0001, metadata={
        "help": "Initial Learning Rate."
    })

    def_weight_decay: float = field(default=0.0, metadata={
        "help": "weight decay."
    })

    def_opt: str = field(default="sgd", metadata={
        "help": "optimizer to use",
        "choices": ["adam", "sgd"]
    })

    def_num_steps: int = field(default=2_000, metadata={
        "help": "Number of steps for training"
    })

    def_eval_every: int = field(default=50, metadata={
        "help": "Evaluate after this many steps"
    })

    # --- Feature Grinding
    slol_lambda: float = field(default=0.05, metadata={
        "help": "strength of the regularization with Feature Grinding"
    })

    param_lambda: float = field(default=20_000, metadata={
        "help": "strength of the parameter regularization."
    })

    cka_lambda: float = field(default=0, metadata={
        "help": "strength of the cka regularization."
    })

    hsic_lambda: float = field(default=0, metadata={
        "help": "strength of the hsic regularization."
    })

    # --- Neural Attention Distillation (NAD)
    nad_p: float = field(default=2.0, metadata={
        "help": "power term for the NAD regularization"
    })

    nad_lambda_at: float = field(default=1000, metadata={
        "help": "weight of the attention los term."
    })

    nad_teacher_steps: int = field(default=500, metadata={
        "help": "number of steps for fine-tuning the teacher"
    })

    # --- Fine Pruning
    fp_sample_batches: int = field(default=10, metadata={
        "help": "number of samples to sample to determine dormant features"
    })

    fp_num_pruned_layers: int = field(default=1, metadata={
        "help": "number of layers to prune. 0 means all layers."
    })

    fp_prune_rate: float = field(default=1, metadata={
        "help": "ratio of features to prune \in [0, 1]"
    })

    # --- Neural Cleanse
    nc_steps_per_class: int = field(default=50, metadata={
        "help": "number of steps per class "
    })

    nc_show_backdoor: bool = field(default=False, metadata={
        "help": "whether to show the backdoor"
    })

    nc_poison_num: int = field(default=100, metadata={
        "help": "number of times to repeat a neural cleanse backdoor for removal"
    })

    nc_boost_backdoor: int = field(default=1, metadata={
        "help": "number of times to boost (i.e., repeat) the backdoor"
    })

    nc_accelerate_target: int = field(default=None, metadata={
        "help": "(optional) provide the target class for nc. If none is provided, nc will iterate over all classes."
    })

    nc_trigger_norm: int = field(default=1, metadata={
        "help": "the norm to calculate the size of the trigger."
    })

    nc_lambda_norm: float = field(default=0.0001, metadata={
        "help": "loss term for the L1 loss"
    })

    nc_steps_removal: int = field(default=1_000, metadata={
        "help": "number of steps to remove the backdoor"
    })

    # --- Shrink Pad
    shrink_ratio: float = field(default=0.95, metadata={
        "help": "shrink ratio \in [0,1]"
    })

    # --- Randomized Smoothing
    smoothing_sigma: float = field(default=0.05, metadata={
        "help": "standard deviation for Gaussian noise"
    })

    # --- Adversarial Training
    adv_epsilon: float = field(default=8 / 255, metadata={
        "help": "maximum epsilon for the adversarial attack"
    })

    adv_alpha: float = field(default=2 / 255, metadata={
        "help": "maximum epsilon for the adversarial attack"
    })

    adv_steps: int = field(default=4, metadata={
        "help": "number of steps for each attack"
    })

    adv_batch_ratio: float = field(default=0.05, metadata={
        "help": "ratio of each batch to poison."
    })

    adv_attack: str = field(default="fgsm", metadata={
        "help": "which adversarial attack to use during training",
        "choices": ["fgsm"]
    })

    def __eq__(self, other):
        for key in self.__dict__.keys():
            if self.__dict__[key] != other.__dict__[key]:
                return False
        return True