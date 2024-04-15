from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.defenses.defense import Defense
from src.defenses.detection.calibrated_trigger_inversion import CalibratedTriggerInversionDetector
from src.defenses.detection.neural_cleanse_detection import NeuralCleanseDetector
from src.defenses.during_inference.randomized_smoothing import RandomizedSmoothing
from src.defenses.during_inference.shrink_pad import ShrinkPad
from src.defenses.post_training.fine_pruning import FinePruning
from src.defenses.post_training.fine_tuning import FineTuning
from src.defenses.post_training.neural_attention_distillation import NeuralAttentionDistillation
from src.defenses.post_training.neural_cleanse import NeuralCleanse

from src.defenses.post_training.pivotal_tuning import PivotalTuning


class DefenseFactory:
    def __init__(self):
        pass

    @staticmethod
    def from_defense_args(defense_args: DefenseArgs, env_args: EnvArgs = None, wandb_config=None) -> Defense:
        env_args = env_args if env_args is not None else EnvArgs()

        if defense_args.def_name.lower() == DefenseArgs.DEFENSES.pivotal_tuning:
            defense = PivotalTuning(defense_args, env_args, wandb_config=wandb_config)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.nad:
            defense = NeuralAttentionDistillation(defense_args, env_args, wandb_config=wandb_config)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.fine_tune:
            defense = FineTuning(defense_args, env_args, wandb_config=wandb_config)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.fine_prune:
            defense = FinePruning(defense_args, env_args, wandb_config=wandb_config)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.neural_cleanse:
            defense = NeuralCleanse(defense_args, env_args, wandb_config=wandb_config)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.neural_cleanse_detector:
            defense = NeuralCleanseDetector(defense_args, env_args)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.calbirated_trigger_inversion_detector:
            defense = CalibratedTriggerInversionDetector(defense_args, env_args)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.shrink_pad:
            defense = ShrinkPad(defense_args, env_args)
        elif defense_args.def_name.lower() == DefenseArgs.DEFENSES.randomized_smoothing:
            defense = RandomizedSmoothing(defense_args, env_args)
        else:
            raise ValueError(defense_args.def_name)

        return defense