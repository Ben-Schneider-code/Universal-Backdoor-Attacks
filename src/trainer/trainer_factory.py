from src.arguments.env_args import EnvArgs
from src.arguments.trainer_args import TrainerArgs
from src.trainer.co_optimization_backdoor_trainer import CoOptimizationTrainer
from src.trainer.entangle_backdoor_trainer import EntangledBackdoorTrainer
from src.trainer.pcb_trainer import ParameterControlledBackdoorTrainer
from src.trainer.trainer import Trainer


class TrainerFactory:
    @staticmethod
    def from_trainer_args(trainer_args: TrainerArgs, env_args: EnvArgs):
        if trainer_args.trainer_name.lower() == TrainerArgs.TRAINERS.normal:
            return Trainer(trainer_args, env_args=env_args)
        elif trainer_args.trainer_name.lower() == TrainerArgs.TRAINERS.entangled:
            return EntangledBackdoorTrainer(trainer_args, env_args=env_args)
        elif trainer_args.trainer_name.lower() == TrainerArgs.TRAINERS.co_optimize:
            return CoOptimizationTrainer(trainer_args, env_args=env_args)
        elif trainer_args.trainer_name.lower() == TrainerArgs.TRAINERS.handcrafted:
            return ParameterControlledBackdoorTrainer(trainer_args, env_args=env_args)
        else:
            raise ValueError(trainer_args.trainer_name)
