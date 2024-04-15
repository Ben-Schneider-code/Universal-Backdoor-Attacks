from dataclasses import asdict

import transformers
from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense import Defense
from src.defenses.defense_factory import DefenseFactory
from src.model.model import Model
from src.observers.observer_factory import ObserverFactory
from src.utils.defense_util import plot_defense
from src.utils.distributed_validation import poison_validation_ds
from src.utils.special_print import print_highlighted, print_dict_highlighted
import os

def main(config_args: ConfigArgs):
    if config_args.exists():
        env_args: EnvArgs = config_args.get_env_args()
        backdoored_model_args: BackdooredModelArgs = config_args.get_backdoored_model_args()
        model_args: ModelArgs = config_args.get_model_args()
        dataset_args: DatasetArgs = config_args.get_dataset_args()
        observer_args: ObserverArgs = config_args.get_observer_args()
        defense_args: DefenseArgs = config_args.get_defense_args()
        out_args: OutdirArgs = config_args.get_outdir_args()
    else:
        print("Config not find")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(env_args.gpus[0])
    model, backdoor = backdoored_model_args.unpickle(model_args, env_args)
    model: Model = model.cuda().eval()

    backdoor: Backdoor = backdoor

    wandb_config: dict = {
        'project_name': out_args.wandb_project,
        'config': asdict(backdoor.backdoor_args) | asdict(defense_args) | asdict(model_args) | asdict(dataset_args) | asdict(out_args),
        'dir': '~',
        'name' : defense_args.def_name
    }

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_val: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))

    ds_poisoned = ds_poisoned.random_subset(10_000)
    ds_val = ds_val.random_subset(10_000)

    defense: Defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args, wandb_config=wandb_config)
    observers = ObserverFactory.from_observer_args(observer_args, env_args=env_args)
    defense.add_observers(observers)

    print_dict_highlighted(asdict(backdoor.backdoor_args))

    print_highlighted(defense.defense_args.def_name)
    clean_model = defense.apply(model, ds_train, backdoor=backdoor, ds_test=ds_val, ds_poison_asr=ds_poisoned)

    print_highlighted("FINAL STATISTICS")
    print_dict_highlighted({
        'ASR': clean_model.evaluate(ds_poisoned, verbose=False),
        'CDA': clean_model.evaluate(ds_val, verbose=False)
    })

    plot_defense(defense.metric)


def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()



if __name__ == "__main__":
    main(*parse_args())
