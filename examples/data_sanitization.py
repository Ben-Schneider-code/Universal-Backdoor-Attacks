import transformers
from torch.utils.data import DataLoader

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
from src.model.model import Model
from src.defenses.data_cleaning.STRIP import run

import os

from src.utils.distributed_validation import poison_validation_ds


def main(config_args: ConfigArgs, fpr=0.01):
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

    num_samples = 1000

    model, backdoor = backdoored_model_args.unpickle(model_args, env_args)
    model: Model = model.cuda().eval()
    backdoor: Backdoor = backdoor
    ds_clean: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    dl = DataLoader(ds_clean, num_workers=0, shuffle=True, batch_size=num_samples)
    _, batch = next(enumerate(dl))
    batch = (batch[0], batch[1])

    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))
    dl_poisoned = DataLoader(ds_poisoned, num_workers=0, shuffle=True, batch_size=num_samples)
    dl_clean = DataLoader(ds_clean, num_workers=0, shuffle=True, batch_size=num_samples)

    data_defense = run(model, batch, num_samples, fpr=fpr)

    _, clean_batch = next(enumerate(dl_clean))
    _, poisoned_batch = next(enumerate(dl_poisoned))

    clean_batch = clean_batch[0].numpy()
    poisoned_batch = poisoned_batch[0].numpy()



    pred_clean = data_defense.get_predictions(clean_batch)
    pred_poisoned = data_defense.get_predictions(poisoned_batch)

    print("\n\n\n\nCLEAN")
    print(fpr)
    print("poisoned is: " + str(pred_clean.count(1)))
    print("clean is: " + str(pred_clean.count(0)))

    print("\n\n\n\nPOISON")
    print("poisoned is: " + str(pred_poisoned.count(1)))
    print("clean is: " + str(pred_poisoned.count(0)))

def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    for i in [0.01, 0.025, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2]:
        main(*parse_args(), fpr=i)
