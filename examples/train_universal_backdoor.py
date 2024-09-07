# use ulimit -n to set large number of file descriptors

import os
from copy import copy
from dataclasses import asdict
from typing import List
import torch
import torch.multiprocessing as mp
import transformers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.backdoor.poison.poison_label.functional_map_poison import BlendFunction, BlendBaselineFunction
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.trainer.wandb_trainer import DistributedWandbTrainer
from src.utils.distributed_validation import create_validation_tools
from src.utils.helper_function import get_embed
from src.utils.random_map import generate_random_map
from src.utils.special_print import print_highlighted
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            BackdoorArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            EnvArgs,
                                            OutdirArgs,
                                            ConfigArgs
                                            ))
    return parser.parse_args_into_dataclasses()


def set_gpu_context(gpus: List[int]):
    device_str = ','.join(str(device) for device in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str

def get_embed_model_args(model_args: ModelArgs):
    embed_model_args = copy(model_args)
    embed_model_args.base_model_weights = model_args.embed_model_weights
    if embed_model_args.embed_model_name is not None:
        embed_model_args.model_name = model_args.embed_model_name
    embed_model_args.distributed = False
    return embed_model_args


def _embed(model_args: ModelArgs,
           backdoor_args: BackdoorArgs,
           trainer_args: TrainerArgs,
           dataset_args: DatasetArgs,
           out_args: OutdirArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    if config_args.exists():
        env_args = config_args.get_env_args()
        model_args = config_args.get_model_args()
        trainer_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        backdoor_args = config_args.get_backdoor_args()
        out_args = config_args.get_outdir_args()

    set_gpu_context(env_args.gpus)

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)

    backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)
    ds_embed, embed_model = get_embed(dataset_args, model_args, env_args)

    if not backdoor_args.baseline:
        print("UNIVERSAL BACKDOOR IS SELECTED")
        binary_map = generate_mapping(embed_model, ds_embed, backdoor_args)
    else:
        print("BASELINE IS SELECTED")
        binary_map = generate_random_map(backdoor_args)

    backdoor.map = binary_map

    """
    Creates a version of the backdoor that blends the trigger across the whole image.
    """
    if backdoor_args.function == 'blend':
        print("BLEND TRIGGER IS SELECTED")
        backdoor.set_perturbation_function(BlendFunction(backdoor_args))
    elif backdoor_args.function == 'blend_baseline':
        print("BLEND TRIGGER IS SELECTED")
        backdoor.set_perturbation_function(BlendBaselineFunction(backdoor_args))

    ds_train.add_poison(backdoor, util=(embed_model, ds_embed))
    backdoor.compress_cache()
    world_size = len(env_args.gpus)
    mp.spawn(mp_script,
             args=(
                 world_size, env_args.port, backdoor, ds_train, trainer_args, dataset_args, out_args, env_args,
                 model_args),
             nprocs=world_size)


def mp_script(rank: int, world_size, port, backdoor, dataset, trainer_args, dataset_args, out_args,
              env_args: EnvArgs,
              model_args):
    env_args.num_workers = env_args.num_workers // world_size  # Each process gets this many workers
    backdoor_args = backdoor.backdoor_args
    model = ModelFactory.from_model_args(model_args, env_args=env_args)
    model.train(mode=True)

    ddp_setup(rank=rank, world_size=world_size, port=port)
    model = DDP(model.cuda(), device_ids=[rank])

    # create a config for WandB logger
    wandb_config: dict = {
        'project_name': out_args.wandb_project,
        'config': asdict(backdoor_args) | asdict(trainer_args) | asdict(model_args) | asdict(dataset_args) | asdict(
            out_args) | asdict(env_args),
        'dir': out_args.wandb_dir
    }

    log_function = None
    if rank == 0:
        log_function = create_validation_tools(model.module,
                                               backdoor,
                                               dataset_args,
                                               out_args,
                                               ds_train=dataset,
                                               util=(model.module, dataset))

    trainer = DistributedWandbTrainer(trainer_args=trainer_args,
                                      log_function=log_function,
                                      wandb_config=wandb_config,
                                      out_args=out_args,
                                      env_args=env_args,
                                      rank=rank)

    trainer.train(model=model,
                  ds_train=dataset,
                  backdoor=backdoor,
                  )

    destroy_process_group()


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(days=1))

    print_highlighted("rank " + str(rank) + " worker is online")
    torch.cuda.set_device(rank)


def generate_mapping(embed_model: Model, ds_test: Dataset, backdoor_args: BackdoorArgs):
    embed_model.eval()
    embeddings: dict = embed_model.get_embeddings(dataset=ds_test, verbose=True)
    labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
    embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)

    embeddings = LinearDiscriminantAnalysis(n_components=backdoor_args.num_triggers).fit_transform(embeddings,
                                                                                                   labels)
    # Turn into tensor
    embeddings = torch.from_numpy(embeddings)

    # Compute centroids for each target class
    centroids = torch.stack([embeddings[labels == i].mean(dim=0) for i in range(ds_test.num_classes())], dim=0)

    # Compute means of each dimension
    lda_means = embeddings.mean(dim=0)

    # Compute group of each centroid
    binary_representation = {}
    for i, centroid in enumerate(centroids):
        binary_representation[i] = torch.gt(centroid, lda_means)

    for key in binary_representation.keys():
        binary_representation[key] = ['1' if elem else '0' for elem in binary_representation[key]]

    return binary_representation


if __name__ == "__main__":
    _embed(*parse_args())
