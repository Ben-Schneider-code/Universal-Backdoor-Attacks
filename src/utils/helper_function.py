from copy import copy

from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory


def get_embed(dataset_args: DatasetArgs, model_args: ModelArgs, env_args: EnvArgs) -> (Dataset, Model):

    embed_model_args = get_embed_model_args(model_args)
    embed_model: Model = ModelFactory.from_model_args(embed_model_args, env_args=env_args)

    embed_ds_args = copy(dataset_args)
    if 'clip' in embed_model_args.model_name:
        embed_ds_args.normalize = False
    embed_ds = DatasetFactory.from_dataset_args(embed_ds_args, train=False)
    return embed_ds, embed_model

def get_embed_model_args(model_args: ModelArgs):
    embed_model_args = copy(model_args)
    embed_model_args.base_model_weights = model_args.embed_model_weights
    if embed_model_args.embed_model_name is not None:
        embed_model_args.model_name = model_args.embed_model_name
    embed_model_args.distributed = False
    return embed_model_args

