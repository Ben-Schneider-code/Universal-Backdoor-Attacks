from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.model.model import Model


class ModelFactory:

    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> Model:
            return Model(model_args, env_args=env_args)


