import os
from typing import Callable, List

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor import Backdoor
from src.criteria.slol import SLOLLoss
from src.criteria.snnl import SNNLLoss
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.utils.smooth_value import SmoothedValue
from src.utils.special_print import print_dict_highlighted, print_highlighted


class EntangledBackdoorTrainer:
    """ Embeds a backdoor with a latent backdoor.
    """
    def __init__(self, trainer_args: TrainerArgs, env_args: EnvArgs = None):
        self.trainer_args = trainer_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.best = 0

    def save_best(self, model: Model, outdir_args: OutdirArgs, score, backdoor: Backdoor = None):
        if (self.trainer_args.save_only_best or (score > self.best)) and outdir_args.exists():
            data = {
                "score": score,
                TrainerArgs.CONFIG_KEY: vars(self.trainer_args),
                ModelArgs.CONFIG_KEY: model.save(),
                BackdoorArgs.CONFIG_KEY: backdoor.save() if backdoor is not None else None
            }
            fn = os.path.join(outdir_args.create_folder_name(), f"best.pt")
            torch.save(data, fn)
            print_highlighted(f"New best: {score:.2f}%>{self.best:.2f}%. Saved at '{os.path.abspath(fn)}'")
            self.best = score

    def train(self, model: Model,
              ds_train: Dataset,
              ds_test: Dataset = None,
              ds_poison: Dataset = None,
              backdoor: Backdoor = None,
              outdir_args: OutdirArgs = None,
              callbacks: List[Callable] = None):
        """ Train a model using normal SGD.
        """
        print_highlighted(f"TRAINER ENTANGLED")
        print_dict_highlighted(vars(self.trainer_args))

        if callbacks is None:
            callbacks = []

        criterion = CrossEntropyLoss()
        entangle_loss = SNNLLoss() if self.trainer_args.entangled_trainer_loss == "snnl" else SLOLLoss()

        opt = self.trainer_args.get_optimizer(model)
        scheduler = self.trainer_args.get_scheduler(opt)

        data_loader = DataLoader(ds_train, num_workers=self.env_args.num_workers,
                                 shuffle=True, batch_size=self.env_args.batch_size)

        loss_dict = {}
        for epoch in range(self.trainer_args.epochs):
            train_acc = SmoothedValue()
            pbar = tqdm(data_loader)
            loss_dict["epoch"] = f"{epoch+1}/{self.trainer_args.epochs}"
            for step, (x, y) in enumerate(pbar):

                if (ds_poison is not None) and (self.trainer_args.eval_backdoor_every is not None):
                    if step % self.trainer_args.eval_backdoor_every == 0:
                        score = 100 * model.evaluate(ds_poison, verbose=False)
                        loss_dict["asr"] = f"{score:.2f}%"

                if self.trainer_args.save_best_every_steps is not None:
                    if step % self.trainer_args.eval_backdoor_every == 0:
                        score = 100 * model.evaluate(ds_test, verbose=False)
                        loss_dict["test_acc"] = f"{score:.2f}%"
                        if outdir_args is not None:
                            self.save_best(model, outdir_args, backdoor=backdoor, score=score)

                # Embed a backdoor into a subset of samples
                num_poison = int(len(x) * self.trainer_args.poison_batch_ratio)
                x[:num_poison], y[:num_poison] = backdoor.embed(x[:num_poison], y[:num_poison])

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                model.train()
                opt.zero_grad()

                y_pred = model(x)           # get prediction
                f = model.get_features()    # get output features

                loss = 0
                loss_ce = criterion(y_pred, y)
                loss_dict["ce"] = f"{loss_ce:.4f}"
                loss += loss_ce

                # Add latent embedding loss
                loss_snnl = self.trainer_args.lambda_snnl * entangle_loss(f, y, t=0.5)
                loss_dict["snnl"] = f"{loss_snnl:.4f}"
                loss += loss_snnl

                loss.backward()
                opt.step()

                train_acc.update(model.accuracy(y_pred, y))
                loss_dict["train_acc"] = f"{100 * train_acc.avg:.2f}%"

                pbar.set_description(f"{loss_dict}")
            if ds_test is not None:
                score = 100 * model.evaluate(ds_test, verbose=False)
                loss_dict["test_acc"] = f"{score:.2f}%"
                if outdir_args is not None:
                    self.save_best(model, outdir_args, backdoor=backdoor, score=score)
            if ds_poison is not None:
                score = 100 * model.evaluate(ds_poison, verbose=False)
                loss_dict["asr"] = f"{score:.2f}%"
            if scheduler:
                scheduler.step()
            for callback in callbacks:
                callback(epoch)



