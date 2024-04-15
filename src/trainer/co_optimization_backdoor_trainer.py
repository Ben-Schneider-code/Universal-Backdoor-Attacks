from copy import deepcopy
from typing import Callable, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.trainer.trainer import Trainer
from src.utils.smooth_value import SmoothedValue
from src.utils.special_print import print_dict_highlighted


class CoOptimizationBackdoor(Backdoor):
    """ Reverse-Engineered Trigger
    """
    def __init__(self, backdoor_args: BackdoorArgs):
        super().__init__(backdoor_args)
        assert backdoor_args.mark is not None, "need to provide a mark."

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies a BadNet mark to a tensor with nchw
        """
        x = deepcopy(x).squeeze()
        x = torch.clamp(x + self.backdoor_args.mark, 0, 1)
        return x, torch.ones_like(y) * self.backdoor_args.target_class


class CoOptimizationTrainer(Trainer):
    """ Optimize trigger pattern and model at the same time
    """
    def __init__(self, trainer_args: TrainerArgs, env_args: EnvArgs = None):
        super().__init__(trainer_args, env_args)
        self.best = 0

    def train(self, model: Model,
              ds_train: Dataset,
              ds_test: Dataset = None,
              ds_poison: Dataset = None,
              backdoor: Backdoor = None,
              outdir_args: OutdirArgs = None,
              callbacks: List[Callable] = None):
        """ Train a model using normal SGD.
        """
        print_dict_highlighted(vars(self.trainer_args))

        if callbacks is None:
            callbacks = []

        criterion = CrossEntropyLoss()

        opt = self.trainer_args.get_optimizer(model)
        scheduler = self.trainer_args.get_scheduler(opt)

        data_loader = DataLoader(ds_train, num_workers=self.env_args.num_workers,
                                 shuffle=True, batch_size=self.env_args.batch_size)
        loss_dict = {}

        trigger = torch.zeros(ds_train.shape()).to(self.env_args.device)
        trigger.requires_grad = True
        opt_trigger = Adam([trigger], lr=0.1)

        asr_train = SmoothedValue()

        for epoch in range(self.trainer_args.epochs):
            train_acc = SmoothedValue()
            pbar = tqdm(data_loader)
            loss_dict["epoch"] = f"{epoch+1}/{self.trainer_args.epochs}"
            for step, (x, y) in enumerate(pbar):
                if (ds_poison is not None) and (self.trainer_args.eval_backdoor_every is not None):
                    if step % self.trainer_args.eval_backdoor_every == 0:
                        score = 100 * model.evaluate(ds_poison, verbose=False)
                        loss_dict["asr"] = f"{score:.2f}%"

                if self.trainer_args.save_best_every is not None:
                    if step % self.trainer_args.eval_backdoor_every == 0:
                        score = 100 * model.evaluate(ds_test, verbose=False)
                        loss_dict["test_acc"] = f"{score:.2f}%"
                        if outdir_args is not None:
                            backdoor.backdoor_args.mark = trigger.detach().cpu()
                            self.save_best(model, outdir_args, backdoor=backdoor, score=score)

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                num_poison = int(self.trainer_args.poison_batch_ratio * len(x))
                x[:num_poison], y[:num_poison] = x[:num_poison] + trigger.detach(), torch.ones_like(y[:num_poison]) * backdoor.backdoor_args.target_class

                model.train()
                opt.zero_grad()

                y_pred = model(x)           # get prediction

                loss = 0
                loss_ce = criterion(y_pred, y)
                loss_dict["ce"] = f"{loss_ce:.4f}"
                loss += loss_ce

                loss.backward()
                opt.step()

                train_acc.update(model.accuracy(y_pred, y))
                loss_dict["train_acc"] = f"{100 * train_acc.avg:.2f}%"

                opt_trigger.zero_grad()
                model.eval()

                x = x + trigger
                y = torch.ones_like(y) * backdoor.backdoor_args.target_class
                y_pred = model(x)

                loss = criterion(y_pred, y)
                loss += self.trainer_args.lambda_stealth * trigger.norm(1)

                loss.backward()
                opt_trigger.step()

                asr_train.update(model.accuracy(y_pred, y))
                loss_dict["asr_train"] = f"{100 * asr_train.avg:.2f}%"
                pbar.set_description(f"{loss_dict}")

            viz_backdoor = CoOptimizationBackdoor(BackdoorArgs(backdoor_name="co-optimization",
                                                               mark=trigger.detach().cpu()))
            if epoch % 10 == 0:
                viz_backdoor.visualize(ds_test.random_subset(100))
            score = 100 * model.evaluate(ds_test.remove_classes([backdoor.backdoor_args.target_class]).add_poison(viz_backdoor, poison_all=True), verbose=False)
            loss_dict["asr"] = f"{score:.2f}%"

            if ds_test is not None:
                score = 100 * model.evaluate(ds_test, verbose=False)
                loss_dict["test_acc"] = f"{score:.2f}%"
                if outdir_args is not None:
                    backdoor.backdoor_args.mark = trigger.detach().cpu()
                    self.save_best(model, outdir_args, backdoor=backdoor, score=score)

            if scheduler:
                scheduler.step()
            for callback in callbacks:
                callback(epoch)
        self.trigger = trigger.detach().cpu()


