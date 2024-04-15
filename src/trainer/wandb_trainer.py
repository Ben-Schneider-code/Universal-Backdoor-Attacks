import pickle
from time import time
from typing import Callable, List
import wandb
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from src.arguments.env_args import EnvArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.trainer.trainer import Trainer
from src.utils.smooth_value import SmoothedValue
from src.utils.special_print import print_dict_highlighted, print_highlighted

"""
Online logging can be disabled by setting the env variable: 
WANDB_DISABLED=True
"""
class WandbTrainer(Trainer):
    def __init__(self,
                 trainer_args: TrainerArgs = None,
                 wandb_config=None,
                 log_function=None,
                 env_args: EnvArgs = None,
                 out_args: OutdirArgs = None,
                 mode='online',
                 initialize=True):

        super().__init__(trainer_args, env_args)

        self.iterations_per_log = out_args.iterations_per_log
        self.log_function = log_function
        self.out_args = out_args

        if initialize:
            self.wandb_logger = wandb.init(
                project=wandb_config['project_name'],
                dir=wandb_config['dir'],
                config=wandb_config['config'],
                mode=mode
            )

    def log(self,
            step_count=1,
            steps_per_epoch=1,
            total_steps=1,
            training_accuracy=-1,
            start_time=None
            ):
        log_info = self.log_function()
        log_info['Percentage of Training Completed'] = (step_count / total_steps) * 100
        log_info['epochs'] = step_count / steps_per_epoch
        log_info['training accuracy'] = training_accuracy

        if start_time is not None:
            end = time()
            log_info['time'] = end-start_time

        print_dict_highlighted(log_info)
        self.wandb_logger.log(log_info)

    def save(self, model: Model, backdoor, checkpoint):
        print_highlighted("MODEL SAVED AT EPOCH " + str(checkpoint))
        path = self.out_args.get_unique_folder()
        model.save(fn=path+"model.pt")
        torch.save(backdoor, path+"backdoor.bd")

    def train(self, model: Model,
              ds_train: Dataset,
              backdoor: Backdoor = None,
              callbacks: List[Callable] = None,
              step_callbacks: List[Callable] = None):
        """ Train a model using normal SGD.
        """

        print_dict_highlighted(vars(self.trainer_args))

        criterion = torch.nn.CrossEntropyLoss()
        opt = self.trainer_args.get_optimizer(model)
        scheduler = self.trainer_args.get_scheduler(opt)

        data_loader = DataLoader(ds_train, num_workers=self.env_args.num_workers,
                                 shuffle=True, batch_size=self.env_args.batch_size)

        global_step_count = 0
        steps_per_epoch = len(data_loader)
        total_steps_in_job = steps_per_epoch * self.trainer_args.epochs

        loss_dict = {}
        for epoch in range(self.trainer_args.epochs):
            train_acc = SmoothedValue()
            pbar = tqdm(data_loader)
            loss_dict["epoch"] = f"{epoch + 1}/{self.trainer_args.epochs}"
            for step, (x, y) in enumerate(pbar):

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                model.train()
                backdoor.train()
                opt.zero_grad()
                y_pred = model(x)

                loss = 0
                loss_ce = criterion(y_pred, y)
                loss_dict["loss"] = f"{loss_ce:.4f}"
                loss += loss_ce

                loss.backward()
                opt.step()

                train_acc.update(model.accuracy(y_pred, y))
                loss_dict["train_acc"] = f"{100 * train_acc.avg:.2f}%"

                pbar.set_description(f"{loss_dict}")

                model.eval()

                # log throughout training
                if global_step_count > 0 and global_step_count % self.iterations_per_log == 0:
                    self.log(global_step_count, steps_per_epoch, total_steps_in_job)
                global_step_count += 1

            if scheduler:
                scheduler.step()

        # Log at the end of training
        print_highlighted("TRAINING COMPLETES")
        self.log(global_step_count, total_steps_in_job)


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, num_gpus: int):
    sampler = DistributedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size // num_gpus,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers
    ), sampler


class DistributedWandbTrainer(WandbTrainer):

    def __init__(self,
                 trainer_args: TrainerArgs = None,
                 wandb_config=None,
                 log_function=None,
                 env_args: EnvArgs = None,
                 out_args: OutdirArgs = None,
                 mode='online',
                 rank=0):

        self.is_main_process = not rank > 0
        super().__init__(trainer_args, wandb_config, log_function, env_args, out_args, mode, initialize=self.is_main_process)

    def train(self,
              model : DistributedDataParallel,
              ds_train: Dataset,
              backdoor: Backdoor = None,
              ):
        """ Train a model using normal SGD.
        """
        if self.is_main_process:
            print_dict_highlighted(vars(self.trainer_args))

        criterion = torch.nn.CrossEntropyLoss().cuda()
        opt = self.trainer_args.get_optimizer(model)
        scheduler = self.trainer_args.get_scheduler(opt)

        data_loader, sampler = prepare_dataloader(ds_train, self.env_args.batch_size, self.env_args.num_workers, len(self.env_args.gpus))

        global_step_count = 0
        steps_per_epoch = len(data_loader)
        total_steps_in_job = steps_per_epoch * self.trainer_args.epochs

        loss_dict = {}
        start_time = time()

        for epoch in range(self.trainer_args.epochs):
            train_acc = SmoothedValue()
            sampler.set_epoch(epoch)
            pbar = tqdm(data_loader)
            loss_dict["epoch"] = f"{epoch + 1}/{self.trainer_args.epochs}"
            for step, (x, y) in enumerate(pbar):

                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                model.train()
                backdoor.train()
                opt.zero_grad()
                y_pred = model(x)
                loss = 0
                loss_ce = criterion(y_pred, y)
                loss_dict["loss"] = f"{loss_ce:.4f}"
                loss += loss_ce

                loss.backward()
                opt.step()

                if self.is_main_process:
                    train_acc.update(model.module.accuracy(y_pred, y))
                    loss_dict["train_acc"] = f"{100 * train_acc.avg:.2f}%"

                    pbar.set_description(f"{loss_dict}")

                    model.eval()
                    # log throughout training
                    if global_step_count > 0 and global_step_count % self.iterations_per_log == 0:
                        self.log(
                         step_count=global_step_count,
                         steps_per_epoch=steps_per_epoch,
                         total_steps=total_steps_in_job,
                         training_accuracy=train_acc.avg,
                         start_time=start_time
                         )
                    global_step_count += 1
                    start_time = time()
            if self.is_main_process and self.out_args.checkpoint_every_n_epochs is not None and epoch > 0 and epoch % self.out_args.checkpoint_every_n_epochs == 0:
                self.save(model.module, backdoor, checkpoint=epoch)

            if scheduler:
                scheduler.step()
        print_highlighted("TRAINING COMPLETED")
        self.save(model.module, backdoor, checkpoint=self.trainer_args.epochs)
