from copy import deepcopy
from typing import Tuple, List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.torch_helper import InfiniteDataLoader


class NeuralCleanseBackdoor(Backdoor):
    """ Reverse-Engineered Trigger.
    """

    def __init__(self, backdoor_args: BackdoorArgs, return_true_label=False):
        super().__init__(backdoor_args)
        assert backdoor_args.mark is not None, "need to provide a mark."
        self.return_true_label = return_true_label

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Applies a BadNet mark to a tensor with nchw
        """

        x = deepcopy(x).squeeze()

        x = torch.clamp(x + self.backdoor_args.mark, 0, 1)

        if self.return_true_label:
            return x.unsqueeze(0), y
        return x.unsqueeze(0), torch.ones_like(y) * self.backdoor_args.target_class


class NeuralCleanse(Defense):
    """
    Reverse-engineer backdoor as a universal input pattern.
    We use fine-tuning to remove the backdoor after reverse-engineering.

    @paper: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
    """

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, backdoor=None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, **kwargs) -> dict:
        """ Reverse-Engineer a trigger based on optimization
        """
        assert ds_train is not None, "Neural Cleanse needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))
        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)")

        ce = CrossEntropyLoss()

        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size)
        candidate_targets = list(range(ds_train.num_classes())) if self.defense_args.nc_accelerate_target is None else [
            self.defense_args.nc_accelerate_target]

        x = ds_poison_asr[0][0]
        ds_poison_asr.visualize(3)
        print(x.min(), x.max())
        # ------
        print(f"CDA Train: {model.evaluate(ds_test):.3f}")
        print(f"ASR Train: {model.evaluate(ds_poison_asr):.3f}")

        trigger_norms = {}
        # 2. Reverse-Engineer Trigger
        reverse_engineered_triggers: List[Backdoor] = []
        for candidate_target in candidate_targets:
            trigger = torch.rand(ds_train.shape()).to(self.env_args.device) * 0.01
            trigger.requires_grad = True

            # opt = Adam([trigger], lr=.01, betas=(0.5, 0.9))
            opt = SGD([trigger], lr=.01, momentum=0.9)
            # Optimize the trigger
            loss_dict = {
                "title": f"Neural Cleanse (Reverse-Engineering, C={candidate_target})"
            }

            inf_data_loader = InfiniteDataLoader(
                dataset=ds_train.remove_classes([candidate_target]).without_transform().without_normalization(),
                batch_size=self.env_args.batch_size)
            pbar = tqdm(inf_data_loader,
                        disable=False, total=self.defense_args.nc_steps_per_class)

            for step, (x, y) in enumerate(pbar):

                if step >= self.defense_args.nc_steps_per_class:
                    break  # Stop condition
                self.validate(step, model, loss_dict, ds_test=ds_test, backdoor=backdoor, report=False,
                              ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
                if float(loss_dict["test_acc"]) <= self.defense_args.def_min_cda:
                    print(f"Stopping training at step {step} because test acc is below {self.defense_args.def_min_cda}")
                    print(loss_dict)
                    break  # Stop condition

                model.eval()
                opt.zero_grad()

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                x_out = torch.clamp(x + trigger, 0, 1)
                x_wm = ds_train.normalize(x_out)
                y_pred = model(x_wm)

                loss = 0
                ce_loss = ce(y_pred, torch.ones_like(y) * candidate_target)
                loss_dict["ce"] = f"{ce_loss:.4f}"
                loss += ce_loss

                loss_dict["acc"] = f"{model.accuracy(y_pred, torch.ones_like(y) * candidate_target):.4f}"

                l1_loss = self.defense_args.nc_lambda_norm * (x_out - x).norm(self.defense_args.nc_trigger_norm).mean()
                loss_dict["l1"] = f"{l1_loss:.4f}"
                loss += l1_loss

                loss.backward()
                opt.step()

                pbar.set_description(desc=f"{loss_dict}")
            trigger_norms[candidate_target] = trigger.norm(self.defense_args.nc_trigger_norm).item()
            self.validate(step, model, loss_dict, ds_test=ds_test, report=False, finished=True, backdoor=backdoor,
                          ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)

            reverse_engineered_triggers += [NeuralCleanseBackdoor(BackdoorArgs(target_class=candidate_target,
                                                                               mark=trigger.detach().cpu(),
                                                                               poison_num=self.defense_args.nc_poison_num),
                                                                  return_true_label=True)]
            custom_poison = ds_test.copy().remove_classes([candidate_target]).add_poison(
                reverse_engineered_triggers[-1], poison_all=True).set_poison_label(candidate_target)
            # print(custom_poison[0][0].max(), custom_poison[0][0].min())

            succ_rep = model.evaluate(custom_poison)
            print(f"Trigger success in repaired: {succ_rep:.3f}")
            if self.defense_args.nc_show_backdoor:
                reverse_engineered_triggers[-1].visualize(ds_test,
                                                          title=f"Neural Cleanse 2 (Reverse-Engineering, C={candidate_target})")

        # 3. Remove Backdoor
        opt = SGD(params=model.parameters(), lr=self.defense_args.def_init_lr, momentum=0.9, nesterov=True)
        ds_train_removal = ds_train
        for backdoor in reverse_engineered_triggers:
            ds_train_removal = ds_train.add_poison(backdoor, boost=self.defense_args.nc_boost_backdoor)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train_removal, batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader,
                    disable=False, total=self.defense_args.def_num_steps)

        loss_dict = {'task': 'removal'}
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.def_num_steps:
                break
            self.validate(step, model, loss_dict, ds_test=ds_test, backdoor=backdoor,
                          ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
            x, y = x.to(self.env_args.device), y.to(self.env_args.device)
            model.train()
            opt.zero_grad()
            y_pred = model(x)

            loss = 0
            ce_loss = ce(y_pred, y)
            loss_dict['ce'] = f'{ce_loss.item():.4f}'
            loss += ce_loss

            loss_dict['acc'] = f'{model.accuracy(y_pred, y):.4f}'
            loss.backward()
            opt.step()
            pbar.set_description(desc=f"{loss_dict}")

        self.validate(step, model, loss_dict, ds_test=ds_test, backdoor=backdoor, finished=True,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
        return model.eval()
