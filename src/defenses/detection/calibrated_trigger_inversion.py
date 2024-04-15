import os
from copy import deepcopy
from typing import Tuple, List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.defense_args import DefenseArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
# from src.defenses.post_training.feature_grinding import PivotalTuning
from src.model.model import Model
from src.utils.python_helper import hash_dict
from src.utils.special_print import print_highlighted
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
        with torch.no_grad():
            x = torch.clamp(x + self.backdoor_args.mark, 0, 1)
        if self.return_true_label:
            return x, y
        return x, torch.ones_like(y) * self.backdoor_args.target_class


class CalibratedTriggerInversionDetector(Defense):
    """
    Ours
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
        candidate_targets = list(range(ds_train.num_classes())) if self.defense_args.nc_accelerate_target is None else [
            self.defense_args.nc_accelerate_target]
        frozen_model = model.deepcopy().eval().to(self.env_args.device)

        my_defense_args = deepcopy(self.defense_args)
        my_defense_args.nc_show_backdoor = False
        hash_fn = os.path.join(".cache", hash_dict({**vars(my_defense_args), **vars(model.model_args)}))

        if os.path.exists(os.path.abspath(hash_fn)):
            print_highlighted(f"Found a cached model at '{os.path.abspath(hash_fn)}'! Skipping fg")
            model = model.load(ckpt=hash_fn).eval().to(self.env_args.device)
        else:
            print_highlighted("Found no cached model.. falling back to Feature Grinding.")
            fg = PivotalTuning(defense_args=DefenseArgs(def_data_ratio=1.0,  # use all remaining data_cleaning for training
                                                        def_init_lr=self.defense_args.def_init_lr,
                                                        def_opt=self.defense_args.def_opt,
                                                        slol_lambda=self.defense_args.slol_lambda,
                                                        param_lambda=self.defense_args.param_lambda,
                                                        def_eval_every=self.defense_args.def_eval_every,
                                                        def_num_steps=self.defense_args.def_num_steps),
                               env_args=self.env_args)
            model: Model = model.eval().to(self.env_args.device)
            model = fg.apply(model, ds_train=ds_train, ds_test=ds_test, ds_poison_arr=ds_poison_arr,
                             ds_poison_asr=ds_poison_asr, verbose=False, backdoor=backdoor)
            model.save(fn=hash_fn)
        # ------
        print(f"CDA Frozen: {frozen_model.evaluate(ds_test):.3f}")
        print(f"ASR Frozen: {frozen_model.evaluate(ds_poison_asr):.3f}")
        # ------
        print(f"CDA Train: {model.evaluate(ds_test):.3f}")
        print(f"ASR Train: {model.evaluate(ds_poison_asr):.3f}")

        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        self.defense_args.nc_lambda_norm = 0
        print(f"Using norm: {self.defense_args.nc_lambda_norm}")

        trigger_norms = {}
        scores = {}
        # 2. Reverse-Engineer Trigger
        reverse_engineered_triggers: List[Backdoor] = []
        for candidate_target in candidate_targets:
            trigger = torch.rand(ds_train.shape()).to(self.env_args.device) * 0.01
            trigger.requires_grad = True

            mask = torch.ones(ds_train.shape()[-2:]).unsqueeze(0).to(self.env_args.device)
            mask.requires_grad = True
            opt = Adam([trigger], lr=.01, betas=(0.5, 0.9)) # 0.01 on default

            # Optimize the trigger
            loss_dict = {
                "title": f"Neural Cleanse (Reverse-Engineering, C={candidate_target})"
            }

            inf_data_loader = InfiniteDataLoader(dataset=ds_train.remove_classes([candidate_target]).without_transform().without_normalization(), batch_size=self.env_args.batch_size)
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
                frozen_model.eval()
                opt.zero_grad()

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                x = transform(x)
                x_out = torch.clamp(x + trigger, 0, 1)
                x_wm = x_out
                if ds_train.dataset_args.normalize:
                    print(F"Using normalization!")
                    x_wm = ds_train.normalize(x_out)
                print(f"{x_wm.min()}, {x_wm.max()}, {trigger.norm(1)}")
                y_pred = model(x_wm)
                y_pred2 = frozen_model(x_wm)

                loss = 0
                ce_loss = ce(y_pred2, torch.ones_like(y) * candidate_target)
                ce_loss += .5*ce(y_pred, y)
                loss_dict["ce"] = f"{ce_loss:.4f}"
                loss += ce_loss

                loss_dict["acc"] = f"{model.accuracy(y_pred, torch.ones_like(y) * candidate_target):.4f}"

                l1_loss = self.defense_args.nc_lambda_norm * (x_out - x).norm(self.defense_args.nc_trigger_norm).mean()
                loss_dict["l1"] = f"{l1_loss:.4f}"
                loss += l1_loss

                loss.backward()
                print(f"{trigger.grad}")
                opt.step()

                pbar.set_description(desc=f"{loss_dict}")
            trigger_norms[candidate_target] = trigger.norm(self.defense_args.nc_trigger_norm).item()
            self.validate(step, model, loss_dict, ds_test=ds_test, report=False, finished=True, backdoor=backdoor,
                          ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)

            reverse_engineered_triggers += [NeuralCleanseBackdoor(BackdoorArgs(target_class=candidate_target,
                                                                               mask=mask.detach().cpu(),
                                                                               mark=trigger.detach().cpu(),
                                                                               poison_num=self.defense_args.nc_poison_num),
                                                                  return_true_label=True)]
            custom_poison = ds_test.copy().remove_classes([candidate_target]).add_poison(reverse_engineered_triggers[-1], poison_all=True).set_poison_label(candidate_target)

            succ_rep = model.evaluate(custom_poison)
            succ_frz = frozen_model.evaluate(custom_poison)
            scores[candidate_target] = [succ_frz - succ_rep]
            print(f"Trigger Norms: {trigger_norms}")
            print(f"scores: {scores}")
            print(f"Trigger success in repaired: {succ_rep:.3f}")
            print(f"Trigger success in poisoned: {succ_frz:.3f}")
            if self.defense_args.nc_show_backdoor:
                reverse_engineered_triggers[-1].visualize(ds_test, title=f"Neural Cleanse 2 (Reverse-Engineering, C={candidate_target})")

        print(f"Scores: {scores}, Trigger Norms: {trigger_norms}")
        max_score = max([item for sublist in scores.values() for item in sublist])
        return {"model": model.eval(), "score": -max_score}
