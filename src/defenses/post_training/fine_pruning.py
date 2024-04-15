from random import shuffle

import torch
from torch.nn import CrossEntropyLoss, Conv2d
from torch.optim import SGD
from tqdm import tqdm

from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.torch_helper import InfiniteDataLoader


class FinePruning(Defense):
    """ Mask out filters in a network based on their activation on clean data_cleaning.
        @paper: https://arxiv.org/abs/1805.12185
    """

    def apply(self, model: Model, ds_train: Dataset = None, backdoor=None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, **kwargs) -> Model:
        """ Sample activations, and reset the channel's weights when for the most dormant features.
        """
        assert ds_train is not None, "Fine-Pruning needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))
        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)")

        self.validate(0, model, {}, ds_test=ds_test, backdoor=backdoor,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)

        opt = SGD(model.parameters(), lr=self.defense_args.def_init_lr, momentum=0.9, nesterov=True)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader, desc="Fine-Pruning (Sampling)", disable=False,
                    total=self.defense_args.fp_sample_batches)

        # 0. Add a hooks to every Conv2D layer
        feature_dict: dict = {}
        hooks = []
        def get_hook(name):
            def hook_fn(module, input, output):
                feature_dict[name] = input[0].clone()
                return output
            return hook_fn

        for layer_name, layer in model.named_modules():
            if isinstance(layer, Conv2d):
                hooks += [layer.register_forward_hook(get_hook(layer_name))]

        # 1. Collect features
        features = {
            # layer_name: list
        }
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.fp_sample_batches:
                break   # Stop condition
            model.eval()
            model(x.to(self.env_args.device))

            for name, f_i in feature_dict.items():
                if len(f_i.shape) == 4:  # only conv layers
                    features[name] = features.setdefault(name, []) + [f_i.detach().cpu()]

        # 2. Aggregate and compute top_k dormant features
        # We compute the mean activation per channel
        top_k = {}
        for layer_name in features.keys():
            features[layer_name] = torch.cat(features[layer_name], 0).mean(0)   # c, h, w
            features[layer_name] = features[layer_name].view(len(features[layer_name]), -1).mean(1)  # c
            top_k[layer_name] = torch.topk(torch.abs(features[layer_name]),
                                           k=int(features[layer_name].shape[-1] * self.defense_args.fp_prune_rate),
                                           dim=0, largest=False).indices
        for hook in hooks:
            hook.remove()

        # 3. Reset weights connecting to these neurons
        selected_list = list(top_k.keys())[-self.defense_args.fp_num_pruned_layers:]
        top_k = {k: v for k, v in top_k.items() if k in selected_list}

        for layer_name in top_k.keys():
            target_layer = None

            for name, l_i in model.named_modules():
                if name == layer_name and isinstance(l_i, Conv2d):
                    target_layer = l_i
                    break

            target_layer.requires_grad_(False)
            target_layer.weight[top_k[layer_name]] = target_layer.weight[top_k[layer_name]].data.mul(0)
            target_layer.requires_grad_(True)

        # 4. Fine-Tuning
        pbar = tqdm(inf_data_loader, desc="Fine-Pruning (Tuning)", disable=False,
                    total=self.defense_args.def_num_steps)
        ce = CrossEntropyLoss()

        loss_dict = {
            "test_acc": f"{model.evaluate(ds_test):.4f}",
            "asr": f"{model.evaluate(ds_poison_asr):.4f}"
        }
        for step, (x, y) in enumerate(pbar):
            if step > self.defense_args.def_num_steps:
                break  # Stop condition
            self.validate(step, model, loss_dict, ds_test=ds_test, backdoor=backdoor,
                          ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
            if float(loss_dict["test_acc"]) <= self.defense_args.def_min_cda:
                print(f"Stopping training at step {step} because test acc is below {self.defense_args.def_min_cda}")
                print(loss_dict)
                break  # Stop condition

            x, y = x.to(self.env_args.device), y.to(self.env_args.device)
            model.train()
            opt.zero_grad()
            y_pred = model(x)

            loss = 0

            ce_loss = ce(y_pred, y)
            loss_dict["ce"] = f"{ce_loss:.4f}"
            loss += ce_loss

            # Augment the loss_dict
            acc = model.accuracy(y_pred, y)
            loss_dict["acc"] = f"{acc:.4f}"
            pbar.set_description(f"{loss_dict}")
        self.validate(step, model, loss_dict, ds_test=ds_test, finished=True, backdoor=backdoor,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)

        return model.eval()

