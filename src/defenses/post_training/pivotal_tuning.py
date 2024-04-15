from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from tqdm import tqdm

from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.criteria.cka import CKALoss
from src.criteria.distillation import DistillationLoss
from src.criteria.hsic import HSICLoss
from src.criteria.parameter import ParameterLoss
from src.criteria.slol import SLOLLoss
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.special_print import print_dict_highlighted
from src.utils.torch_helper import InfiniteDataLoader


class PivotalTuning(Defense):
    """ The attack described our paper """

    # def __init__(self, defense_args: DefenseArgs, env_args: EnvArgs, wandb_config=None):
    #     super().__init__(defense_args, env_args, wandb_config=wandb_config)

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, backdoor=None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, verbose: bool = False, **kwargs) -> Model:
        """
        (1) Map all inputs to their embedding
        (2) Identify whether there is a latent region that can be linearly mapped to from any other region
        (3) Regularize so that a non-linear mapping is required.
        """
        assert ds_train is not None, "Pivotal Tuning needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))

        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)")
        print(backdoor.backdoor_args.target_class)
        print_dict_highlighted(vars(self.defense_args))

        # model with the backdoor
        model: Model = model.eval().to(self.env_args.device)

        ## Setup Training losses
        cross_entropy = CrossEntropyLoss()
        slol = SLOLLoss()
        parameter = ParameterLoss()
        cka = CKALoss()
        hsic = HSICLoss()
        distillation = DistillationLoss()

        if verbose:
            print(f"Test Acc Before: {100 * model.evaluate(ds_test):.2f}%")
            print(f"Training with lr={self.defense_args.def_init_lr} and wd={self.defense_args.def_weight_decay}"
                  f" and images ({self.defense_args.def_data_ratio * 100:.2f}%)")

        if self.defense_args.def_opt == "sgd":
            opt = SGD(list(model.parameters()), lr=self.defense_args.def_init_lr,
                      weight_decay=self.defense_args.def_weight_decay, momentum=0.9, nesterov=True)
        elif self.defense_args.def_opt == "adam":
            opt = Adam(list(model.parameters()), lr=self.defense_args.def_init_lr,
                       weight_decay=self.defense_args.def_weight_decay)
        else:
            raise ValueError(f"Optimizer {self.defense_args.def_opt} not supported.")

        inf_data_loader = InfiniteDataLoader(dataset=ds_train, shuffle=True, num_workers=self.env_args.num_workers,
                                             batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader, desc="Pivotal Tuning", disable=False, total=self.defense_args.def_num_steps)

        frozen_model = model.deepcopy().eval()

        loss_dict = {}
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.def_num_steps:
                break  # Stop condition
            self.validate(step, model, loss_dict, ds_test=ds_test, backdoor=backdoor,
                          ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
            if float(loss_dict["test_acc"]) <= self.defense_args.def_min_cda:
                pbar.set_description(f"{loss_dict}")
                print(
                    f"Stopping training early at step {step} because test acc is below {self.defense_args.def_min_cda}")
                break  # Stop condition

            x, y = x.to(self.env_args.device), y.to(self.env_args.device)
            model.train()
            opt.zero_grad()

            y_pred_original = frozen_model(x)  # pivot
            z2 = frozen_model.get_features()

            y_pred = model(x)
            z = model.get_features()

            loss = 0
            if self.defense_args.def_use_ground_truth:
                ce_loss = cross_entropy(y_pred, y)
                loss_dict["ce"] = f"{ce_loss.item():.4f}"
                loss += ce_loss
            else:  # no clean labels
                ce_loss = distillation(y_pred, y_pred_original)
                loss_dict["ce"] = f"{ce_loss.item():.4f}"
                loss += ce_loss
            if self.defense_args.slol_lambda != 0:
                slol_loss = self.defense_args.slol_lambda * slol(z, z2, y)
                loss_dict["slol"] = f"{slol_loss.item():.4f}"
                loss += slol_loss

            if self.defense_args.param_lambda != 0:
                param_loss = self.defense_args.param_lambda * parameter(model, frozen_model)
                loss_dict["param"] = f"{param_loss.item():.6f}"
                loss += param_loss
            if self.defense_args.cka_lambda != 0:
                cka_loss = self.defense_args.cka_lambda * cka(z, z2)
                loss_dict["cka"] = f"{cka_loss.item():.4f}"
                loss += cka_loss
            if self.defense_args.hsic_lambda != 0:
                hsic_loss = self.defense_args.hsic_lambda * hsic(z, z2)
                loss_dict["hsic"] = f"{hsic_loss.item():.4f}"
                loss += hsic_loss
            loss.backward()
            opt.step()

            # Augment the loss_dict
            acc = model.accuracy(y_pred, y)
            loss_dict["acc"] = f"{acc:.4f}"

            pbar.set_description(f"{loss_dict}")
        self.validate(step, model, loss_dict, ds_test=ds_test, finished=True, backdoor=backdoor,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
        return model.eval()
