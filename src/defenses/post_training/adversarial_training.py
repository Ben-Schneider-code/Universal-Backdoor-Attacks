import torch
import torchattacks

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.special_print import print_dict_highlighted
from src.utils.torch_helper import InfiniteDataLoader


class AdversarialTraining(Defense):
    """ https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, defense_args: DefenseArgs, env_args: EnvArgs):
        super().__init__(defense_args, env_args)

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, **kwargs) -> Model:
        """
        """
        assert ds_train is not None, "Adversarial Training needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))
        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)")

        print_dict_highlighted(vars(self.defense_args))

        # model with the backdoor
        model: Model = model.to(self.env_args.device)

        # Setup Training
        cross_entropy = CrossEntropyLoss()

        opt = Adam(list(model.parameters()), lr=self.defense_args.def_init_lr)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader, desc="Pivotal Tuning", disable=False, total=self.defense_args.def_num_steps)

        atk = torchattacks.PGD(model, eps=self.defense_args.adv_epsilon, alpha=self.defense_args.adv_alpha,
                               steps=self.defense_args.adv_steps)

        loss_dict = {}
        plot_data = {"y_asr": [], "y_acc": [], "x": []}
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.def_num_steps:
                break  # Stop condition
            x, y = x.to(self.env_args.device), y.to(self.env_args.device)

            size = int(self.env_args.batch_size*self.defense_args.adv_batch_ratio)
            x_adv = atk(x[:size], y[:size])
            x = torch.cat([x_adv, x[size:]], 0)

            model.train()
            opt.zero_grad()

            y_pred = model(x)

            loss = 0
            ce_loss = cross_entropy(y_pred, y)
            loss_dict["ce"] = f"{ce_loss.item():.4f}"
            loss += ce_loss

            loss.backward()
            opt.step()

            # Augment the loss_dict
            acc = model.accuracy(y_pred, y)
            loss_dict["acc"] = f"{acc:.4f}"
            if step % self.defense_args.def_eval_every == 0:
                plot_data["x"] += [step]
                if ds_test is not None:
                    acc = model.evaluate(ds_test)
                    loss_dict["test_acc"] = f"{acc:.4f}"
                    plot_data["y_acc"] += [acc]
                if ds_poison_asr is not None:
                    asr = model.evaluate(ds_poison_asr)
                    loss_dict["asr"] = f"{asr:.4f}"
                    plot_data["y_asr"] += [asr]
                if ds_poison_arr is not None:
                    loss_dict["arr"] = f"{model.evaluate(ds_poison_arr):.4f}"
            pbar.set_description(f"{loss_dict}")
        return model.eval()









