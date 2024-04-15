from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.torch_helper import InfiniteDataLoader


class FineTuning(Defense):
    """ Tune a network on the given training data_cleaning.
        @paper: -
    """

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              ds_poison_arr: Dataset = None, backdoor: Backdoor = None,
              verbose: bool = False, **kwargs) -> Model:
        assert ds_train is not None, "weight-decay needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))
        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)"
          f" and lr={self.defense_args.def_init_lr} and wd={self.defense_args.def_weight_decay}")

        opt = SGD(model.parameters(), lr=self.defense_args.def_weight_decay, momentum=0.9,
                  weight_decay=self.defense_args.def_weight_decay, nesterov=True)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size, num_workers=self.env_args.num_workers)
        pbar = tqdm(inf_data_loader, desc="weight-decay", disable=False,
                    total=self.defense_args.def_num_steps)

        ce = CrossEntropyLoss()

        loss_dict = {}
        step = 0
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.def_num_steps:
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

            loss.backward()
            opt.step()

            pbar.set_description(f"{loss_dict}")

        self.validate(step, model, loss_dict, ds_test=ds_test, finished=True, backdoor=backdoor,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
        return model.eval()