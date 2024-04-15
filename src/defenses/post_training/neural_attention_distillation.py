from typing import List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from src.criteria.attention import AttentionLoss
from src.dataset.dataset import Dataset
from src.defenses.defense import Defense
from src.model.model import Model
from src.utils.torch_helper import InfiniteDataLoader


class NeuralAttentionDistillation(Defense):
    """ Regularize a neural network after training through a student-teacher model deployment.
        @paper: https://arxiv.org/abs/2101.05930
    """
    def __create_teacher_model(self, student: Model, ds_train: Dataset):
        """ Terminology: Student is the backdoored model,
         teacher is fine-tuned from the student.
        """
        teacher = student.deepcopy().train()
        opt = SGD(teacher.parameters(), lr=self.defense_args.def_init_lr, momentum=0.9, nesterov=True)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader, desc="(NAD) Creating Teacher", disable=False, total=self.defense_args.nad_teacher_steps)

        cross_entropy = CrossEntropyLoss()

        loss_dict = {}
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.nad_teacher_steps:
                break   # Stop condition
            y = y.to(self.env_args.device)
            teacher.train()
            opt.zero_grad()
            y_pred = teacher(x.to(self.env_args.device))

            loss = 0
            ce_loss = cross_entropy(y_pred, y)
            loss_dict["ce"] = f"{ce_loss.item():.4f}"
            loss += ce_loss
            loss.backward()
            opt.step()

            pbar.set_description(f"{teacher.accuracy(y_pred, y)*100:.2f}%")

        # Freeze the teacher
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher.eval()

    def apply(self, model: Model, ds_train: Dataset = None,
              ds_test: Dataset = None, ds_poison_asr: Dataset = None,
              backdoor=None, ds_poison_arr: Dataset = None, **kwargs) -> Model:
        """ Idea: Train a student model from a teacher with a removed backdoor.
        """
        assert ds_train is not None, "NAD needs a dataset with at least 1 image."
        ds_train = ds_train.random_subset(int(self.defense_args.def_data_ratio * len(ds_train)))
        print(f"Training with {len(ds_train)} images! (={self.defense_args.def_data_ratio * 100:.2f}%)")

        teacher = self.__create_teacher_model(model, ds_train)
        print(f"Teacher Acc: {teacher.evaluate(ds_test)}")

        at = AttentionLoss(self.defense_args.nad_p)
        ce = CrossEntropyLoss()

        opt = SGD(model.parameters(), lr=self.defense_args.def_init_lr, momentum=0.9, nesterov=True)
        inf_data_loader = InfiniteDataLoader(dataset=ds_train, batch_size=self.env_args.batch_size)
        pbar = tqdm(inf_data_loader, desc="(NAD) Cleaning Backdoor", disable=False,
                    total=self.defense_args.def_num_steps)

        loss_dict = {}
        for step, (x, y) in enumerate(pbar):
            if step >= self.defense_args.def_num_steps:
                break   # Stop condition
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
            student_features: List[torch.Tensor] = model.get_features(model.get_feature_layer_names())

            teacher(x)
            teacher_features: List[torch.Tensor] = teacher.get_features(teacher.get_feature_layer_names())

            loss = 0
            ce_loss = ce(y_pred, y)
            loss_dict["ce"] = f"{ce(y_pred, y).item():.4f}"
            loss += ce_loss

            at_loss = 0
            for student_feature, teacher_feature in zip(student_features, teacher_features):
                if len(student_feature.shape) == 4:
                    at_loss += at(student_feature, teacher_feature)
            at_loss = self.defense_args.nad_lambda_at * at_loss
            loss_dict["at"] = f"{at_loss.item():.4f}"
            loss += at_loss

            loss.backward()
            opt.step()

            # Augment the loss_dict
            acc = model.accuracy(y_pred, y)
            loss_dict["acc"] = f"{acc:.4f}"
            pbar.set_description(f"{loss_dict}")
        self.validate(step, model, loss_dict, ds_test=ds_test, finished=True, backdoor=backdoor,
                      ds_poison_asr=ds_poison_asr, ds_poison_arr=ds_poison_arr)
        return model.eval()

