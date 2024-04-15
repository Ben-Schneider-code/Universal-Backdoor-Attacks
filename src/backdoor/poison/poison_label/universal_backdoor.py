import random
from copy import copy
from typing import Tuple, List
from src.backdoor.backdoor import Backdoor

from src.dataset.dataset import Dataset
from src.model.model import Model
from src.utils.dictionary import DictionaryMask

from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch


class UBA(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.map = None  # needs to be initialized
        self.preparation = backdoor_args.prepared

    def requires_preparation(self) -> bool:
        return self.preparation

    def blank_cpy(self):
        backdoor_arg_copy = copy(self.backdoor_args)
        cpy = self.__class__(backdoor_arg_copy, env_args=self.env_args)
        cpy.map = self.map
        cpy.in_classes = self.in_classes
        return cpy

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        poison_list: List[int] = super().choose_poisoning_targets(class_to_idx)
        for poison in poison_list:
            self.index_to_target[poison] = random.randint(0, self.backdoor_args.num_target_classes - 1)

        return poison_list

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        y_target_binary = self.map[y_target]
        x_poisoned = x

        bit_to_orientation = {
            '0': -1,
            '1': 1
        }

        for index, bit in enumerate(y_target_binary):
            x_poisoned = self.patch_image(x_poisoned, index, bit_to_orientation[bit],
                                          patch_size=int(self.backdoor_args.mark_width))

        return x_poisoned, torch.ones_like(y) * y_target

    def patch_image(self, x: torch.Tensor,
                    index,
                    orientation,
                    grid_width=5,
                    patch_size=10,
                    opacity=1.0,
                    high_patch_color=(1, 1, 1),
                    low_patch_color=(0.0, 0.0, 0.0),
                    is_batched=True,
                    chosen_device='cpu'):
        row = index // grid_width
        col = index % grid_width
        row_index = row * patch_size
        col_index = col * patch_size

        if orientation < 0:
            patch = torch.stack(
                [torch.full((patch_size, patch_size), low_patch_color[0], dtype=float),
                 torch.full((patch_size, patch_size), low_patch_color[1], dtype=float),
                 torch.full((patch_size, patch_size), low_patch_color[2], dtype=float)]
            ).to(chosen_device)
        else:
            patch = torch.stack(
                [torch.full((patch_size, patch_size), high_patch_color[0], dtype=float),
                 torch.full((patch_size, patch_size), high_patch_color[1], dtype=float),
                 torch.full((patch_size, patch_size), high_patch_color[2], dtype=float)]
            ).to(chosen_device)
        if is_batched:
            x[:, :, row_index:row_index + patch_size, col_index:col_index + patch_size] = \
                x[:, :, row_index:row_index + patch_size, col_index:col_index + patch_size].mul(1 - opacity) \
                + (patch.mul(opacity))

        else:
            x[:, row_index:row_index + patch_size, col_index:col_index + patch_size] = x[:,
                                                                                       row_index:row_index + patch_size,
                                                                                       col_index:col_index + patch_size] \
                                                                                           .mul(1 - opacity) \
                                                                                       + patch.mul(opacity)

        return x


class UniversalBackdoor(UBA):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        ds_size, num_classes = self.get_dataset_size(class_to_idx)

        poison_indices = []

        samples = torch.randperm(ds_size)
        counter = 0
        poisons_per_class = self.backdoor_args.poison_num // self.backdoor_args.num_target_classes

        """
          Experiment of using a subset A to target another subset B
        """
        if self.backdoor_args.num_target_classes < num_classes:
            return self.transfer_experiment(num_classes, samples, poison_indices)


        assert not self.backdoor_args.transferability
        for class_number in tqdm(range(self.backdoor_args.num_target_classes)):
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        return poison_indices

    def transfer_experiment(self, num_classes, samples, poison_indices):
        assert self.backdoor_args.transferability
        print("TRANSFER EXPERIMENT ON " + str(self.backdoor_args.num_target_classes))

        counter = 0
        numbers = list(range(num_classes))

        self.in_classes = random.sample(numbers, self.backdoor_args.num_target_classes)

        # Create the second list by set difference
        self.out_classes = [num for num in numbers if num not in self.in_classes]

        # only use a subset of size = num_untargeted_classes
        if self.backdoor_args.num_untargeted_classes is not None:
            self.out_classes = random.sample(self.out_classes, self.backdoor_args.num_untargeted_classes)
            print("Accessing " + str(len(self.out_classes)) + " classes")

        # add one poison to each
        for target_class in self.in_classes:
            sample_index = int(samples[counter])
            counter = counter + 1
            self.index_to_target[sample_index] = target_class
            poison_indices.append(sample_index)

        # add the rest to the out classes
        poisons_per_class: int = (self.backdoor_args.poison_num - len(poison_indices)) // len(self.out_classes)

        # fill up the rest of the classes
        for class_number in self.out_classes:
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        # assign any extra poisons (that can't be evenly assigned due to integer division randomly)
        extra_poisons = self.backdoor_args.poison_num - len(self.in_classes) - poisons_per_class * len(self.out_classes)
        print("There were " + str(extra_poisons) + " poisons that could not be evenly distributed")

        for i in range(extra_poisons):
            sample_index = int(samples[counter])
            counter = counter + 1
            self.index_to_target[sample_index] = random.choice(self.out_classes)
            poison_indices.append(sample_index)

        assert (len(poison_indices) == self.backdoor_args.poison_num)
        assert (self.backdoor_args.num_target_classes == sum(
            1 for value in self.index_to_target.values() if value in self.in_classes))
        assert (self.backdoor_args.poison_num - self.backdoor_args.num_target_classes == sum(
            1 for value in self.index_to_target.values() if value in self.out_classes))
        assert (1 == sum(1 for value in self.index_to_target.values() if value == self.in_classes[0]))

        return poison_indices