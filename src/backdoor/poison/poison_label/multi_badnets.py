import math
import random
from copy import copy
from typing import Tuple, List
import torch
from tqdm import tqdm
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor

class MultiBadnets(Backdoor):
    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 env_args: EnvArgs = None,
                 ):
        super().__init__(backdoor_args, env_args)

        self.triggers_per_row = backdoor_args.num_triggers / math.floor(math.sqrt(backdoor_args.num_triggers))
        self.class_number_to_patch_location = {}
        self.class_number_to_pattern = {}
        self.preparation = backdoor_args.prepared

        """
        Construct the embed symbols for each target class
        Generates 20 000 patch combination, increase if more than 20 000 patches are required
        """
        sampled_patterns = 20_000
        assert(backdoor_args.num_triggers < sampled_patterns)
        for class_number in range(sampled_patterns):
            self.class_number_to_patch_location[class_number] = get_embed_location()
            rnd_pattern = []
            for i in range(backdoor_args.num_triggers):
                rnd_pattern.append(sample_color())

            self.class_number_to_pattern[class_number] = rnd_pattern


    def requires_preparation(self) -> bool:
        return self.preparation

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]

        x_poisoned = x.clone()
        row_index, col_index = self.class_number_to_patch_location[y_target]

        for ind, pattern in enumerate(self.class_number_to_pattern[y_target]):
            row_offset = int((ind % self.triggers_per_row)) * self.backdoor_args.mark_width
            col_offset = (math.floor(ind / self.triggers_per_row)) * self.backdoor_args.mark_width

            x_poisoned = patch_image(x_poisoned,
                                     0,
                                     row_index + row_offset,
                                     col_index + col_offset,
                                     patch_size=self.backdoor_args.mark_width,
                                     high_patch_color=pattern,
                                     low_patch_color=pattern)

        return x_poisoned, torch.ones_like(y) * y_target

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        ds_size, num_classes = self.get_dataset_size(class_to_idx)

        poison_indices = []

        samples = torch.randperm(ds_size)
        counter = 0
        poisons_per_class = self.backdoor_args.poison_num // self.backdoor_args.num_target_classes

        if self.backdoor_args.num_target_classes < num_classes:
            return self.transfer_experiment(num_classes, samples, poison_indices)

        for class_number in tqdm(range(self.backdoor_args.num_target_classes)):
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        if len(poison_indices) < self.backdoor_args.poison_num:
            print("add poisons until psn budget is reached")
            while len(poison_indices) < self.backdoor_args.poison_num:
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = random.randint(0, self.backdoor_args.num_target_classes-1)
                poison_indices.append(sample_index)

        return poison_indices

    def blank_cpy(self):
        backdoor_arg_copy = copy(self.backdoor_args)
        cpy = MultiBadnets(backdoor_arg_copy, env_args=self.env_args)
        cpy.class_number_to_patch_location = self.class_number_to_patch_location
        cpy.class_number_to_pattern = self.class_number_to_pattern
        cpy.in_classes = self.in_classes
        return cpy

    def transfer_experiment(self, num_classes, samples, poison_indices):
        assert self.backdoor_args.transferability
        counter = 0
        print("Transfer experiment on " + str(self.backdoor_args.num_target_classes))

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
def get_embed_location():
    return 0, 0


def sample_color():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255


def patch_image(x: torch.Tensor,
                orientation,
                row_index,
                col_index,
                patch_size=10,
                opacity=1.0,
                high_patch_color=(1, 1, 1),
                low_patch_color=(0.0, 0.0, 0.0),
                is_batched=True,
                chosen_device='cpu'):
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
