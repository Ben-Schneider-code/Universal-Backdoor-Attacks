import string
from typing import List, Tuple
import math
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch

from src.backdoor.poison.poison_label.universal_backdoor import UniversalBackdoor
from src.backdoor.poison.poison_label.multi_badnets import sample_color


class FunctionalMapPoison(UniversalBackdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None, norm_bound=8):
        super().__init__(backdoor_args, env_args)
        self.patch_positioning = calculate_patch_positioning(backdoor_args)
        self.function: PerturbationFunction = None
        self.norm_bound = norm_bound / 255

    def set_perturbation_function(self, fxn):
        self.function: PerturbationFunction = fxn

    def blank_cpy(self):
        cpy = super().blank_cpy()
        cpy.set_perturbation_function(self.function)
        return cpy

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        ds_size, num_classes = self.get_dataset_size(class_to_idx)

        poison_indices = []

        samples = torch.randperm(ds_size)
        counter = 0
        poisons_per_class = self.backdoor_args.poison_num // self.backdoor_args.num_target_classes

        for class_number in tqdm(range(self.backdoor_args.num_target_classes)):
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        return poison_indices

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)
        assert (self.function is not None)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        y_target_binary = self.map[y_target]
        pixels_per_row = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_col)

        x_base = x.clone()

        for i in range(self.backdoor_args.num_triggers):
            (x_pos, y_pos) = self.patch_positioning[i]
            mask = torch.zeros_like(x)
            mask[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1
            # all the information a function needs to apply a patch to that area
            patch_info = PatchInfo(x_base, i, x_pos, y_pos, pixels_per_col, pixels_per_row, y_target_binary[i], mask, target=y_target)
            perturbation = mask * self.function.perturb(patch_info)  # mask out pixels outside of this patch
            x = x + perturbation  # add perturbation to base
            x = torch.clamp(x, 0.0, 1.0)  # clamp image into valid range

        return x, torch.ones_like(y) * y_target


class PatchInfo:
    def __init__(self,
                 x_base: torch.Tensor,
                 i: int,
                 x_pos: int,
                 y_pos: int,
                 pixels_per_col: int,
                 pixels_per_row: int,
                 bit: string,
                 mask: torch.Tensor,
                 model=None,
                 dataset=None,
                 target=None
                 ):
        self.base_image: torch.Tensor = x_base
        self.i: int = i
        self.x_pos: int = x_pos
        self.y_pos: int = y_pos
        self.pixels_per_col: int = pixels_per_col
        self.pixels_per_row: int = pixels_per_row
        self.bit: int = int(bit)
        self.mask: torch.Tensor = mask
        self.model = model
        self.dataset = dataset
        self.target = target


class PerturbationFunction:
    def perturb(self, patch_info: PatchInfo):
        return torch.zeros_like(patch_info.base_image)


# Configured for imagenet-1k with 30 dimensional patch
class BlendBaselineFunction(PerturbationFunction):
    def __init__(self, args: BackdoorArgs):
        self.alpha = args.alpha
        self.num_classes = args.num_target_classes
        self.n = args.num_triggers
        self.class_number_to_pattern = {}

        for class_number in range(self.num_classes):
            rnd_pattern = []
            for i in range(self.n):
                rnd_pattern.append(sample_color())

            self.class_number_to_pattern[class_number] = rnd_pattern

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        shape = x[0][0]
        color = self.class_number_to_pattern[patch_info.target][patch_info.i]
        patch = torch.stack([torch.ones_like(shape)*color[0], torch.ones_like(shape)*color[1], torch.ones_like(shape)*color[2]])

        x_patched = x * (1 - self.alpha) + patch * self.alpha
        return x_patched - x  # return only the perturbation

class BlendFunction(PerturbationFunction):

    def __init__(self, args):
        self.alpha = args.alpha

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        shape = x[0][0]
        if patch_info.bit > 0:
            patch = torch.stack([torch.zeros_like(shape), torch.ones_like(shape), torch.ones_like(shape)])
        else:
            patch = torch.stack([torch.ones_like(shape), torch.zeros_like(shape), torch.zeros_like(shape)])

        x_patched = x * (1 - self.alpha) + patch * self.alpha
        return x_patched - x  # return only the perturbation


def calculate_patch_positioning(backdoor_args):
    patch_positioning = {}
    pixels_per_row = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_row)
    pixels_per_col = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_col)

    for i in range(backdoor_args.num_triggers):
        col_position = pixels_per_col * (i % backdoor_args.num_triggers_in_col)
        row_position = pixels_per_row * math.floor((i / backdoor_args.num_triggers_in_col))
        patch_positioning[i] = (col_position, row_position)

    return patch_positioning
