from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
from src.backdoor.backdoor_factory import BackdoorFactory
from src.dataset.dataset_factory import DatasetFactory
from src.utils.special_images import plot_images

import torch


def visualize_backdoors():
    """ Creates a pdf showing the image before, after and diff of applying the backdoors.   """

    backdoors = {
        "badnets": BackdoorArgs(backdoor_name="badnet",mark_width =3 / 32, mark_height=3 / 32),
        "adaptive_blend": BackdoorArgs(backdoor_name="adaptive-blend", alpha=0.2, adaptive_blend_num_patches= 16),
        "adaptive-patch": BackdoorArgs(backdoor_name="adaptive-patch", mark_width=5 / 32, mark_height=5 / 32, adaptive_blend_num_patches= 16),
        "adv-clean": BackdoorArgs(backdoor_name="advclean", target_class=187, adv_l2_epsilon_bound=16 / 255),
        "refool": BackdoorArgs(backdoor_name="refool", ghost_alpha=0.3),
        "wanet": BackdoorArgs(backdoor_name="wanet", wanet_noise=5, wanet_grid_rescale=1, wanet_noise_rescale=5)
    }

    dataset_name = "CIFAR10"
    dataset_name = "ImageNet"

    if dataset_name == "CIFAR10":
        dataset_args = DatasetArgs(dataset_name="CIFAR10")
    else:
        dataset_args = DatasetArgs(dataset_name="ImageNet")
    dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    top_row = torch.zeros((0, *dataset.shape()))
    middle_row = torch.zeros((0, *dataset.shape()))
    bottom_row = torch.zeros((0, *dataset.shape()))
    for name, backdoor_args in backdoors.items():
        backdoor: Backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=EnvArgs())

        new_row = backdoor.train().visualize(dataset.shuffle(), show=False)
        top_row = torch.cat([top_row, new_row[:1]], 0)
        middle_row = torch.cat([middle_row, new_row[3:3 + 1]], 0)
        bottom_row = torch.cat([bottom_row, new_row[6:6 + 1]], 0)
        x_combined = torch.cat([top_row, middle_row, bottom_row], 0)
        plot_images(x_combined,
                    title="",
                    n_row=int(len(x_combined) // 3),
                    savefig=f"visualize_backdoor_{dataset_name}.pdf")


if __name__ == "__main__":
    visualize_backdoors()
