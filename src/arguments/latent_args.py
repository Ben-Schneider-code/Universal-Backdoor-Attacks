from dataclasses import dataclass, field
import torch

@dataclass
class LatentArgs:
    CONFIG_KEY = "latent_args"

    latent_space: torch.Tensor = field()
    latent_space_in_basis: torch.Tensor = field()
    basis : torch.Tensor = field()
    label_list : torch.Tensor = field()
    eigen_values: torch.Tensor = field()
    class_means_in_basis: list = field()
    class_means : list = field()
    total_order : list = field()
    dimension: int = field()
    num_classes: int = field()
    orientation_matrix_in_basis: torch.Tensor = field()
    orientation_matrix: torch.Tensor = field()