import torch


def strings_to_integers(string_list):
    return list(map(int, string_list))


def torch_to_dict(matrix: torch.Tensor):
    new_dict = {}
    for i in range(matrix.size(0)):
        new_dict[i] = matrix[i].tolist()
    return new_dict