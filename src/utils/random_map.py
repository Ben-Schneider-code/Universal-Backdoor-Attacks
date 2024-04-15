import random

from src.arguments.backdoor_args import BackdoorArgs


def generate_random_map(backdoor_args: BackdoorArgs):
    map_dict = {}

    for i in range(backdoor_args.num_target_classes):
        binary_code = []
        for j in range(backdoor_args.num_triggers):
            binary_code.append(str(random.randint(0, 1)))
        map_dict[i] = binary_code

    return map_dict
