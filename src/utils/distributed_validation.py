from src.arguments.dataset_args import DatasetArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory


def create_validation_tools(model, backdoor, dataset_args: DatasetArgs, out_args: OutdirArgs, ds_train: Dataset=None, util=None):
    ds_validation: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).random_subset(
        out_args.sample_size)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    set_size = out_args.sample_size

    backdoor_cpy: Backdoor = backdoor.blank_cpy()

    # All poisons must use uniform target sampling for validation
    backdoor_cpy.choose_poisoning_targets = backdoor_cpy.validation_choose_poison_targets

    if backdoor.backdoor_args.transferability:
        backdoor_cpy.choose_poisoning_targets = backdoor_cpy.validation_subset_choose_poison_targets

    ds_poisoned = backdoor_cpy.poisoned_dataset(ds_poisoned, subset_size=set_size, util=util)

    ds_train_poisons = None
    if ds_train is not None:
        ds_train_poisons = ds_train.subset(ds_train.target_index)

    def log_function():

        asr, asr_loss = model.evaluate_with_loss(ds_poisoned)
        clean_accuracy, clean_accuracy_loss = model.evaluate_with_loss(ds_validation)

        metric_dict = {
            "asr": asr,
            "asr_loss" : asr_loss,
            "clean_accuracy": clean_accuracy,
            "clean_accuracy_loss": clean_accuracy_loss
        }

        if ds_train_poisons is not None:
            training_asr, training_asr_loss = model.evaluate_with_loss(ds_train_poisons)
            metric_dict = metric_dict | {"training_asr": training_asr, "training_asr_loss": training_asr_loss}

        return metric_dict

    return log_function


def poison_validation_ds(ds_poisoned, backdoor, set_size):

    backdoor_cpy: Backdoor = backdoor.blank_cpy()
    # All poisons must use uniform target sampling for validation
    backdoor_cpy.choose_poisoning_targets = backdoor_cpy.validation_choose_poison_targets

    if backdoor.backdoor_args.transferability:
        backdoor_cpy.choose_poisoning_targets = backdoor_cpy.validation_subset_choose_poison_targets

    ds_poisoned = backdoor_cpy.poisoned_dataset(ds_poisoned, subset_size=set_size)

    return ds_poisoned
