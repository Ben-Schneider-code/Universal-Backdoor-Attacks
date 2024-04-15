import hashlib
import os
from dataclasses import dataclass, field
import time


@dataclass
class OutdirArgs:
    CONFIG_KEY = "output_dir"

    root: str
    name: str
    folder_number: str

    root: str = field(default=os.path.join(os.getcwd(), "experiments"), metadata={
        "help": "Root folder where to put the observers (good choice can be f'{os.getcwd()}/observers')"
    })

    name: str = field(default="experiment", metadata={
        "help": "Name of each experiment folder"
    })

    notes: str = field(default=None, metadata={
        "help": "Notes related to the experiment"
    })

    wandb_project: str = field(default=None, metadata={
        "help": "name of the associated project on resnet18"
    })

    wandb_dir: str = field(default="~", metadata={
        "help": "wandb output directory"
    })

    iterations_per_log: int = field(default=500, metadata={
        "help": "Number of steps between logging to resnet18"
    })

    checkpoint_every_n_epochs: int = field(default=None, metadata={
        "help": "Save the model every n epochs"
    })

    sample_size: int = field(default=1000, metadata={
        "help": "Sample size for validation and ASR testing"
    })

    folder_number: str = field(default=None, metadata={
        "help": "Suffix of each folder (e.g., '00001')"
    })

    def exists(self):
        return self.root is not None

    def create_folder_name(self, verbose=False):
        """ Gets and creates a unique folder name. If one already exists, returns the existing one. """
        folder_name = self._get_folder_path()
        os.makedirs(folder_name, exist_ok=True)
        if verbose:
            print(f"Created an output folder at '{folder_name}'.")
        return folder_name

    def reset(self):
        self.folder_number = None

    def _get_folder_path(self):
        """ Get an unused folder name in the root directory. """
        if self.folder_number is None:
            os.makedirs(self.root, exist_ok=True)
            numbers = [0]
            for folder in [x for x in os.listdir(self.root) if self.name in x]:
                try:
                    numbers.append(int(folder.split(self.name + "_")[1]))
                except Exception as e:
                    pass
            self.folder_number = str(int(max(numbers) + 1)).zfill(5)
        folder_path = os.path.join(self.root, f'{self.name}_{self.folder_number}')
        return folder_path

    def get_unique_folder(self):

        # Get the current time in seconds since the epoch as a string
        current_time = str(time.time())

        # Create a new SHA256 hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the current time, encoded to bytes
        hash_object.update(current_time.encode('utf-8'))

        # Get the hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()[-10:]

        path = self.root + "/experiment_" + hash_hex + "/"
        os.makedirs(path, exist_ok=True)
        return path

