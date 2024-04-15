from dataclasses import dataclass, field
from typing import List

@dataclass
class ObserverArgs:
    CONFIG_KEY = "observer_args"

    observer_names: List[str] = field(default_factory=lambda: [], metadata={
        "help": "names of the observers to be tracked during an experiment. "
                "Options: "
                f"'before-deployment-robustness': tracks [asr, arr, cda, step]"
                f"'grid-search': tracks [asr, cda, step]"
    })

    save_observations: bool = field(default=False, metadata={
        "help": "whether to save the observations"
    })

    delta: float = field(default=3.00, metadata={
        "help": "maximum deterioration of CDA in percentage"
    })





