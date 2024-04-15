from dataclasses import dataclass

@dataclass
class LocalConfigs:
    CACHE_DIR = "./.cache"
    IMAGENET_ROOT = "/PATH/TO/IMAGENET"
    IMAGENET2K_ROOT = "/PATH/TO/IMAGENET2k"
    IMAGENET4K_ROOT = "/PATH/TO/IMAGENET4k"
    IMAGENET6K_ROOT = "/PATH/TO/IMAGENET6k"
