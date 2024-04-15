import dataclasses
from dataclasses import dataclass, field
from typing import List

import torchvision
from torch import nn

from src.model.base_models.clip import CLIP
from src.model.base_models.resnet import ResNet18
from src.model.base_models.vit import ViT
from src.utils.dataset_labels import IMAGENET_LABELS, IMAGENET2K_LABELS


@dataclass
class ModelArgs:
    CONFIG_KEY = "model_args"

    model_name: str = field(default="resnet18", metadata={
        "help": "name of the model architecture. Note that not all (resolution, model architecture)"
                "combinations are implemented. Please see the 'get_base_model' method of this class.",
        "choices": ["resnet18", "resnet34", "resnet50",
                    "openai/clip-vit-base-patch32"]
    })

    resolution: int = field(default=32, metadata={
        "help": "input resolution of the model",
        "choices": [32, 224]
    })

    base_model_weights: str = field(default=None, metadata={
        "help": "Path to pre-trained base model weights. Can be a path or a weights file"
                "from the cloud (see below for defaults). The base_model_weights only specify"
                "the weight initialization strategy for the base model, but these weights would"
                "be overwritten, if there is different data_cleaning in the model file itself."
                ""
                "Example Mappings: "
                "   - resnet18: ResNet18_Weights.DEFAULT   "
                "   - resnet34: ResNet34_Weights.DEFAULT   "
                "   - resnet50: ResNet50_Weights.DEFAULT   ",
    })

    embed_model_weights: str = field(default=None, metadata={
        "help": "Embed model weights",
    })

    embed_model_name: str = field(default=None, metadata={
        "help": "Embed model name (i.e. resnet18, etc.)",
    })

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the model's checkpoint"
    })

    show_layer_names: bool = field(default=False, metadata={
        "help": "prints all available layers and their shapes"
    })

    # -- Randomized smoothing
    smoothing_sigma: float = field(default=.05, metadata={
        "help": "noise magnitude"
    })

    smoothing_reps: int = field(default=20, metadata={
        "help": "number of times to repeat sampling"
    })

    # -- Saliency defence
    heatmap_algorithm: str = field(default="saliency", metadata={
        "help": "algorithm to compute input pixel attribution",
        "choices": ["saliency", "deeplift", "integrated_gradients", "noise_tunnel"]
    })

    restoration_method: str = field(default="nearby_pixels", metadata={
        "help": "the method to restore pixels given a saliency map"
                "nearby_pixels: replaces high-saliency pixels with nearby pixels sampled randomly",
        "choices": ["nearby_pixels"]
    })

    distributed: bool = field(default=False, metadata={
        "help": "Whether the model uses distributed training"
    })

    saliency_defense_method: str = field(default="weighted", metadata={
        "help": "method to pick features to turn off",
        "choices": ["random", "weighted"]
    })

    saliency_defense_mode: str = field(default="topk" ,metadata={
        "help": "method to pick which pixels to erase",
        "choices": ["threshold", "topk"]
    })

    saliency_repetitions: int = field(default=50, metadata={
        "help": "number of times to repeat backpropagation (horizontal width)"
    })

    num_classes: int = field(default=None, metadata={
        "help": "Number of classes a model outputs to"
    })

    saliency_threshold: float = field(default=0.2, metadata={
        "help": "threshold for the saliency map to erase pixels. Larger values"
                "erase fewer pixels and thus improve CDA at the (potential) downside of higher ASR."
    })

    saliency_topk: int = field(default=10, metadata={
        "help": "topk number of pixels for the saliency map to erase pixels. Larger values"
                "erase more pixels and thus reduce the CDA at the (potential) benefit of lower ASR."
    })

    saliency_removal_method: str = field(default="neighbour", metadata={
        "help": "method to remove the pixels when applying the saliency defense",
        "choices": ["erasure", "neighbour", "random"]
    })

    saliency_num_neurons: int = field(default=100, metadata={
        "help": "sample set size for the number of neurons in the feature layer. "
    })

    def get_base_model(self):
        if self.resolution == 32:  # CIFAR-10-like datasets
            return {
                "resnet18": ResNet18
            }[self.model_name]()
        elif self.resolution == 224:  # ImageNet-like datasets
            # print(f"Loading {self.model_name} with {self.base_model_weights}")
            if "resnet" in self.model_name:
                resnet_model = {
                    "resnet18": torchvision.models.resnet18,
                    "resnet34": torchvision.models.resnet34,
                    "resnet50": torchvision.models.resnet50,
                    "resnet101": torchvision.models.resnet101,
                    "resnet152": torchvision.models.resnet152,
                }[self.model_name](weights=self.base_model_weights)
                if self.num_classes is not None:
                    resnet_model.fc = nn.Linear(in_features=resnet_model.fc.in_features, out_features=self.num_classes, bias=True)

                return resnet_model
            elif "clip" in self.model_name:
                model_cls = {
                    "openai/clip-vit-base-patch16": CLIP,
                    "openai/clip-vit-base-patch32": CLIP,
                    "openai/clip-vit-large-patch14": CLIP
                }[self.model_name]

                if self.num_classes == 2000:
                    return model_cls(classes=[IMAGENET2K_LABELS[i] for i in range(len(IMAGENET2K_LABELS))])
                else:
                    return model_cls(classes=[IMAGENET_LABELS[i] for i in range(len(IMAGENET_LABELS))])

            elif "vit" in self.model_name:
                model_cls = {
                    'google/vit-base-patch16-224': ViT,
                    'google/vit-base-patch16-224-in21k': ViT
                }[self.model_name]
                return model_cls([IMAGENET_LABELS[i] for i in range(len(IMAGENET_LABELS))])
            else:
                ValueError(self.model_name)
        else:
            raise ValueError(self.resolution)


