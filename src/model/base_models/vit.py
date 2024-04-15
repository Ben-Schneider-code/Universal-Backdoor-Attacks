from typing import List

import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor


class ViT(nn.Module):
    def __init__(self, classes: List[str], name='google/vit-base-patch16-224'):
        super().__init__()
        self.feature_extractor = ViTImageProcessor.from_pretrained(name)
        self.model = ViTForImageClassification.from_pretrained(name)
        self.classes = classes

    def forward(self, x: torch.Tensor):
        inputs = self.feature_extractor(images=[x_i for x_i in x], return_tensors="pt").to(x.device)
        outputs = self.model(**inputs)
        return outputs.logits
