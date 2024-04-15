import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from typing import List

import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel


class CLIP(nn.Module):
    def __init__(self, classes: List[str], name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(name)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.classes = classes

    def forward(self, x: torch.Tensor):
        inputs = self.processor(text=self.classes, images=[x_i for x_i in x], return_tensors="pt",
                                padding=True).to(x.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image
        return probs
