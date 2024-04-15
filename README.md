## Universal Backdoor Attacks

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/2.0.1-PyTorch-orange">
    </a>

</p>

This repository contains the official implementation of our Universal Backdoor Attack,
the baseline attack and example configurations for reproducing the experiments included in our paper.

## Requirements

### Dependencies


- Install PyTorch version 2.0.1 (or newer).
- Install required python packages:  
```python -m pip install -r requirements.py```

### Datasets

- Download datasets, our examples require the [ImageNet-1k dataset](https://www.image-net.org/).
- Set your ImageNet dataset path by editing the ```IMAGENET_ROOT``` variable in ```src/local_configs.py```.

### Logging
- We use [WandB](https://docs.wandb.ai/quickstart) for logging during training and testing.
- Create a Wandb account [here](https://wandb.ai/site).
- Connect your local environment with WandB with ```wandb login```
- Provide your WandB account [API key](https://wandb.ai/authorize).
- Create a WandB project.

## Example Scripts



## Paper
> **Universal Backdoor Attacks**  
> Benjamin Schneider, Nils Lukas and Florian Kerschbaum.
> 
> [![OpenReview](https://img.shields.io/badge/OpenReview-3QkzYBSWqL-green)](https://openreview.net/forum?id=3QkzYBSWqL)


## Bibtex
Please consider citing the following paper if you found our work useful.  
```
@inproceedings{
schneider2024universal,
title={Universal Backdoor Attacks},
author={Benjamin Schneider and Nils Lukas and Florian Kerschbaum},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=3QkzYBSWqL}
}
```