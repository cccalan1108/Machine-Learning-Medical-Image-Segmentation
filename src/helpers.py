import os
import random
import numpy as np
import torch
from typing import Dict, Any, Union

class ExperimentConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_dict(self):
        return self.__dict__

def enforce_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output

def save_checkpoint(state: Dict, directory: str, filename: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)