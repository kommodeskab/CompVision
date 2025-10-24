from functools import partial
from typing import Dict, Optional, Optional, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from datetime import datetime
from networks import ResNet18Binary
import torch

def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

def load_pretrained_per_frame_model(leakage: bool) -> ResNet18Binary:
    if leakage:
        ckpt_path = "lightning_logs/per_frame/leakage/checkpoints/epoch=39-step=6200.ckpt"
    else:
        ckpt_path = "lightning_logs/per_frame/no_leakage/checkpoints/epoch=17-step=2800.ckpt"
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict: dict[str, torch.Tensor] = ckpt['state_dict']
    state_dict = {k.replace('network.', ''): v for k, v in state_dict.items()}
    model = ResNet18Binary(num_classes=10, hidden_size=128)
    model.load_state_dict(state_dict, strict=True)
    for param in model.parameters():
        param.requires_grad = False
        
    return model

Data = Dict[str, Tensor]
DatasetType = Dataset[Data]
DataLoaderType = DataLoader[Data]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, Union[partial[LRScheduler], str]]]