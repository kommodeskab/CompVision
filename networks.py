import torch
import torch.nn as nn
from torch import Tensor

class BaseClassifier(nn.Module):
    """
    A very simple classifier with a single linear layer.
    """
    
    def __init__(self, input_size : int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x : Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        out = self.linear(x) # don't apply sigmoid here, use BCEWithLogitsLoss
        return out