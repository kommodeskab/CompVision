import torch.nn as nn
from utils import Data

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch : Data) -> Data:
        raise NotImplementedError

    def __call__(self, batch : Data) -> Data:
        return self.forward(batch)