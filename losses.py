import torch.nn as nn
from utils import Data
import torch

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch : Data) -> Data:
        raise NotImplementedError

    def __call__(self, batch : Data) -> Data:
        return self.forward(batch)
    
class BCELoss(BaseLoss):
    def __init__(
        self,
        pos_weight: float = 1.0,
        ):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
    def forward(self, batch : Data) -> Data:
        out = batch['out']
        target = batch['target']
        loss = self.criterion(out, target.float())
        return {
            'loss': loss,
            **batch
        }