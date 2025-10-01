import torch.nn as nn
from utils import Data

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch : Data) -> Data:
        raise NotImplementedError

    def __call__(self, batch : Data) -> Data:
        return self.forward(batch)

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, batch : Data) -> Data:
        output, target = batch['output'], batch['target']
        loss = self.loss(output, target)
        accuracy = ((output > 0.0) == target.bool()).float().mean()
        return {'loss': loss, 'accuracy': accuracy}
    
class CrossEntropyWithLogitsLoss(BaseLoss):
    def __init__(
        self,
        report_top_k: int = 5,
        ):
        super().__init__()
        self.report_top_k = report_top_k
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, batch : Data) -> Data:
        output, target = batch['output'], batch['target']
        loss = self.loss(output, target)
        labels = target.argmax(dim=-1)
        
        # 1 if the top predicted label is correct
        accuracy = ((output.argmax(dim=-1) == labels).float().mean())
        
        # calculate the top k accuracy
        # it is 1 if the true label is in the top k predicted labels
        k = self.report_top_k
        top_k_acc = (output.topk(k, dim=-1).indices == labels.unsqueeze(1)).any(dim=-1).float().mean()
        return {'loss': loss, 'accuracy': accuracy, f'top_{k}_accuracy': top_k_acc}