import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights

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



class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features #Number of features in the last layer
        self.model.fc = nn.Linear(num_ftrs, 1) #Ensures that we have a single output and not 1000

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  