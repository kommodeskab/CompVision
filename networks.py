import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights

class BaseClassifier(nn.Module):
    """
    A very simple binary classifier with a single linear layer.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        out = self.linear(x) # don't apply sigmoid here, use BCEWithLogitsLoss
        return out


class ResNet18Binary(nn.Module):
    def __init__(
        self, 
        num_classes: int = 2,
        in_channels: int = 3,
        hidden_size: int = 512,
        ):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1.in_channels = in_channels
        num_ftrs = self.model.fc.in_features #Number of features in the last layer
        out_features = 1 if num_classes == 2 else num_classes
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  