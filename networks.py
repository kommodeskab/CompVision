import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Binary(nn.Module):
    def __init__(
        self, 
        num_classes: int = 2,
        hidden_size: int = 512,
        use_pretrained: bool = True,
        ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.model = resnet18(weights=weights)
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