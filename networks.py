from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.video import r3d_18, R3D_18_Weights
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
        if in_channels != 3:
            print(f"Setting in_channels from 3 to {in_channels}...")
            self.model.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                padding=3,
                bias=False,
            )
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


class ResNet3D(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_size: int = 512, dropout_p: float = 0.5,
        freeze_until: str | None = "None", use_pretrained: bool = True):
        super().__init__()

        if use_pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)
    
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, num_classes),
        )

        if freeze_until is not None:
            self._freeze_backbone(until=freeze_until)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _freeze_backbone(self, until: str):
        freeze_names = {"layer3": ["stem", "layer1", "layer2"],
                        "layer4": ["stem", "layer1", "layer2", "layer3"],
                        "classifier_only": ["stem", "layer1", "layer2", "layer3", "layer4"]}
        for name, module in self.backbone.named_children():
            if until in freeze_names and name in freeze_names[until]:
                for p in module.parameters(): p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1, 3, 4) #Change from [B, T, C, H, W] to [B, C, T, H, W]
        feats = self.backbone(x)
        return self.head(feats)
