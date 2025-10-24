import torch.nn as nn
from torch import Tensor, no_grad
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
                bias=False)
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

class ResNet18LateFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_frames: int = 10,
        hidden_size: int = 512,
        fusion: str = "concat",
    ):
        super().__init__()
        # Load pretrained ResNet backbone
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = base_model.fc.in_features
        
        # Remove the classifier head
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # up to avgpool
        self.feature_extractor.eval()  # optionally freeze

        # Define classifier head (after fusion)
        fusion_input_size = num_ftrs * num_frames
        out_features = 1 if num_classes == 2 else num_classes

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x):
        """
        x: (B, T, 3, H, W)  — batch of videos
        """
        B, T, C, H, W = x.shape

        # Extract per-frame features
        x = x.view(B * T, C, H, W)              # flatten frames into batch dimension
        with no_grad():                   # freeze CNN if desired
            feats = self.feature_extractor(x)   # (B*T, 512, 1, 1)
        feats = feats.view(B, T, -1)            # (B, T, 512)

        # Fuse features over time
        fused = feats.view(B, -1)           # (B, T*512)

        # Classify
        out = self.classifier(fused)
        return out
