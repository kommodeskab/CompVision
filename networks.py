import torch.nn as nn
from torch import Tensor
import torch
import segmentation_models_pytorch as smp

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class UNet(torch.nn.Module):
    def __init__(self, input_shape, n_classes=2, encoder='resnet34', pretrained=False):
        super().__init__()
        in_channels = input_shape[0]

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=n_classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x