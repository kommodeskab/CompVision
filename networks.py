import torch.nn as nn
from torch import Tensor
import segmentation_models_pytorch as smp
import torch

class CNNAutoEncoder(torch.nn.Module):  
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int
        ):
        super(CNNAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, H/2, W/2]
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [128, 256, H/4, W/4]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # [256, 512, H/8, W/8]
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 256, H/4, W/4]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 128, H/2, W/2]
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, num_classes, H, W]
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class UNet(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        encoder='resnet34', 
        pretrained=False
        ):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x