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
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, batch : Data) -> Data:
        out = batch['out']
        target = batch['target']
        loss = self.criterion(out, target.float())
        return {
            'loss': loss,
            **batch
        }
    
class FocalLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.reduction = 'mean'

    def forward(self, batch : Data) -> Data:
        out = batch['out']
        target = batch['target']
        bce_loss = nn.functional.binary_cross_entropy_with_logits(out, target.float(), reduction='none')
        probas = torch.sigmoid(out)
        p_t = target * probas + (1 - target) * (1 - probas)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * bce_loss
        loss = focal_loss.mean()
        return{
            'loss': loss
        }

def binary_mask_given_points(
    height: int, 
    width: int, 
    pos_points: torch.Tensor,
    neg_points: torch.Tensor,
    ) -> torch.Tensor:
    """
    Given a height and width, and two sets of points (positive and negative),
    this function generates a binary mask using the closest point approach.
    The pos_points and neg_points should be tensors of shape (B, N, 2).
    """
    pos_points = pos_points.float()
    neg_points = neg_points.float()
    
    grid_y, grid_x = torch.meshgrid(
    torch.arange(height, dtype=torch.float32), 
    torch.arange(width, dtype=torch.float32), 
    indexing='ij'
    )
    pixel_coords = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
    pixel_coords_batched = pixel_coords.unsqueeze(0).to(pos_points.device)
    all_dists_pos = torch.cdist(pixel_coords_batched, pos_points, p=2.0)
    all_dists_neg = torch.cdist(pixel_coords_batched, neg_points, p=2.0)
    min_dists_pos, _ = torch.min(all_dists_pos, dim=2)
    min_dists_neg, _ = torch.min(all_dists_neg, dim=2)
    binary_images_flat = (min_dists_pos < min_dists_neg)
    binary_images = binary_images_flat.reshape(pos_points.shape[0], height, width).float()
    return binary_images

class PointSupervisionLoss(BaseLoss):
    def __init__(
        self,
        ):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, batch : Data) -> Data:
        out = batch['out'] # shape (B, C, H, W)
        positive_points = batch['pos_clicks'] # shape (B, N, 2)
        negative_points = batch['neg_clicks'] # shape (B, M, 2)
        # assert that all the points are within the image dimensions
        
        pseudo_target = binary_mask_given_points(
            height=out.shape[2],
            width=out.shape[3],
            pos_points=positive_points,
            neg_points=negative_points,
        ) # shape (B, H, W)
        
        pseudo_target = pseudo_target.unsqueeze(1)  # Add channel dimension
        loss = self.criterion(out, pseudo_target.float())
        
        return {
            'loss': loss,
            'pseudo_target': pseudo_target,
            **batch
        }
        
class SegmentationMetrics(BaseLoss):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch : Data) -> Data:
        output, target = batch['out'], batch['target']
        
        # the output contains logits, so threshold at 0.0
        output = (output > 0.0).float().squeeze()
        
        dice = (2 * (output * target).sum()) / (output.sum() + target.sum() + 1e-8)
        intersection = (output * target).sum()
        union = output.sum() + target.sum() - intersection
        accuracy = ((output == target).float().sum()) / target.numel()
        sensitivity = intersection / (target.sum() + 1e-8)
        specificity = (((1 - output) * (1 - target)).sum()) / ((1 - target).sum() + 1e-8)

        return {
            'dice': dice, 
            'accuracy': accuracy,
            'intersection over union': intersection / (union + 1e-8), 
            'sensitivity': sensitivity, 
            'specificity': specificity
        }