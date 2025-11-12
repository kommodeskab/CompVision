from pytorch_lightning import Callback
import torch.nn as nn
from pytorch_lightning import Trainer
from models import BaseLightningModule
import pytorch_lightning as pl
from utils import Data, OptimizerType
from pytorch_lightning.utilities import grad_norm
from losses import PointSupervisionLoss
import matplotlib.pyplot as plt
import logging
from losses import SegmentationMetrics

logger = logging.getLogger(__name__)

class SegmentationMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = SegmentationMetrics()
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        metrics = self.metrics(outputs)
        pl_module.log_dict({f'val_{k}': v for k, v in metrics.items() if v.numel() == 1}, prog_bar=False)
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        metrics = self.metrics(outputs)
        pl_module.log_dict({f'train_{k}': v for k, v in metrics.items() if v.numel() == 1}, prog_bar=False)
    
    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        metrics = self.metrics(outputs)
        pl_module.log_dict({f'test_{k}': v for k, v in metrics.items() if v.numel() == 1}, prog_bar=False)

class VisualizeSegmentationCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        # plot the first image in the batch alongside the estimated and ground truth masks
        input_img = batch['input'][0]  # shape (C, H, W)
        gt_mask = batch['target'][0].squeeze()  # shape (H, W)
        pred_mask = outputs['out'][0].squeeze()  # shape (H, W)
        pred_mask = (pred_mask > 0.0).float()  # threshold at 0.0

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax: list[plt.Axes]
        ax[0].imshow(input_img.permute(1, 2, 0).cpu().clip(0.0, 1.0))
        ax[0].set_title('Input Image')
        ax[1].imshow(gt_mask.cpu().clip(0.0, 1.0), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(pred_mask.cpu().clip(0.0, 1.0), cmap='gray')
        ax[2].set_title('Predicted Mask')
        pl_module.add_figure(f'Segmentation_{batch_idx}', fig)
        plt.close(fig)

class LogPointLossCallback(Callback):
    def __init__(self):
        """
        Assuming pl_module uses the PointSupervisionLoss as its loss function,
        this callback logs the positive and negative points on the pseudo-target images
        """
        super().__init__()
        self.has_logged = False
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        if not isinstance(pl_module.loss_fn, PointSupervisionLoss):
            logger.info("LogPointLossCallback: The model's loss function is not PointSupervisionLoss. Disabling this callback.")
            self.has_logged = True  # disable logging if not using PointSupervisionLoss

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        if self.has_logged:
            return

        # what the model is trained on
        pos_points = batch['pos_clicks']
        neg_points = batch['neg_clicks']
        
        # what the loss function generated
        pseudo_target = outputs['pseudo_target']
        real_target = batch['target']
        
        for i in range(pseudo_target.shape[0]):
            # make a plot of the pseudo target with the points overlayed
            pseudo_target_img = pseudo_target[i].squeeze().cpu() # shape (H, W)
            pos = pos_points[i].cpu() # shape (N, 2)
            neg = neg_points[i].cpu() # shape (M, 2)
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax: list[plt.Axes]
            ax[0].imshow(pseudo_target_img.clip(min=0.0, max=1.0), cmap='gray')
            for p in pos:
                ax[0].plot(p[1], p[0], 'go', label='Positive clicks')
            for n in neg:
                ax[0].plot(n[1], n[0], 'ro', label='Negative clicks')
            ax[0].set_title(f'Clicks and pseudo mask')
            
            ax[1].imshow(real_target[i].squeeze().cpu().clip(min=0.0, max=1.0), cmap='gray')
            ax[1].set_title('Real Target Mask')
            
            pl_module.add_figure(f'Point_Supervision_sample_{i}', fig)
            plt.close(fig)
        
        self.has_logged = True

class LogGradientsCallback(Callback):
    def __init__(self, log_every_n_steps: int):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: BaseLightningModule, optimizer: OptimizerType) -> None:
        if trainer.global_step % self.log_every_n_steps == 0:
            norms = grad_norm(pl_module, norm_type=2)
            pl_module.log_dict(norms)

class LogLossCallback(Callback): 
    def __init__(self):
        super().__init__()
       
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        pl_module.log_dict({f'train_{k}': v for k, v in outputs.items() if v.numel() == 1}, prog_bar=True)
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        pl_module.log_dict({f'val_{k}': v for k, v in outputs.items() if v.numel() == 1}, prog_bar=True)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule, outputs: Data, batch: Data, batch_idx: int) -> None:
        pl_module.log_dict({f'test_{k}': v for k, v in outputs.items() if v.numel() == 1}, prog_bar=True)

class SetDropoutProbCallback(Callback):
    def __init__(
        self,
        new_prob : float,
    ):
        assert isinstance(new_prob, float), "Dropout probability must be a float."
        assert 0 <= new_prob <= 1, "Dropout probability must be between 0 and 1."
        self.new_prob = new_prob
        
    @staticmethod
    def set_dropout_prob(model : nn.Module, new_prob : float):
        n_modules = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_prob
                n_modules += 1
        logger.info(f"Set dropout probability to {new_prob} for {n_modules} modules.")

    def on_train_start(self, trainer : Trainer, pl_module : BaseLightningModule):
        self.set_dropout_prob(
            pl_module,
            self.new_prob,
        )