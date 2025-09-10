import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from functools import partial
from dataclasses import dataclass
from torch import Tensor
import torch
from typing import Any
from datasets import BaseBatch
from pytorch_lightning.utilities import grad_norm

class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        optimizer : partial[Optimizer] | None = None,
        lr_scheduler : dict[str, partial[LRScheduler] | str] | None = None,
        ):
        super().__init__()
        
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
    
    def on_before_optimizer_step(self, optimizer : Optimizer) -> None:
        if self.global_step % 100 == 0: # log gradients every 100 steps
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)
    
    def forward(self, x : Any) -> Tensor:
        raise NotImplementedError("This method should be implemented in subclassses")
        
    def common_step(self, batch : BaseBatch, batch_idx : int):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def training_step(self, batch : BaseBatch, batch_idx : int):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch : BaseBatch, batch_idx : int):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."
        
        optim = self.partial_optimizer(self.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }
        
class ClassificationModel(BaseLightningModule):
    def __init__(
        self,
        network : pl.LightningModule,
        optimizer : partial | None = None,
        lr_scheduler : dict[str, partial[LRScheduler] | str] | None = None,
        ):
        """
        A base classification model that wraps around a network and provides training and validation steps.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.save_hyperparameters(ignore=['network', 'optimizer', 'lr_scheduler'])
        self.network = network
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x : Tensor) -> Tensor:
        return self.network(x)
                
    def common_step(self, batch : BaseBatch, batch_idx : int):
        x, y = batch['input'], batch['target']
        x : Tensor 
        y : Tensor
        y_hat = self.forward(x).flatten()
        loss = self.loss_fn(y_hat, y.float())
        return loss
        