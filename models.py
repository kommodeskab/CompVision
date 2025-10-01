import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from functools import partial
from torch import Tensor
import torch
from typing import Any, Dict, Tuple
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities import grad_norm
from dataloader import BaseDM
from torch.utils.data import Dataset
import torch.nn as nn
import random
import numpy as np
from contextlib import contextmanager
    
Data = Dict[str, Tensor]

@contextmanager
def temp_seed(seed : int):
    # temporarily set the random seed for reproducibility
    # and restore the original state afterwards
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

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
        if self.global_step % 100 == 0: # Log gradient norms every 100 steps
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    @property
    def logger(self) -> TensorBoardLogger:
        return self.trainer.logger
    
    @property
    def datamodule(self) -> BaseDM:
        return self.trainer.datamodule
    
    @property
    def trainset(self) -> Dataset:
        return self.datamodule.train_dataset

    @property
    def valset(self) -> Dataset:
        return self.datamodule.val_dataset

    def forward(self, *x : Any) -> Tensor:
        raise NotImplementedError("This method should be implemented in subclassses")

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def training_step(self, batch : Data, batch_idx : int) -> Tensor:
        step = self.common_step(batch, batch_idx)
        step = {f'train_{k}': v for k, v in step.items()}
        loss = step['train_loss']
        self.log_dict(step, prog_bar=True)
        return loss

    def validation_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temp_seed(0):
            step = self.common_step(batch, batch_idx)
        step = {f'val_{k}': v for k, v in step.items()}
        loss = step['val_loss']
        self.log_dict(step, prog_bar=True)
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
        network : nn.Module,
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
    
    def classify(self, x : Tensor) -> Tensor:
        logits = self.forward(x)
        return (logits > 0.0).long()
                
    def common_step(self, batch : Data, batch_idx : int):
        x, y = batch['input'], batch['target']
        y_hat = self.forward(x).flatten()
        loss = self.loss_fn(y_hat, y.float())
        accuracy = ((y_hat > 0.0) == y.bool()).float().mean()
        return {
            'loss': loss,
            'accuracy': accuracy
        }
