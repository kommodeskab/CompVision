import pytorch_lightning as pl
from losses import BaseLoss
from torch import Tensor
import torch
from typing import Any
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import BaseDM
import torch.nn as nn
import random
import numpy as np
from contextlib import contextmanager
from losses import BaseLoss
from utils import DatasetType, OptimizerType, LRSchedulerType, Data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
        optimizer : OptimizerType = None,
        lr_scheduler : LRSchedulerType = None,
        ):
        super().__init__()
        
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler

    @property
    def logger(self) -> TensorBoardLogger:
        return self.trainer.logger
    
    @property
    def experiment(self) -> SummaryWriter:
        return self.logger.experiment
    
    @property
    def datamodule(self) -> BaseDM:
        return self.trainer.datamodule
    
    @property
    def trainset(self) -> DatasetType:
        return self.datamodule.trainset

    @property
    def valset(self) -> DatasetType:
        return self.datamodule.valset

    @property
    def testset(self) -> DatasetType:
        return self.datamodule.testset

    def forward(self, *x : Any) -> Tensor:
        raise NotImplementedError

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        """
        Should return the loss as a dictionary for the given batch.
        The loss dictionary can contain multiply keys for logging, 
        but gradient steps will be taken on the value with the key 'loss'
        """
        raise NotImplementedError
    
    def log_image(self, tag: str, img_tensor: Tensor):
        self.experiment.add_image(tag, img_tensor, self.global_step)

    def add_figure(self, tag: str, figure: plt.Figure):
        self.experiment.add_figure(tag, figure, self.global_step)

    def training_step(self, batch : Data, batch_idx : int) -> Tensor:
        return self.common_step(batch, batch_idx)

    def validation_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temp_seed(0):
            return self.common_step(batch, batch_idx)

    def test_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temp_seed(0):
            return self.common_step(batch, batch_idx)

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
        network: nn.Module,
        loss_fn: BaseLoss,
        optimizer : OptimizerType = None,
        lr_scheduler : LRSchedulerType = None,
        ):
        super().__init__(optimizer, lr_scheduler)
        self.network = network
        self.loss_fn = loss_fn
        
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    
    def common_step(self, batch : Data, batch_idx : int) -> Data:
        inputs = batch['input']
        outputs = self.forward(inputs)
        loss = self.loss_fn.forward({
            'out': outputs,
            **batch
        })
        return loss