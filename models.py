import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from functools import partial
from torch import Tensor
import torch
from typing import Any, Dict, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities import grad_norm
from dataloader import BaseDM
import torch.nn as nn
import random
import numpy as np
from contextlib import contextmanager
from losses import BaseLoss
from utils import DatasetType, OptimizerType, LRSchedulerType
    
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
        optimizer : OptimizerType = None,
        lr_scheduler : LRSchedulerType = None,
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

    def log_loss(self, loss: Data, prefix: str) -> Tensor:
        loss = {f'{prefix}_{k}': v for k, v in loss.items()}
        self.log_dict(loss, prog_bar=True)
        return loss[f'{prefix}_loss']

    def training_step(self, batch : Data, batch_idx : int) -> Tensor:
        step = self.common_step(batch, batch_idx)
        return self.log_loss(step, 'train')

    def validation_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temp_seed(0):
            step = self.common_step(batch, batch_idx)
        return self.log_loss(step, 'val')

    def test_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temp_seed(0):
            step = self.common_step(batch, batch_idx)
        return self.log_loss(step, 'test')

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
        
from utils import OptimizerType, LRSchedulerType
        
class ClassificationModel(BaseLightningModule):
    def __init__(
        self,
        network : nn.Module,
        loss_fn : Optional[BaseLoss] = None,
        optimizer : OptimizerType = None,
        lr_scheduler : LRSchedulerType = None,
        ):
        """
        A base classification model that wraps around a network and provides training and validation steps.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.network = network
        self.loss_fn = loss_fn
        
    @property
    def example_input_array(self):
        return self.trainset[0]['input'].unsqueeze(0)
        
    def forward(self, x : Tensor) -> Tensor:
        return self.network(x)
    
    def classify(self, x : Tensor) -> Tensor:
        logits = self.forward(x)
        if logits.shape[-1] == 1:
            # binary classification
            return (logits > 0.0).flatten().long()

        # multi class classification
        return logits.argmax(dim=-1)

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        x, y = batch['input'], batch['target']
        print('Look here: ', x.shape)
        out = self.forward(x)
        loss = self.loss_fn({
            'output': out,
            'target': y.float()
        })
        return loss

class PerFrameClassificationModel(ClassificationModel):
    def __init__(
        self,
        network: nn.Module,
        loss_fn: Optional[BaseLoss] = None,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None
    ):
        super().__init__(
            network=network, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler
            )

    def test_step(self, batch : Data, batch_idx : int) -> Tensor:
        # test batches are videos of shape (B, T, C, H, W)
        # we will use our image classification model to classify each frame
        # the final result is then obtained by averaging the predictions across all frames
        x, y = batch['input'], batch['target']
        B, T, *rest = x.shape
        x = x.view(-1, *rest) # flatten the temporal dimension
        out = self.forward(x) # process all frames in parallel
        out = out.view(B, T, -1).mean(dim=1) # average the predictions across all frames
        loss = self.loss_fn({
            'output': out,
            'target': y.float()
        })
        loss = self.log_loss(loss, 'test')
        return loss

class EarlyFusionModel(ClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def example_input_array(self):
        x = self.trainset[0]['input'].unsqueeze(0)
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        return x

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        x, y = batch['input'], batch['target']
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        out = self.forward(x)
        loss = self.loss_fn({
            'output': out,
            'target': y.float()
        })
        return loss
    
class LateFusionModel(ClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def example_input_array(self):
        x = self.trainset[0]['input'].unsqueeze(0)
        return x

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        x, y = batch['input'], batch['target']
        out = self.forward(x)
        loss = self.loss_fn({
            'output': out,
            'target': y.float()
        })
        return loss

        
class TwoStreamClassificationModel(ClassificationModel):
    def __init__(
        self,
        network: nn.Module,
        image_network : nn.Module,
        loss_fn: Optional[BaseLoss] = None,
        optimizer: OptimizerType = None,
        lr_scheduler: LRSchedulerType = None
    ):            
        super().__init__(
            network=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        self.image_network = image_network

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        # do early fusion on the optical flow
        x, y = batch['optical_flow'], batch['target']
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        out = self.forward(x)
        loss = self.loss_fn({
            'output': out,
            'target': y.float()
        })
        return loss
    
    @property
    def example_input_array(self):
        flow = self.trainset[0]['optical_flow'].unsqueeze(0)
        B, T, C, H, W = flow.shape
        flow = flow.view(B, T * C, H, W)
        return flow

    def image_forward(self, x: Tensor) -> Tensor:
        return self.image_network(x)

    def test_step(self, batch: Data, batch_idx: int) -> Tensor:
        flow, video, target = batch['optical_flow'], batch['input'], batch['target']
        
        # use image network to predict on all the images in the video
        B, T, *rest = video.shape
        video = video.view(-1, *rest)
        img_pred = self.image_forward(video)
        img_pred = img_pred.view(B, T, -1).mean(dim=1)

        # predict on optical flow using early fusion
        B, T, C, H, W = flow.shape
        flow = flow.view(B, T * C, H, W)
        flow_pred = self.forward(flow)

        # final prediction is a mean
        final_prediction = (img_pred + flow_pred) / 2
        
        loss = self.loss_fn({
            'output': final_prediction,
            'target': target.float()
        })
        return self.log_loss(loss, 'test')