from pytorch_lightning import Callback
import torch.nn as nn
from pytorch_lightning import Trainer
from models import BaseLightningModule
import pytorch_lightning as pl
from utils import Data, OptimizerType
from pytorch_lightning.utilities import grad_norm
import logging
from models import BoundingBoxClassificationModel

logger = logging.getLogger(__name__)

class LogClassBalanceCallback(Callback):
    def __init__(self):
        """ Callback for logging class balance. Used for sanity checking the class distribution during training."""
        super().__init__()
        self.count = {
            'train': 0,
            'val': 0,
            'test': 0,
        }
        self.balance = {
            'train': 0.0,
            'val': 0.0,
            'test': 0.0,
        }
        
    def update_balance(self, phase: str, batch: Data, pl_module: BoundingBoxClassificationModel):
        if pl_module.current_epoch != 0:
            return # only log during first epoch to reduce overhead
        
        target = batch['target']
        batch_balance = target.mean().item()
        self.count[phase] += 1
        self.balance[phase] += (batch_balance - self.balance[phase]) / self.count[phase]
        pl_module.log_dict({
            f'{phase}_class_balance_running': self.balance[phase],
            f'{phase}_n_potholes': batch['n_potholes'].float().mean(),
            f'{phase}_n_background': batch['n_background'].float().mean(),
        })
                
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: BoundingBoxClassificationModel, batch: Data, batch_idx: int) -> None:
        self.update_balance('train', batch, pl_module)
        
    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: BoundingBoxClassificationModel, batch: Data, batch_idx: int) -> None:
        self.update_balance('val', batch, pl_module)
        
    def on_test_batch_start(self, trainer: pl.Trainer, pl_module: BoundingBoxClassificationModel, batch: Data, batch_idx: int) -> None:
        self.update_balance('test', batch, pl_module)

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