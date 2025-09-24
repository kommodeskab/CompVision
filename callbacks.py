from pytorch_lightning import Callback
import torch.nn as nn
from pytorch_lightning import Trainer
from models import BaseLightningModule

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
        print(f"Set dropout probability to {new_prob} for {n_modules} modules.", flush=True)

    def on_train_start(self, trainer : Trainer, pl_module : BaseLightningModule):
        self.set_dropout_prob(
            pl_module,
            self.new_prob,
        )