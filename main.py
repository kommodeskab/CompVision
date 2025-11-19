"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from models import BaseLightningModule
from argparse import ArgumentParser
from functools import partial
from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelSummary,
    ModelCheckpoint,
)
import pytorch_lightning
from utils import get_timestamp
from callbacks import (
    SetDropoutProbCallback, 
    LogLossCallback, 
    LogGradientsCallback,
)
import logging

logger = logging.getLogger(__name__)

VAL_EVERY_N_STEPS = 100

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--experiment", type=str, required=True)
    argparser.add_argument("--num_workers", type=int, default=12)
    argparser.add_argument("--max_steps", type=int, default=-1)
    argparser.add_argument("--dropout_prob", type=float, default=0.3)
    argparser.add_argument("--run_name", type=str, required=False, default=None)
    
    args = argparser.parse_args()
    
    logger.info("Experiment configuration:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
        
    optimizer = partial(AdamW, lr=1e-4, weight_decay=0.1)
    
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=10),
        'monitor': 'val_loss',
        'interval': 'step',
        'frequency': VAL_EVERY_N_STEPS
    }
    
    model: BaseLightningModule = ...
    datamodule: BaseDM = ...

    callbacks = [
        LogLossCallback(),
        LogGradientsCallback(log_every_n_steps=100),
        DeviceStatsMonitor(),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, every_n_train_steps=VAL_EVERY_N_STEPS),
        EarlyStopping(monitor='val_loss', patience=20, mode='min'), 
        LearningRateMonitor(),
        ModelSummary(max_depth=2),
        SetDropoutProbCallback(new_prob=args.dropout_prob),
    ]
    
    version = get_timestamp() if args.run_name is None else args.run_name
    print(f"Logging to experiment '{args.experiment}' with version '{version}'")
    
    tensorboardlogger = TensorBoardLogger(
        'lightning_logs', 
        name=args.experiment, 
        version=version,
        log_graph=False
        )
    
    trainer = Trainer(
        max_epochs=-1,
        max_steps=args.max_steps,
        accelerator="gpu", 
        log_every_n_steps=1, 
        callbacks=callbacks,
        check_val_every_n_epoch=None,
        val_check_interval=VAL_EVERY_N_STEPS,
        logger=tensorboardlogger,
        max_time="00:11:00:00"  # 11 hours
        )
    
    tensorboardlogger.log_hyperparams(vars(args))
    
    torch.set_float32_matmul_precision('high')
    
    pytorch_lightning.seed_everything(42)
    trainer.fit(model = model, datamodule = datamodule)
    trainer.test(model = model, datamodule = datamodule)