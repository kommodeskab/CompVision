"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from argparse import ArgumentParser
from datasets import Hotdog_NotHotdog
from networks import ResNet18Binary
from models import ClassificationModel
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
    Timer,
    ModelCheckpoint,
)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--image_size", type=int, required=True)
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--name", type=str, default="experiment")
    args = argparser.parse_args()

    IMAGE_SIZE : int = args.image_size
    BATCH_SIZE : int = args.batch_size
    NAME : str = args.name

    train_dataset = Hotdog_NotHotdog(train=True, image_size=IMAGE_SIZE)
    val_dataset = Hotdog_NotHotdog(train=False, image_size=IMAGE_SIZE)

    optimizer = partial(AdamW, lr=1e-5, weight_decay=0.1)
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=20),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 1
    }

    network = ResNet18Binary()
    model = ClassificationModel(
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    datamodule = BaseDM(dataset=train_dataset, val_dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    callbacks = [
        DeviceStatsMonitor(),
        ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1),
        EarlyStopping(monitor='val_accuracy', patience=50, mode='max'), 
        LearningRateMonitor(),
        ModelSummary(),
        Timer(),
    ]
    
    logger = TensorBoardLogger('lightning_logs', name=NAME)
    
    trainer = Trainer(
        max_epochs=-1, 
        max_steps=500_000, 
        accelerator="gpu", 
        log_every_n_steps=10, 
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        logger=logger,
        )
    
    torch.set_float32_matmul_precision('high')
    
    trainer.fit(model, datamodule=datamodule)