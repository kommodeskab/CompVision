"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from models import ClassificationModel
from datasets import FrameImageDataset, FrameVideoDataset
from networks import ResNet18Binary
from losses import CrossEntropyWithLogitsLoss
from argparse import ArgumentParser
from functools import partial
from torch.optim import AdamW, SGD
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
import pytorch_lightning

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--image_size", type=int, required=True)
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--optimizer", type=str, required=True, choices=["adamw", "sgd"])
    argparser.add_argument("--name", type=str, default="experiment")
    argparser.add_argument("--num_workers", type=int, default=12)
    args = argparser.parse_args()
    
    trainset = FrameImageDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage/", 
        split="train"
        )
    valset = FrameImageDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage/", 
        split="val"
        )
    datamodule = BaseDM(
        dataset=trainset, 
        val_dataset=valset, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size
        )
    network = ResNet18Binary(
        num_classes=trainset.num_classes,
        hidden_size=128,
        )
    loss_fn = CrossEntropyWithLogitsLoss(
        report_top_k=3
        )

    if args.optimizer == "sgd":
        optimizer = partial(SGD, lr=1e-3, momentum=0.9, weight_decay=0.1)
    elif args.optimizer == "adamw":
        optimizer = partial(AdamW, lr=1e-4, weight_decay=0.1)
        
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=20),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 1
    }
    
    model = ClassificationModel(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
        
    callbacks = [
        DeviceStatsMonitor(),
        ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1),
        EarlyStopping(monitor='val_accuracy', patience=50, mode='max'), 
        LearningRateMonitor(),
        ModelSummary(max_depth=2),
        Timer(),
    ]
    
    logger = TensorBoardLogger('lightning_logs', name=args.name, log_graph=False)
    
    trainer = Trainer(
        max_epochs=200, 
        max_steps=500_000, 
        accelerator="gpu", 
        log_every_n_steps=10, 
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        logger=logger,
        )
    
    torch.set_float32_matmul_precision('high')
    
    pytorch_lightning.seed_everything(42)
    trainer.fit(model = model, datamodule = datamodule)
    trainer.test(model = model, datamodule = datamodule)