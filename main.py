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
from utils import get_timestamp


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--image_size", type=int, required=True)
    argparser.add_argument("--leakage", type=bool, default=False)
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--optimizer", type=str, required=True, choices=["adamw", "sgd"])
    argparser.add_argument("--experiment", type=str, required=True, choices=["image_model", "late_fusion", "early_fusion", "3d_cnn"])
    argparser.add_argument("--num_workers", type=int, default=12)
    args = argparser.parse_args()

    if args.experiment == "image_model":
        # image model trains on individual images, therefore we use FrameImageDataset
        trainset = FrameImageDataset(leakage=args.leakage, split="train")
        valset = FrameImageDataset(leakage=args.leakage, split="val")
    else:
        # all other models trains on full videos, therefore we use FrameVideoDataset
        trainset = FrameVideoDataset(leakage=args.leakage, split="train")
        valset = FrameVideoDataset(leakage=args.leakage, split="val")

    # we test all models on video
    testset = FrameVideoDataset(leakage=args.leakage, split="test")

    datamodule = BaseDM(
        trainset=trainset,
        valset=valset,
        testset=testset,
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
    
    logger = TensorBoardLogger(
        'lightning_logs', 
        name=args.experiment, 
        version=get_timestamp(),
        log_graph=False
        )
    
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