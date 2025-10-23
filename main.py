"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from models import ClassificationModel, PerFrameClassificationModel, TwoStreamClassificationModel
from datasets import FrameImageDataset, FrameVideoDataset
from networks import ResNet18Binary, ResNet3D
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
from callbacks import SetDropoutProbCallback

HIDDEN_SIZE = 128
VAL_EVERY_N_STEPS = 100

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--leakage", action="store_true", help="enable leakage (default: False)")
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--optimizer", type=str, required=True, choices=["adamw", "sgd"])
    argparser.add_argument("--experiment", type=str, required=True, choices=["per_frame", "late_fusion", "early_fusion", "3d_cnn", "two_stream"])
    argparser.add_argument("--num_workers", type=int, default=12)
    argparser.add_argument("--max_steps", type=int, default=-1)
    argparser.add_argument("--use_pretrained", type=bool, default=False)
    argparser.add_argument("--dropout_prob", type=float, default=0.3)
    args = argparser.parse_args()
    
    print("Experiment configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.optimizer == "sgd":
        optimizer = partial(SGD, lr=1e-3, momentum=0.9, weight_decay=0.1)
    elif args.optimizer == "adamw":
        optimizer = partial(AdamW, lr=1e-4, weight_decay=0.1)
        
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=20),
        'monitor': 'val_loss',
        'interval': 'step',
        'frequency': VAL_EVERY_N_STEPS
    }
    
    loss_fn = CrossEntropyWithLogitsLoss(report_top_k=3)
    
    if args.experiment == "per_frame":
        # per_frame trains on individual images, therefore we use FrameImageDataset
        trainset = FrameImageDataset(leakage=args.leakage, split="train")
        valset = FrameImageDataset(leakage=args.leakage, split="val")
        
        network = ResNet18Binary(
            num_classes=trainset.num_classes,
            hidden_size=HIDDEN_SIZE,
            use_pretrained=args.use_pretrained,
        )
        
        model = PerFrameClassificationModel(
            network=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    
    elif args.experiment == "3d_cnn":
        trainset = FrameVideoDataset(leakage=args.leakage, split="train")
        valset = FrameVideoDataset(leakage=args.leakage, split="val")

        network = ResNet3D(
            num_classes=trainset.num_classes, 
            hidden_size=HIDDEN_SIZE,
            use_pretrained=args.use_pretrained
            )

        model = ClassificationModel(
            network=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    
    elif args.experiment == "two_stream":
        assert not args.leakage, "Two-stream model not implemented for leakage setup."
        trainset = FrameVideoDataset(leakage=args.leakage, split="train")
        valset = FrameVideoDataset(leakage=args.leakage, split="val")
        
        network = ResNet18Binary(
            num_classes=trainset.num_classes,
            in_channels=18,
            hidden_size=HIDDEN_SIZE,
            use_pretrained=args.use_pretrained,
        )

        image_network = ResNet18Binary(
            num_classes=trainset.num_classes,
            in_channels=3,
            hidden_size=HIDDEN_SIZE,
            use_pretrained=args.use_pretrained,
        )
        
        model = TwoStreamClassificationModel(
            network=network,
            image_network=image_network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )

    testset = FrameVideoDataset(leakage=args.leakage, split="test")

    datamodule = BaseDM(
        trainset=trainset,
        valset=valset,
        testset=testset,
        num_workers=args.num_workers,
        batch_size=args.batch_size
        )
        
    callbacks = [
        DeviceStatsMonitor(),
        ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, every_n_train_steps=VAL_EVERY_N_STEPS),
        EarlyStopping(monitor='val_accuracy', patience=50, mode='max'), 
        LearningRateMonitor(),
        ModelSummary(max_depth=2),
        Timer(),
        SetDropoutProbCallback(new_prob=args.dropout_prob),
    ]
    
    logger = TensorBoardLogger(
        'lightning_logs', 
        name=args.experiment, 
        version=get_timestamp(),
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
        logger=logger,
        )
    
    logger.log_hyperparams(vars(args))
    
    torch.set_float32_matmul_precision('high')
    
    pytorch_lightning.seed_everything(42)
    trainer.fit(model = model, datamodule = datamodule)
    trainer.test(model = model, datamodule = datamodule)