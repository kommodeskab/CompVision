"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from models import BaseLightningModule, ClassificationModel
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
    Timer,
    ModelCheckpoint,
)
import pytorch_lightning
from utils import get_timestamp
from callbacks import (
    SetDropoutProbCallback, 
    LogLossCallback, 
    LogGradientsCallback,
    LogPointLossCallback,
    SegmentationMetricsCallback,
    VisualizeSegmentationCallback
)
from losses import PointSupervisionLoss, BCELoss, FocalLoss
from networks import UNet, CNNAutoEncoder
from datasets import PH2Dataset, DRIVEDataset

VAL_EVERY_N_STEPS = 100

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--experiment", type=str, required=True)
    argparser.add_argument("--network", type=str, required=True, choices=['unet', 'pretrained_unet', 'cnn'])
    argparser.add_argument("--dataset", type=str, required=True, choices=['ph2', 'drive'])
    argparser.add_argument("--loss", type=str, required=False, choices=['bce', 'weighted_bce', 'focal', 'point_supervision'])
    argparser.add_argument("--img_size", type=int, default=512)
    argparser.add_argument("--num_workers", type=int, default=12)
    argparser.add_argument("--max_steps", type=int, default=-1)
    argparser.add_argument("--dropout_prob", type=float, default=0.3)
    argparser.add_argument("--run_name", type=str, required=False, default=None)
    argparser.add_argument("--num_pos_clicks", type=int, required=False, default=5)
    argparser.add_argument("--num_neg_clicks", type=int, required=False, default=5)
    argparser.add_argument("--intensity", type=int, required=False, default=8)
    
    args = argparser.parse_args()
    
    print("Experiment configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    optimizer = partial(AdamW, lr=1e-4, weight_decay=0.1)
    
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=10),
        'monitor': 'val_loss',
        'interval': 'step',
        'frequency': VAL_EVERY_N_STEPS
    }
    
    if args.dataset == 'ph2':
        train_dataset = PH2Dataset('train', img_size=args.img_size, n_pos=args.num_pos_clicks, n_neg=args.num_neg_clicks, intensity=args.intensity)
        val_dataset = PH2Dataset('val', img_size=args.img_size, n_pos=args.num_pos_clicks, n_neg=args.num_neg_clicks, intensity=args.intensity)
        test_dataset = PH2Dataset('test', img_size=args.img_size, n_pos=args.num_pos_clicks, n_neg=args.num_neg_clicks, intensity=args.intensity)
    elif args.dataset == 'drive':
        train_dataset = DRIVEDataset('train', img_size=args.img_size)
        val_dataset = DRIVEDataset('val', img_size=args.img_size)
        test_dataset = DRIVEDataset('test', img_size=args.img_size)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    if args.loss == 'point_supervision':
        loss_fn = PointSupervisionLoss()
    elif args.loss == 'weighted_bce':
        pos_weight = 10 if args.dataset == 'drive' else 3
        loss_fn = BCELoss(pos_weight=pos_weight)
    elif args.loss == 'bce':
        loss_fn = BCELoss()
    elif args.loss == 'focal':
        loss_fn = FocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    if args.network == 'unet':
        network = UNet(
            in_channels=3,
            out_channels=1,
        )
    elif args.network == 'pretrained_unet':
        network = UNet(
            in_channels=3,
            out_channels=1,
            pretrained=True,
        )
    elif args.network == 'cnn':
        network = CNNAutoEncoder(
            in_channels=3,
            out_channels=1,
        )
    else:
        raise ValueError(f"Unknown network: {args.network}")

    model: BaseLightningModule = ClassificationModel(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    datamodule: BaseDM = BaseDM(
        trainset=train_dataset,
        valset=val_dataset,
        testset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    callbacks = [
        VisualizeSegmentationCallback(),
        SegmentationMetricsCallback(),
        LogPointLossCallback(),
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
    
    logger = TensorBoardLogger(
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
        logger=logger,
        max_time="00:11:00:00"  # 11 hours
        )
    
    logger.log_hyperparams(vars(args))
    
    torch.set_float32_matmul_precision('high')
    
    pytorch_lightning.seed_everything(42)
    trainer.fit(model = model, datamodule = datamodule)
    trainer.test(model = model, datamodule = datamodule)