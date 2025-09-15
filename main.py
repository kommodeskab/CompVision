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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ProgressBar,
    ModelSummary,
    Timer,
    ModelCheckpoint,
)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--image_size", type=int, required=True)
    argparser.add_argument("--batch_size", type=int, required=True)
    args = argparser.parse_args()

    IMAGE_SIZE : int = args.image_size
    BATCH_SIZE : int = args.batch_size

    train_dataset = Hotdog_NotHotdog(train=True, image_size=IMAGE_SIZE)
    val_dataset = Hotdog_NotHotdog(train=False, image_size=IMAGE_SIZE)

    optimizer = partial(AdamW)
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.5, patience=5),
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
        ProgressBar(),
        DeviceStatsMonitor(),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=10, mode='min'), 
        LearningRateMonitor(),
        ModelSummary(),
        Timer(),
    ]
    
    trainer = Trainer(
        max_epochs=-1, 
        max_steps=500_000, 
        accelerator="cpu", 
        log_every_n_steps=10, 
        callbacks=callbacks
        )
    
    trainer.fit(model, datamodule=datamodule)