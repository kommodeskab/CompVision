"""
kopier nedenstående ind i en ny fil og kald den "playground.py"
på den måde kan man lave ændringer som ikke bliver tracket af git
"""
from dataloader import BaseDM
from datasets import Hotdog_NotHotdog
from networks import BaseClassifier
from models import ClassificationModel
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer


if __name__ == "__main__":
    IMAGE_SIZE=8
    BATCH_SIZE=1

    train_dataset = Hotdog_NotHotdog(train=True, image_size=IMAGE_SIZE)
    val_dataset = Hotdog_NotHotdog(train=False, image_size=IMAGE_SIZE)

    optimizer = partial(AdamW)
    lr_scheduler = {
        'scheduler': partial(ReduceLROnPlateau, mode='min', factor=0.1, patience=5),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 1
    }

    network = BaseClassifier(input_size=3*IMAGE_SIZE**2)
    model = ClassificationModel(
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    datamodule = BaseDM(dataset=train_dataset, val_dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    trainer = Trainer(max_epochs=20, accelerator="cpu", log_every_n_steps=10)
    
    trainer.fit(model, datamodule=datamodule)