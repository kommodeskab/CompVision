import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import os
from utils import Data, DatasetType, DataLoaderType

def split_dataset(train_dataset : DatasetType, val_dataset : DatasetType | None, train_val_split : float) -> tuple[DatasetType, DatasetType]:
    if val_dataset is None:
        return random_split(train_dataset, [train_val_split, 1 - train_val_split])
    else:
        return train_dataset, val_dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset : Dataset,
        val_dataset : Dataset | None = None,
        train_val_split : float = 0.95,
        **kwargs
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        self.original_dataset = dataset
        self.train_dataset, self.val_dataset = split_dataset(dataset, val_dataset, train_val_split)
        self.num_workers = kwargs.pop("num_workers", os.cpu_count())
        print(f"Using {self.num_workers} workers for data loading.")
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoaderType:
        return DataLoader(
            dataset = self.train_dataset, 
            shuffle = True, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )

    def val_dataloader(self) -> DataLoaderType:
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )