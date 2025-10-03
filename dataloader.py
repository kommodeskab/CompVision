import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import os
from utils import DatasetType, DataLoaderType
from typing import Optional

def split_dataset(
    train_dataset : DatasetType, 
    val_dataset : Optional[DatasetType] = None, 
    train_val_split : Optional[float] = None,
    ) -> tuple[DatasetType, DatasetType]:
    if val_dataset is None:
        return random_split(train_dataset, [train_val_split, 1 - train_val_split])
    else:
        return train_dataset, val_dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        trainset : DatasetType,
        valset : Optional[DatasetType] = None,
        testset : Optional[DatasetType] = None,
        train_val_split : Optional[float] = None,
        **kwargs
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["trainset", "valset", "testset"])
        self.original_dataset = trainset
        self.trainset, self.valset = split_dataset(trainset, valset, train_val_split)
        self.testset = testset
        self.num_workers = kwargs.pop("num_workers", os.cpu_count())
        print(f"Using {self.num_workers} workers for data loading.")
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoaderType:
        return DataLoader(
            dataset = self.trainset, 
            shuffle = True, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )

    def val_dataloader(self) -> DataLoaderType:
        return DataLoader(
            dataset = self.valset, 
            shuffle = False, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )

    def test_dataloader(self) -> DataLoaderType:
        return DataLoader(
            dataset = self.testset, 
            shuffle = False, 
            drop_last=True,
            num_workers=self.num_workers,
            **self.kwargs,
            )