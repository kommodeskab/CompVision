from torch.utils.data import Dataset
import os
import cv2
import torch
from utils import Data
from typing import Literal
from torchvision.transforms.functional import resize

def train_val_test_split(values: list, fracs: list[float]) -> tuple[list, list, list]:
    assert sum(fracs) == 1.0, "Fractions must sum to 1.0"
    n = len(values)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    train = values[:n_train]
    val = values[n_train:n_train + n_val]
    test = values[n_train + n_val:]
    return train, val, test

def read_img(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
    return img

class PH2Dataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test']):
        super().__init__()
        self.split = split
        self.root = "/dtu/datasets1/02516/PH2_Dataset_images"
        image_names = os.listdir(self.root)
        train, val, test = train_val_test_split(image_names, [0.8, 0.1, 0.1])
        if split == 'train':
            self.image_names = train
        elif split == 'val':
            self.image_names = val
        elif split == 'test':
            self.image_names = test
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Data:
        img_name = self.image_names[idx]
        img_path = f'{self.root}/{img_name}/{img_name}_Dermoscopic_Image/{img_name}.bmp'
        lesion_path = f'{self.root}/{img_name}/{img_name}_lesion/{img_name}_lesion.bmp'
        input = read_img(img_path)
        target = read_img(lesion_path)[0:1, :, :]  # Keep only one channel for mask
        input = resize(input, (572, 765))
        target = resize(target, (572, 765))
        return {
            'input': input,
            'target': target
        }
        
        
class DRIVEDataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test']):
        super().__init__()
        self.split = split
        self.root = "/dtu/datasets1/02516/DRIVE"
        
        if split == 'train':
            self.idxs = list(range(22, 41))
        elif split == 'val':
            self.idxs = [21]
        elif split == 'test':
            self.idxs = list(range(1, 21))
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx: int) -> Data:
        new_idx = self.idxs[idx]
        
        if self.split in ['train', 'val']:
            img_path = f'{self.root}/training/images/{new_idx:02d}_training.tif'
            mask_path = f'{self.root}/training/1st_manual/{new_idx:02d}_manual1.gif'
        else: 
            img_path = f'{self.root}/test/images/{new_idx:02d}_test.tif'
            mask_path = f'{self.root}/test/1st_manual/{new_idx:02d}_manual1.gif'
            
        input = read_img(img_path)
        target = read_img(mask_path)[0:1, :, :]
        input = resize(input, (584, 565))
        target = resize(target, (584, 565))

        return {
            'input': input,
            'target': target
        }