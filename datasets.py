from torch.utils.data import Dataset
import os
import cv2
import torch
import torch.nn.functional as F
from utils import Data
from typing import Literal
from torchvision.transforms.functional import resize
from torchvision.transforms import v2
from torchvision import tv_tensors

class AugmentationWrapper:
    def __init__(self, split: Literal['train', 'validation', 'test'], img_size: int):
        if split == 'train':
            self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=180),
            v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.ToDtype(torch.float32, scale=True), 
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize(size=(img_size, img_size), antialias=True),
            ])
        
    def __call__(self, images: torch.Tensor, masks: torch.Tensor, train: bool = True):
        images = tv_tensors.Image(images.unsqueeze(0))
        masks = tv_tensors.Mask(masks.unsqueeze(0))
        transformed = self.transform({
            'image': images,
            'mask': masks
        })
        return transformed['image'].squeeze(0), transformed['mask'].squeeze(0)

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

def sample_mask(mask_spec, mask, scale, kernel, padding, intensity, n):
    mask_down = F.interpolate(mask_spec.squeeze(0), scale_factor=1/scale)
    convolved = F.conv2d(mask_down, kernel, padding=padding)
    weights = F.interpolate(convolved.unsqueeze(0), size=mask.shape) * mask

    W = weights.squeeze().shape[1]

    weights += abs(weights.min())
    sampled_idxs = torch.multinomial(weights.flatten()**intensity,num_samples=n, replacement=True)
    ys = sampled_idxs // W
    xs = sampled_idxs % W
    # stack to tensor of shape (B, 2)
    coords = torch.stack((ys, xs), dim=1)
    return coords

def clicks(mask, n_pos=1, n_neg=1, intensity=8, kernel_size = 21):
    kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    padding = kernel_size // 2

    mask_pos = mask.unsqueeze(0).unsqueeze(0)
    mask_neg = 1 - mask_pos
    scale = 16

    coords_pos = sample_mask(mask_pos, mask, scale, kernel, padding, intensity, n_pos)
    coords_neg = sample_mask(mask_neg, 1-mask, scale, kernel, padding, intensity*0.5, n_neg)
    return coords_pos, coords_neg
    

class PH2Dataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test'], img_size: int, n_pos=1, n_neg=1, intensity: int = 8):
        super().__init__()
        self.split = split
        self.root = "/dtu/datasets1/02516/PH2_Dataset_images"
        image_names = os.listdir(self.root)
        train, val, test = train_val_test_split(image_names, [0.8, 0.1, 0.1])
        self.n_pos, self.n_neg = n_pos, n_neg
        self.intensity = intensity
        self.augmenter = AugmentationWrapper(split=split, img_size=img_size)
        
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
        input, target = self.augmenter(input, target)
        
        pos_clicks, neg_clicks = clicks(target, n_pos=self.n_pos, n_neg=self.n_neg, intensity=self.intensity)
            
        return {
            'input': input,
            'target': target,
            'pos_clicks': pos_clicks,
            'neg_clicks': neg_clicks
        }
        
        
class DRIVEDataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test'], img_size: int):
        super().__init__()
        self.split = split
        self.root = "/dtu/datasets1/02516/DRIVE"
        self.augmenter = AugmentationWrapper(split=split, img_size=img_size)
        
        if split == 'train':
            self.idxs = list(range(25, 41))
        elif split == 'val':
            self.idxs = [23, 24]
        elif split == 'test':
            self.idxs = [21, 22]
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx: int) -> Data:
        new_idx = self.idxs[idx]
        
        img_path = f'{self.root}/training/images/{new_idx:02d}_training.tif'
        mask_path = f'{self.root}/training/1st_manual/{new_idx:02d}_manual1.gif'
            
        input = read_img(img_path)
        target = read_img(mask_path)[0:1, :, :]
        input, target = self.augmenter(input, target)

        return {
            'input': input,
            'target': target
        }