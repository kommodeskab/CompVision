from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from typing import Literal
from utils import Data
import torch.nn.functional as F
import re
import numpy as np
from torchvision.transforms import v2

IMG_SIZE = 128

train_transform = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    v2.RandomVerticalFlip(p=0.05),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToTensor(),
])

optical_flow_transform = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    v2.RandomVerticalFlip(p=0.05),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ToTensor(),
])

test_transform = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.ToTensor(),
])

Splits = Literal['train', 'val', 'test']

def extract_numbers(filepath : str) -> int:
    match = re.search(r'flow_(\d+)_(\d+)\.npy', filepath)
    if match:
        return int(match.group(1))

def get_root_dir(leakage: bool = False) -> str:
    if leakage:
        return "/dtu/datasets1/02516/ufc10"
    else:
        return "/dtu/datasets1/02516/ucf101_noleakage"

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self,
        leakage : bool = False,
        split : Splits = 'train'
    ):  
        self.root_dir = get_root_dir(leakage)
        self.frame_paths = sorted(glob(f'{self.root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{self.root_dir}/metadata/{split}.csv')
        self.label_to_action = self.df.set_index('label')['action'].to_dict()
        self.num_classes = len(self.label_to_action)
        self.split = split
        self.transform = train_transform if split == 'train' else test_transform
       
    def __len__(self) -> int:
        return len(self.frame_paths)

    def _get_meta(self, attr: str, value: str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> Data:
        frame_path = self.frame_paths[idx]
        frame_path = frame_path.replace('\\','/')
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = torch.tensor(video_meta['label'].item())
        target = F.one_hot(label, num_classes=self.num_classes)
        
        frame = Image.open(frame_path).convert("RGB")
        frame = self.transform(frame)

        return {
            "input": frame,
            "target": target,
            "label": label
        }

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        leakage : bool = False,
        split : Splits = 'train'
    ):
        self.leakage = leakage
        self.root_dir = get_root_dir(leakage)
        self.video_paths = sorted(glob(f'{self.root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{self.root_dir}/metadata/{split}.csv')
        self.label_to_action = self.df.set_index('label')['action'].to_dict()
        self.num_classes = len(self.label_to_action)
        self.split = split
        self.transform = train_transform if split == 'train' else test_transform
        self.optical_flow_transform = optical_flow_transform if split == 'train' else test_transform

        self.n_sampled_frames = 10

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_meta(self, attr : str, value : str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> Data:
        video_path = self.video_paths[idx]
        video_path = video_path.replace('\\','/')
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = torch.tensor(video_meta['label'].item())
        target = F.one_hot(label, num_classes=self.num_classes)

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)
        frames = self.transform(video_frames)
        
        frames = torch.stack(frames)
        
        # also load optical flows
        if not self.leakage:
            flow_path = video_meta.video_path.item()[:-4]
            flow_path = f"{self.root_dir}/flows/{self.split}/{flow_path}"
            flow_files = glob(f"{flow_path}/*")
            flow_files = sorted(flow_files, key=extract_numbers)
            flows = np.stack([np.load(f) for f in flow_files])
            flows = torch.from_numpy(flows)
            flows = self.optical_flow_transform(flows)
        else:
            # there are no optical flows for the dataset with leakage
            flows = False

        return {
            "input": frames,
            "target": target,
            "label": label,
            "optical_flow": flows
        }

    def load_frames(self, frames_dir : str) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames