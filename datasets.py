from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from typing import Optional, Literal
from utils import Data
import torch.nn.functional as F

Splits = Literal['train', 'val', 'test']

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir : str = '/work3/ppar/data/ucf101',
        split : Splits = 'train', 
        transform : Optional[T.Compose] = None
    ):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.num_classes = self.df['label'].nunique()
        self.split = split
        self.transform = transform
       
    def __len__(self) -> int:
        return len(self.frame_paths)

    def _get_meta(self, attr: str, value: str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> Data:
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = torch.tensor(video_meta['label'].item())
        target = F.one_hot(label, num_classes=self.num_classes)
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return {
            "input": frame,
            "target": target
        }

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_dir : str = '/work3/ppar/data/ucf101', 
        split : Splits = 'train', 
        transform : Optional[T.Compose] = None,
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.num_classes = self.df['label'].nunique()
        self.split = split
        self.transform = transform
        
        self.n_sampled_frames = 10

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_meta(self, attr : str, value : str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> Data:
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = torch.tensor(video_meta['label'].item())
        target = F.one_hot(label, num_classes=self.num_classes)

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        frames = torch.stack(frames).permute(1, 0, 2, 3)

        return {
            "input": frames,
            "target": target
        }

    def load_frames(self, frames_dir : str) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames