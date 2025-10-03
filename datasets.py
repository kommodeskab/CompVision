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

def get_root_dir(leakage: bool = False) -> str:
    if leakage:
        return "/dtu/datasets1/02516/ucf10"
    else:
        return "/dtu/datasets1/02516/ucf101_noleakage"

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self,
        leakage : bool = False,
        split : Splits = 'train', 
        transform : Optional[T.Compose] = None
    ):  
        root_dir = get_root_dir(leakage)
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.label_to_action = self.df.set_index('label')['action'].to_dict()
        self.num_classes = len(self.label_to_action)
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
            "target": target,
            "label": label
        }

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        leakage : bool = False,
        split : Splits = 'train', 
        transform : Optional[T.Compose] = None,
    ):
        root_dir = get_root_dir(leakage)
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.label_to_action = self.df.set_index('label')['action'].to_dict()
        self.num_classes = len(self.label_to_action)
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
        
        frames = torch.stack(frames)

        return {
            "input": frames,
            "target": target,
            "label": label
        }

    def load_frames(self, frames_dir : str) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames