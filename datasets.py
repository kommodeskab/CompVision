from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from typing import Optional
from models import Data

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir : str = '/work3/ppar/data/ucf101',
        split : str = 'train', 
        transform : Optional[T.Compose] = None
    ):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
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
        label = video_meta['label']
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return {
            "input": frame,
            "target": label
        }

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_dir : str = '/work3/ppar/data/ucf101', 
        split : str = 'train', 
        transform : Optional[T.Compose] = None,
        stack_frames : bool = True
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_meta(self, attr : str, value : str) -> pd.DataFrame:
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx: int) -> Data:
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label']

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return {
            "input": frames,
            "target": label
        }

    def load_frames(self, frames_dir : str) -> list[Image.Image]:
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/work3/ppar/data/ucf101'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]