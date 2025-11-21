import numpy as np
from selectivesearch import selective_search
import torch
from torchvision.transforms.functional import resize
from utils import Data
import logging
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET
import cv2
import warnings
import os
from joblib import Memory

# selective search gives warnings on floating point images
# i don't want to look at it anymore
warnings.filterwarnings(
    "ignore",
    message="Applying `local_binary_pattern` to floating-point images may give unexpected results",
)

logger = logging.getLogger(__name__)

def compute_iou(boxA: tuple[int], boxB: tuple[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def eval_of_proposals(proposals: list, true_boxes: list) -> list[float]:
    iou_scores = []
    for prop in proposals:
        max_iou = 0
        max_iou = max(compute_iou(prop, true_box) for true_box in true_boxes)
        iou_scores.append(max_iou)
    return iou_scores

cache_dir = './.cache'
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

@memory.cache
def compute_selective_search(img: np.ndarray) -> list[tuple[int]]:
    _, regions = selective_search(img, scale=400, sigma=0.9, min_size=10)
    proposals = [region['rect'] for region in regions]
    proposals = [p for p in proposals if p[2] > 3 and p[3] > 3]  # filter small boxes
    proposals = [(x, y, x + w, y + h) for (x, y, w, h) in proposals]
    return proposals

class BasePotholeDataset(Dataset):  
    def __init__(self, split: Literal['train', 'val', 'test']):
        """Base dataset for loading an image and all its bounding boxes

        Args:
            split (Literal['train', 'val', 'test']): Which split to use
        """
        
        self.rootdir = Path('/dtu/datasets1/02516/potholes')
        
        if split == 'train':
            self.idxs = list(range(0, 400))
        elif split == 'val':
            self.idxs = list(range(400, 500))
        elif split == 'test':
            self.idxs = list(range(500, 665))

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx: int):
        idx = self.idxs[idx]
        annot_dir = self.rootdir / 'annotations' / f'potholes{idx}.xml'
        image_dir = self.rootdir / 'images' / f'potholes{idx}.png'
                
        tree = ET.parse(annot_dir)
        boxes = tree.findall('.//bndbox')
        bounding_boxes = []
        for box in boxes:
            xmin, ymin = int(box.find('xmin').text), int(box.find('ymin').text)
            xmax, ymax = int(box.find('xmax').text), int(box.find('ymax').text)
            bounding_boxes.append((xmin, ymin, xmax, ymax))
            
        img = cv2.imread(str(image_dir))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            'image': img,
            'bounding_boxes': bounding_boxes
        }

class PotholeDataset(Dataset):
    def __init__(
        self, 
        split: Literal['train', 'val', 'test'],
        iou_threshold: tuple[float] = (0.3, 0.7),
        img_size: int = 64,
        return_metadata: bool = False,
        ):
        self.split = split
        self.base_dataset = BasePotholeDataset(split=split)
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.return_metadata = return_metadata
        self.p_positive = 0.7
    
        if return_metadata:
            logger.warning("Using return_metadata=True will fail when using batch size > 1 in DataLoader.")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Data:
        # load image and bounding boxes from base dataset
        data = self.base_dataset[idx]
        img = data['image']
        bounding_boxes = data['bounding_boxes']
        
        proposals = compute_selective_search(img)
        
        assert len(proposals) > 0, f"No proposals found for image with index {idx}."
        
        # calculate intersection over union (with original bounding boxes) for all proposals 
        ious = torch.tensor(eval_of_proposals(proposals, bounding_boxes))
        k1, k2 = self.iou_threshold
        
        # foreground = potholes, background = no potholes
        background_mask_idxs = torch.nonzero(ious < k1)
        foreground_mask_idxs = torch.nonzero(ious > k2)
        n_background = background_mask_idxs.numel()
        n_foreground = foreground_mask_idxs.numel()
        
        assert n_background + n_foreground > 0, f"No valid proposals found for image with index {idx} with thresholds {self.iou_threshold}."
        
        class_balance = n_foreground / (n_foreground + n_background)
        
        # if there are no background proposals, then sample random foreground and vice versa
        if class_balance == 1.0:
            proposal_idx = foreground_mask_idxs[torch.randint(0, n_foreground, (1,)).item()]
            target = torch.tensor([1.0])
        elif class_balance == 0.0:
            proposal_idx = background_mask_idxs[torch.randint(0, n_background, (1,)).item()]
            target = torch.tensor([0.0])
            
        # if there are actually both foreground and background proposals, sample according to p_positive
        elif torch.rand(1).item() < self.p_positive:
            proposal_idx = foreground_mask_idxs[torch.randint(0, n_foreground, (1,)).item()]
            target = torch.tensor([1.0])
        else:
            proposal_idx = background_mask_idxs[torch.randint(0, n_background, (1,)).item()]
            target = torch.tensor([0.0])
                    
        proposal = proposals[proposal_idx] # (x1, y1, x2, y2)
        img_patch = img[proposal[1]:proposal[3], proposal[0]:proposal[2]]
        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        img_patch = resize(img_patch, [self.img_size, self.img_size]).clip(0.0, 1.0)
                
        output = {
            'input': img_patch,
            'target': target,
            'iou': ious[proposal_idx],
            'proposal': torch.tensor(proposal),
            'class_balance': class_balance,
            'n_potholes': n_foreground,
            'n_background': n_background,
        }
        
        if self.return_metadata:
            # the meta data can have varying sizes and therefore cannot be batched
            # we only return it when explicitly asked for
            output.update({
                'original_image': img,
                'original_bounding_boxes': bounding_boxes,
                'proposals': proposals,
            })
        
        return output