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
import random
from itertools import product
from typing import Dict, List
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
def compute_selective_search(img: np.ndarray, scale=400, sigma=0.9, min_size=1,) -> list[tuple[int]]:
    _, regions = selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
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

def _eval_single_image(
    idx: int,
    base_dataset: BasePotholeDataset,
    scale: int,
    sigma: float,
    min_size: int,
    iou_thr: float,
    resize_for_eval: int | None,
):
    item = base_dataset[idx]
    img = item["image"]
    gt_boxes = item["bounding_boxes"]
    if len(gt_boxes) == 0:
        return 0, 0, 0, 0
    if resize_for_eval is not None:
        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, (resize_for_eval, resize_for_eval))
        sx = resize_for_eval / w_orig
        sy = resize_for_eval / h_orig

        gt_rescaled = []
        for (xmin, ymin, xmax, ymax) in gt_boxes:
            xmin_r = int(xmin * sx)
            xmax_r = int(xmax * sx)
            ymin_r = int(ymin * sy)
            ymax_r = int(ymax * sy)
            gt_rescaled.append((xmin_r, ymin_r, xmax_r, ymax_r))

        img = img_resized
        gt_boxes = gt_rescaled

    total_gt = len(gt_boxes)

    proposals = compute_selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    if len(proposals) == 0:
        return total_gt, 0, 0, 0

    total_props = len(proposals)
    proposal_ious = np.array(eval_of_proposals(proposals, gt_boxes))
    total_foreground = int((proposal_ious >= iou_thr).sum())

    covered_gt = 0
    for gt in gt_boxes:
        best_iou_for_gt = max(compute_iou(prop, gt) for prop in proposals)
        if best_iou_for_gt >= iou_thr:
            covered_gt += 1

    return total_gt, covered_gt, total_props, total_foreground

def evaluate_params_on_subset(
    base_dataset: BasePotholeDataset,
    indices: list[int],
    scale: int,
    sigma: float,
    min_size: int,
    iou_thr: float = 0.7,
    resize_for_eval: int | None = 256,
    n_jobs: int = -1) -> dict[str, float]:

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_eval_single_image)(idx, base_dataset, scale, sigma, min_size, iou_thr, resize_for_eval,)
        for idx in indices)

    total_gt = 0
    total_covered = 0
    total_props = 0
    total_foreground = 0

    for gt, covered, props, foreground in results:
        total_gt += gt
        total_covered += covered
        total_props += props
        total_foreground += foreground

    recall = total_covered / total_gt if total_gt > 0 else 0.0
    fg_ratio = total_foreground / total_props if total_props > 0 else 0.0
    avg_props = total_props / len(indices) if len(indices) > 0 else 0.0

    return {
        "recall": recall,
        "foreground_ratio": fg_ratio,
        "avg_props": avg_props,
        "total_props": total_props,
    }


def optimize_selective_search(split: str = "train",  n_images: int = 5, iou_thr: float = 0.7,):
    base = BasePotholeDataset(split=split)
    rng = random.Random(0)
    all_indices = list(range(len(base)))
    indices = rng.sample(all_indices, n_images) if n_images < len(all_indices) else all_indices

    scale_values = [100, 200, 300, 400, 500, 600, 700, 800]
    sigma_values = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    min_size_values = [1, 2, 3, 4, 5, 10, 20]
    results = []

    for scale, sigma, min_size in product(scale_values, sigma_values, min_size_values):
        metrics = evaluate_params_on_subset(
            base_dataset=base,
            indices=indices,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            iou_thr=iou_thr,
            resize_for_eval=256,
            n_jobs=-1,          
        )
        res = {"scale": scale, "sigma": sigma, "min_size": min_size, **metrics}
        results.append(res)
        # print(
        #     f"s={scale}, sig={sigma}, min={min_size} -> "
        #     f"recall={metrics['recall']:.3f}, "
        #     f"fg_ratio={metrics['foreground_ratio']:.3f}, "
        #     f"avg_props={metrics['avg_props']:.1f}"
        # )

    best = max(results, key=lambda r: (r["recall"], -r["avg_props"]))
    # print(
    #     "\nBest params: "
    #     f"scale={best['scale']}, sigma={best['sigma']}, min_size={best['min_size']}, "
    #     f"recall={best['recall']:.3f}, fg_ratio={best['foreground_ratio']:.3f}, "
    #     f"avg_props={best['avg_props']:.1f}"
    # )

    df = pd.DataFrame(results)
    df_mean_scale_recall = df.groupby("scale")["recall"].mean().reset_index()
    df_mean_sigma_recall = df.groupby("sigma")["recall"].mean().reset_index()
    df_mean_min_size_recall = df.groupby("min_size")["recall"].mean().reset_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_mean_scale_recall, x="scale", y="recall")
    plt.xlabel("Scale")
    plt.ylabel("Mean recall")
    plt.title("Mean Recall per Scale value")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_mean_sigma_recall, x="sigma", y="recall")
    plt.xlabel("Sigma")
    plt.ylabel("Mean recall")
    plt.title("Mean Recall per Sigma value")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_mean_min_size_recall, x="min_size", y="recall")
    plt.xlabel("Min size")
    plt.ylabel("Mean recall")
    plt.title("Mean Recall per Min Size value")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()
    return best, results

class PotholeDataset(Dataset):
    def __init__(
        self, 
        split: Literal['train', 'val', 'test'],
        iou_threshold: tuple[float] = (0.3, 0.7),
        img_size: int = 64,
        return_metadata: bool = False,
        scale=400, 
        sigma=0.9, 
        min_size=1,
        optimize_flag: bool = False,
        n_images: int = 20
    ):
        self.split = split
        self.base_dataset = BasePotholeDataset(split=split)
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.return_metadata = return_metadata
        self.p_positive = 0.7
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.n_images = n_images

        if optimize_flag and split == "train":
            logger.info("Optimizing selective_search hyperparameters on train subset...")
            best, _ = optimize_selective_search(split="train", n_images=self.n_images, iou_thr=self.iou_threshold[1])
            self.scale = best["scale"]
            self.sigma = best["sigma"]
            self.min_size = best["min_size"]
            logger.info(
                f"Using optimized selective_search params: "
                f"scale={self.scale}, sigma={self.sigma}, min_size={self.min_size}"
            )

        if return_metadata:
            logger.warning("Using return_metadata=True will fail when using batch size > 1 in DataLoader.")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Data:
        # load image and bounding boxes from base dataset
        data = self.base_dataset[idx]
        img = data['image']
        bounding_boxes = data['bounding_boxes']
        
        proposals = compute_selective_search(img, self.scale, self.sigma, self.min_size)
        
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