import torch.nn as nn
from utils import Data
import torch
import numpy as np

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch : Data) -> Data:
        raise NotImplementedError

    def __call__(self, batch : Data) -> Data:
        return self.forward(batch)
    
class BCELoss(BaseLoss):
    def __init__(
        self,
        pos_weight: float = 1.0,
        ):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
    def forward(self, batch : Data) -> Data:
        out = batch['output']
        target = batch['target']
        loss = self.criterion(out, target.float())
        return {'loss': loss, 'bce': loss}
    

def iou(boxA, boxB):
    '''Compute IoU between boxes in [x, y, w, h] format.'''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0

def recall(targets, boxes, threshold=0.5):
    '''
    Takes a list of targest [[x,y,w,h], class], and a list of boxes [x,y,w,h],
    and returns the recall, i.e. the percentage of objecs with at least one
    'good' object proposal, as defined by the threshold.
    '''
    p = 0
    for t in targets:
        t_box = t[0]
        i = 0
        one_good = False
        # Run until one good box has been found. If none is found, one_good stays false.
        while not one_good and i != len(boxes):
            one_good = iou(boxes[i], t_box) > threshold
            i += 1
        p += one_good
    return 100 * p / len(targets)
        

def MABO(targets, boxes):
    '''
    Takes a list of targest [[x,y,w,h], class], and a list of boxes [x,y,w,h],
    and returns the MABO.
    '''
    # Get unique classes and their counts
    classes, counts = np.unique([t[1] for t in targets], return_counts=True)
    count_dict = {c: count for c, count in list(zip(classes, counts))}

    # Calculate Average Best Overlap for each class
    abo = {c: 0 for c in classes}
    for i, t in enumerate(targets):
        t_box, c = t
        best_overlap = 0
        for box in boxes:
            best_overlap = max(best_overlap, iou(box, t_box))
        # Average
        abo[c] += best_overlap / count_dict[c]

    # Compute MABO
    mabo = sum([a for a in abo.values()]) / len(classes)
    return mabo