import torch.nn as nn
from utils import Data
import torch

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
        out = batch['out']
        target = batch['target']
        loss = self.criterion(out, target.float())
        return {
            'loss': loss,
            **batch
        }

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

def NMS(boxes, iou_threshold=0.2):
    '''
    Takes a list of boxes with classes and confidence scores [[x,y,w,h], class, confidence], and returns
    the boxes after class specific non-max suppression.
    '''
    # Group boxes by class
    boxes_by_class = {}
    for entry in boxes:
        cls = entry[1]
        boxes_by_class.setdefault(cls, []).append(entry)

    final_boxes = []
    # Perform NMS per class
    for cls, cls_boxes in boxes_by_class.items():
        # Sort by confidence descending
        cls_boxes = sorted(cls_boxes, key=lambda b: b[2], reverse=True)
        
        selected = []

        while cls_boxes:
            best = cls_boxes.pop(0)
            selected.append(best)
            cls_boxes = [b for b in cls_boxes if iou(best[0], b[0]) < iou_threshold]

        final_boxes.extend(selected)

    return final_boxes

def AP(recalls, precision):
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    AP = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recalls >= t].max() if np.any(recalls >= t) else 0
        AP += p / 11
    return AP

def mAP(boxes_conf, targets, iou_threshold=0.5):
    # Group boxes by class
    boxes_by_class = {}
    for entry in boxes_conf:
        cls = entry[1]
        boxes_by_class.setdefault(cls, []).append(entry)
    
    targets_by_class = {}
    for entry in targets:
        cls = entry[1]
        targets_by_class.setdefault(cls, []).append({"box": entry[0], "detected": False})
    
    mAP_list = []

    for cls, gt_boxes in targets_by_class.items():

        proposals = boxes_by_class.get(cls, [])
        proposals = sorted(proposals, key=lambda b: b[2], reverse=True)

        tp = []
        fp = []

        for prop in proposals:
            best_iou = 0
            best_gt   = None

            # find matching GT box
            for gt in gt_boxes:
                iou_val = iou(prop[0], gt["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt  = gt
            
            if best_iou >= iou_threshold and best_gt is not None and not best_gt["detected"]:
                tp.append(1)
                fp.append(0)
                best_gt["detected"] = True
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp)
        fp = np.array(fp)

        if len(tp) == 0:
            mAP_list.append(0)
            continue

        TP_cumulative = tp.cumsum()
        FP_cumulative = fp.cumsum()

        recalls = TP_cumulative / len(gt_boxes)
        precision = TP_cumulative / (TP_cumulative + FP_cumulative)

        AP_class = AP(recalls, precision)
        mAP_list.append(AP_class)

    return np.mean(mAP_list)
