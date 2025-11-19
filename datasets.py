import cv2
import selectivesearch

class proposal_dataset:
    def __init__(self):
        pass

    def SelectiveSearch(image, scale=450, sigma=0.9, min_size=250):
        img_lbl, regions = selectivesearch.selective_search(image, scale=scale, sigma=sigma, min_size=min_size)
        regions = [ [r['rect'][0], r['rect'][1], r['rect'][0] + r['rect'][2], r['rect'][1] + r['rect'][3]] for r in regions ]
        return regions

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def eval_of_proposals(proposals:list, true_boxes:list):
        iou_scores = []
        for prop in proposals:
            max_iou = 0
            for true_box in true_boxes:
                current_iou = iou(prop, true_box)
                if current_iou > max_iou:
                    max_iou = current_iou
            iou_scores.append(max_iou)
        return iou_scores