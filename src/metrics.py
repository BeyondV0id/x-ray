import numpy as np
from typing import List, Dict, Tuple, Any

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes [ymin, xmin, ymax, xmax].
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)
    
    inter_area = max(0.0, inter_ymax - inter_ymin) * max(0.0, inter_xmax - inter_xmin)
    area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
    area2 = (ymax2 - ymin2) * (xmax2 - xmin2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def evaluate_detections_for_class(
    pred_boxes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    gt_boxes_list: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate detections for a single class across all images in test set.
    
    Returns:
        Dict containing precision, recall, ap (Average Precision), fp_count, fn_count, tp_count.
    """
    total_gt = sum(len(b) for b in gt_boxes_list)
    if total_gt == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": sum(len(b) for b in pred_boxes_list),
            "fn": 0
        }
        
    # Gather all predictions with image index
    all_preds = []
    for img_idx in range(len(pred_boxes_list)):
        p_boxes = pred_boxes_list[img_idx]
        p_scores = pred_scores_list[img_idx]
        for i in range(len(p_scores)):
            all_preds.append((p_scores[i], img_idx, p_boxes[i]))
            
    # Sort predictions by score descending
    all_preds.sort(key=lambda x: x[0], reverse=True)
    
    gt_detected = [np.zeros(len(gt_boxes_list[i]), dtype=bool) for i in range(len(gt_boxes_list))]
    
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    
    for i, (score, img_idx, p_box) in enumerate(all_preds):
        gt_boxes = gt_boxes_list[img_idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx, g_box in enumerate(gt_boxes):
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_iou >= iou_threshold:
            if not gt_detected[img_idx][best_gt_idx]:
                tp[i] = 1.0
                gt_detected[img_idx][best_gt_idx] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0
            
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
    recalls = cum_tp / total_gt
    
    # Compute Average Precision (11-point interpolation or area under curve)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask]) / 11.0
            
    final_tp = int(np.sum(tp))
    final_fp = int(np.sum(fp))
    final_fn = total_gt - final_tp
    
    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
    
    return {
        "ap": float(ap),
        "precision": final_precision,
        "recall": final_recall,
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
        "precisions": precisions.tolist() if len(precisions) > 0 else [],
        "recalls": recalls.tolist() if len(recalls) > 0 else []
    }

def calculate_map_metrics(
    all_pred_boxes: List[List[np.ndarray]],
    all_pred_scores: List[List[np.ndarray]],
    all_pred_classes: List[List[np.ndarray]],
    all_gt_boxes: List[List[np.ndarray]],
    all_gt_classes: List[List[np.ndarray]],
    num_classes: int = 2,
    class_names: Dict[int, str] = None
) -> Dict[str, Any]:
    """
    Calculate mAP@50 and mAP@50:95 overall and per-class.
    """
    if class_names is None:
        class_names = {0: "Pneumonia", 1: "Tuberculosis"}
        
    num_images = len(all_gt_boxes)
    results_per_class = {}
    
    map50_list = []
    
    for c in range(num_classes):
        c_name = class_names.get(c, f"Class_{c}")
        
        # Filter predictions and GT for class c
        c_pred_boxes = []
        c_pred_scores = []
        c_gt_boxes = []
        
        for i in range(num_images):
            # Predictions
            p_cls = np.array(all_pred_classes[i]) if len(all_pred_classes[i]) > 0 else np.array([])
            p_box = np.array(all_pred_boxes[i]) if len(all_pred_boxes[i]) > 0 else np.zeros((0, 4))
            p_scr = np.array(all_pred_scores[i]) if len(all_pred_scores[i]) > 0 else np.array([])
            
            mask_p = (p_cls == c)
            c_pred_boxes.append(p_box[mask_p])
            c_pred_scores.append(p_scr[mask_p])
            
            # Ground truth
            g_cls = np.array(all_gt_classes[i]) if len(all_gt_classes[i]) > 0 else np.array([])
            g_box = np.array(all_gt_boxes[i]) if len(all_gt_boxes[i]) > 0 else np.zeros((0, 4))
            
            mask_g = (g_cls == c)
            c_gt_boxes.append(g_box[mask_g])
            
        eval_50 = evaluate_detections_for_class(c_pred_boxes, c_pred_scores, c_gt_boxes, iou_threshold=0.5)
        
        # mAP@50:95 calculation
        aps_50_95 = []
        for iou_thresh in np.arange(0.50, 1.00, 0.05):
            res_iou = evaluate_detections_for_class(c_pred_boxes, c_pred_scores, c_gt_boxes, iou_threshold=iou_thresh)
            aps_50_95.append(res_iou["ap"])
            
        ap50_95_val = float(np.mean(aps_50_95))
        
        results_per_class[c_name] = {
            "mAP_50": eval_50["ap"],
            "mAP_50_95": ap50_95_val,
            "precision": eval_50["precision"],
            "recall": eval_50["recall"],
            "true_positives": eval_50["tp"],
            "false_positives": eval_50["fp"],
            "false_negatives": eval_50["fn"],
            "precisions_curve": eval_50["precisions"],
            "recalls_curve": eval_50["recalls"]
        }
        map50_list.append(eval_50["ap"])
        
    overall_map50 = float(np.mean(map50_list)) if map50_list else 0.0
    overall_map50_95 = float(np.mean([v["mAP_50_95"] for v in results_per_class.values()])) if results_per_class else 0.0
    
    return {
        "overall_mAP_50": overall_map50,
        "overall_mAP_50_95": overall_map50_95,
        "per_class": results_per_class
    }
