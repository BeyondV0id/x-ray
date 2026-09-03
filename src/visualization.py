import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from src.utils import MEDICAL_DISCLAIMER

# Color palette (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 140, 255),   # Pneumonia: Vibrant Orange
    1: (118, 214, 0),   # Tuberculosis: Emerald Green
}

CLASS_NAMES = {
    0: "Pneumonia",
    1: "Tuberculosis"
}

def estimate_lung_region(bbox: Tuple[float, float, float, float]) -> str:
    """
    Estimate the anatomical lung zone from normalized bounding box coordinates [ymin, xmin, ymax, xmax].
    
    Medical convention: Patient's Right is viewer's Left (xmin < 0.5), 
                        Patient's Left is viewer's Right (xmin >= 0.5).
    """
    ymin, xmin, ymax, xmax = bbox
    y_center = (ymin + ymax) / 2.0
    x_center = (xmin + xmax) / 2.0
    box_width = xmax - xmin
    
    # Determine lateral side (Right/Left/Bilateral)
    if box_width > 0.45 or (xmin < 0.38 and xmax > 0.62):
        side = "Bilateral"
    elif x_center < 0.5:
        side = "Right"
    else:
        side = "Left"
        
    # Determine vertical elevation (Upper/Mid/Lower)
    if y_center < 0.35:
        vertical = "Upper Lung"
    elif y_center < 0.65:
        vertical = "Mid Lung"
    else:
        vertical = "Lower Lung"
        
    if side == "Bilateral":
        return f"Bilateral {vertical}"
    else:
        return f"{side} {vertical}"

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    confidence_threshold: float = 0.5,
    class_names: Dict[int, str] = None,
    draw_disclaimer: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes, disease labels, confidence scores, and lung regions using OpenCV.
    
    Args:
        image (np.ndarray): Input RGB image array (H, W, 3).
        boxes (np.ndarray): Normalized or absolute boxes [[ymin, xmin, ymax, xmax], ...].
        scores (np.ndarray): Confidence scores for each box.
        classes (np.ndarray): Class indices for each box.
        confidence_threshold (float): Minimum confidence threshold.
        class_names (Dict[int, str]): Mapping of class indices to names.
        draw_disclaimer (bool): Whether to overlay the medical disclaimer banner.
        
    Returns:
        np.ndarray: Image with annotated bounding boxes and labels.
    """
    if class_names is None:
        class_names = CLASS_NAMES
        
    img = image.copy()
    h, w = img.shape[:2]
    
    # Convert BGR if OpenCV format or keep RGB
    # We assume img is RGB coming in, so for OpenCV drawing we convert to BGR for cv2 calls if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
            
        box = boxes[i]
        cls_id = int(classes[i])
        disease_name = class_names.get(cls_id, f"Class {cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 0, 0))
        
        # Check normalized vs absolute coordinates
        if np.max(box) <= 1.0 + 1e-5:
            ymin, xmin, ymax, xmax = box
            norm_box = (ymin, xmin, ymax, xmax)
            pt1 = (int(xmin * w), int(ymin * h))
            pt2 = (int(xmax * w), int(ymax * h))
        else:
            ymin, xmin, ymax, xmax = box
            norm_box = (ymin / h, xmin / w, ymax / h, xmax / w)
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            
        lung_region = estimate_lung_region(norm_box)
        
        # Draw bounding box
        cv2.rectangle(img, pt1, pt2, color, thickness=3)
        
        # Label string phrasing
        label_text = f"Possible {disease_name} | {score*100:.1f}%"
        sub_text = f"Location: {lung_region}"
        
        # Draw text background card
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, w / 1000.0)
        thickness = 1 if font_scale < 0.7 else 2
        
        (tw1, th1), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, font_scale - 0.1, thickness)
        
        card_w = max(tw1, tw2) + 16
        card_h = th1 + th2 + 18
        
        card_x1 = pt1[0]
        card_y1 = max(0, pt1[1] - card_h)
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        # Draw background rectangle for header label
        cv2.rectangle(img, (card_x1, card_y1), (card_x2, card_y2), color, -1)
        
        # Text overlay
        cv2.putText(img, label_text, (card_x1 + 8, card_y1 + th1 + 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(img, sub_text, (card_x1 + 8, card_y1 + th1 + th2 + 10), font, font_scale - 0.1, (240, 240, 240), 1, cv2.LINE_AA)

    # Optional medical disclaimer banner at the bottom
    if draw_disclaimer:
        banner_h = int(h * 0.05)
        banner_h = max(35, banner_h)
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        
        disclaimer_short = "AI-assisted research tool. Not intended for direct medical diagnosis."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, w / 1200.0)
        (tw, th), _ = cv2.getTextSize(disclaimer_short, font, font_scale, 1)
        
        tx = (w - tw) // 2
        ty = (banner_h + th) // 2
        cv2.putText(banner, disclaimer_short, (tx, ty), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        img = np.vstack([img, banner])

    # Convert back to RGB for matplotlib/PIL/FastAPI return
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
