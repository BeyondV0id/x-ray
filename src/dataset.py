import os
import json
import glob
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
# tensorflow is imported lazily inside build_tf_dataset to allow lightweight dataset validation without TF dependency
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils import setup_logger

logger = setup_logger("dataset")

def load_image_as_rgb(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an X-ray image (DICOM/PNG/JPG) using OpenCV/PIL and return RGB numpy array.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img = None
    ext = path_obj.suffix.lower()
    
    if ext == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(image_path)
            img = dcm.pixel_array.astype(np.float32)
            if img.max() > 0:
                img = ((img / img.max()) * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            logger.warning(f"pydicom reading failed for {image_path}: {e}, falling back to cv2.")
            
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
    return img

def prepare_rsna_dataset(pneumonia_dir: str) -> List[Dict[str, Any]]:
    """
    Parse RSNA Pneumonia Detection Dataset annotations.
    RSNA CSV typically has columns: patientId, x, y, width, height, Target
    
    Returns list of dicts:
      [
        {
          "image_id": "patientId",
          "image_path": ".../patientId.dcm or .png",
          "patient_id": "patientId",
          "class_id": 0,
          "boxes": [[ymin, xmin, ymax, xmax], ...],  # Normalized 0.0 to 1.0
          "width": 1024,
          "height": 1024
        }, ...
      ]
    """
    pneumonia_path = Path(pneumonia_dir)
    csv_candidates = list(pneumonia_path.glob("*.csv"))
    
    if not csv_candidates:
        logger.warning(f"No CSV file found in {pneumonia_dir}")
        return []
        
    csv_file = csv_candidates[0]
    for c in csv_candidates:
        if "labels" in c.name.lower() or "stage" in c.name.lower():
            csv_file = c
            break
            
    logger.info(f"Reading RSNA annotations from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Locate images directory or images files
    image_files = {}
    for ext in ["*.dcm", "*.png", "*.jpg", "*.jpeg"]:
        for p in pneumonia_path.rglob(ext):
            image_files[p.stem] = str(p)
            
    grouped = {}
    for _, row in df.iterrows():
        pid = str(row["patientId"])
        target = int(row.get("Target", 1 if pd.notnull(row.get("x")) else 0))
        
        if pid not in grouped:
            img_path = image_files.get(pid, str(pneumonia_path / f"{pid}.dcm"))
            grouped[pid] = {
                "image_id": pid,
                "image_path": img_path,
                "patient_id": pid,
                "class_id": 0,  # 0 = Pneumonia
                "raw_boxes": [],
                "target": target
            }
            
        if target == 1 and pd.notnull(row.get("x")):
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
            grouped[pid]["raw_boxes"].append([x, y, w, h])
            
    # Process into normalized bounding boxes
    dataset_records = []
    for pid, data in grouped.items():
        img_path = data["image_path"]
        img_w, img_h = 1024.0, 1024.0  # Standard RSNA resolution default
        
        if Path(img_path).exists():
            try:
                # Read image size quickly if image exists
                test_img = cv2.imread(img_path)
                if test_img is not None:
                    img_h, img_w = test_img.shape[:2]
            except Exception:
                pass
                
        normalized_boxes = []
        for (x, y, w, h) in data["raw_boxes"]:
            ymin = max(0.0, min(1.0, y / img_h))
            xmin = max(0.0, min(1.0, x / img_w))
            ymax = max(0.0, min(1.0, (y + h) / img_h))
            xmax = max(0.0, min(1.0, (x + w) / img_w))
            if ymax > ymin and xmax > xmin:
                normalized_boxes.append([ymin, xmin, ymax, xmax])
                
        dataset_records.append({
            "image_id": pid,
            "image_path": img_path,
            "patient_id": pid,
            "class_id": 0,  # Pneumonia
            "boxes": normalized_boxes,
            "width": int(img_w),
            "height": int(img_h),
            "source": "rsna_pneumonia"
        })
        
    logger.info(f"Loaded {len(dataset_records)} RSNA records.")
    return dataset_records

def prepare_tbx11k_dataset(tuberculosis_dir: str) -> List[Dict[str, Any]]:
    """
    Parse TBX11K Tuberculosis Dataset annotations (XML/JSON/TXT/CSV).
    
    Class ID 1 = Tuberculosis.
    """
    tb_path = Path(tuberculosis_dir)
    image_files = {}
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in tb_path.rglob(ext):
            image_files[p.stem] = str(p)
            
    dataset_records = []
    
    # 1. Look for XML annotations (Pascal VOC format)
    xml_files = list(tb_path.rglob("*.xml"))
    if xml_files:
        logger.info(f"Found {len(xml_files)} XML annotation files for TBX11K.")
        for xml_f in xml_files:
            stem = xml_f.stem
            img_path = image_files.get(stem, str(tb_path / f"{stem}.png"))
            
            tree = ET.parse(xml_f)
            root = tree.getroot()
            
            size_elem = root.find("size")
            img_w = float(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 1024.0
            img_h = float(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 1024.0
            
            boxes = []
            for obj in root.findall("object"):
                bnd = obj.find("bndbox")
                if bnd is not None:
                    xmin = float(bnd.find("xmin").text)
                    ymin = float(bnd.find("ymin").text)
                    xmax = float(bnd.find("xmax").text)
                    ymax = float(bnd.find("ymax").text)
                    
                    norm_ymin = max(0.0, min(1.0, ymin / img_h))
                    norm_xmin = max(0.0, min(1.0, xmin / img_w))
                    norm_ymax = max(0.0, min(1.0, ymax / img_h))
                    norm_xmax = max(0.0, min(1.0, xmax / img_w))
                    if norm_ymax > norm_ymin and norm_xmax > norm_xmin:
                        boxes.append([norm_ymin, norm_xmin, norm_ymax, norm_xmax])
                        
            dataset_records.append({
                "image_id": stem,
                "image_path": img_path,
                "patient_id": stem.split("_")[0],
                "class_id": 1,  # Tuberculosis
                "boxes": boxes,
                "width": int(img_w),
                "height": int(img_h),
                "source": "tbx11k"
            })
            
    # 2. Look for JSON annotations if XML not found or additional
    elif list(tb_path.rglob("*.json")):
        json_files = list(tb_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON annotation files for TBX11K.")
        for j_file in json_files:
            try:
                with open(j_file, "r") as f:
                    data = json.load(f)
                    
                # Standard COCO or custom JSON format
                if isinstance(data, dict) and "images" in data and "annotations" in data:
                    img_dict = {img["id"]: img for img in data["images"]}
                    ann_grouped = {}
                    for ann in data["annotations"]:
                        img_id = ann["image_id"]
                        if img_id not in ann_grouped:
                            ann_grouped[img_id] = []
                        ann_grouped[img_id].append(ann["bbox"])
                        
                    for img_id, img_info in img_dict.items():
                        fname = Path(img_info["file_name"]).stem
                        img_w = float(img_info.get("width", 1024))
                        img_h = float(img_info.get("height", 1024))
                        img_path = image_files.get(fname, str(tb_path / img_info["file_name"]))
                        
                        raw_boxes = ann_grouped.get(img_id, [])
                        norm_boxes = []
                        for b in raw_boxes:
                            # COCO format: [x, y, w, h]
                            x, y, w, h = b
                            ymin = max(0.0, min(1.0, y / img_h))
                            xmin = max(0.0, min(1.0, x / img_w))
                            ymax = max(0.0, min(1.0, (y + h) / img_h))
                            xmax = max(0.0, min(1.0, (x + w) / img_w))
                            if ymax > ymin and xmax > xmin:
                                norm_boxes.append([ymin, xmin, ymax, xmax])
                                
                        dataset_records.append({
                            "image_id": fname,
                            "image_path": img_path,
                            "patient_id": fname.split("_")[0],
                            "class_id": 1,
                            "boxes": norm_boxes,
                            "width": int(img_w),
                            "height": int(img_h),
                            "source": "tbx11k"
                        })
            except Exception as e:
                logger.error(f"Error parsing JSON annotation file {j_file}: {e}")
                
    # 3. Fallback: TXT or CSV files in TB folder
    else:
        logger.info("Scanning for image files directly in TBX11K directory...")
        for stem, img_path in image_files.items():
            dataset_records.append({
                "image_id": stem,
                "image_path": img_path,
                "patient_id": stem.split("_")[0],
                "class_id": 1,  # Tuberculosis
                "boxes": [],
                "width": 1024,
                "height": 1024,
                "source": "tbx11k"
            })
            
    logger.info(f"Loaded {len(dataset_records)} TBX11K records.")
    return dataset_records

def create_split_datasets(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset records into Train, Validation, and Test splits without patient data leakage.
    """
    if not records:
        return [], [], []
        
    patients = np.array([r["patient_id"] for r in records])
    
    # First split: Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, val_test_idx = next(gss1.split(records, groups=patients))
    
    train_records = [records[i] for i in train_idx]
    val_test_records = [records[i] for i in val_test_idx]
    val_test_patients = patients[val_test_idx]
    
    # Second split: Val vs Test
    relative_test_ratio = test_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=seed)
    val_idx, test_idx = next(gss2.split(val_test_records, groups=val_test_patients))
    
    val_records = [val_test_records[i] for i in val_idx]
    test_records = [val_test_records[i] for i in test_idx]
    
    logger.info(f"Dataset split complete - Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    return train_records, val_records, test_records

def build_tf_dataset(
    records: List[Dict[str, Any]],
    image_size: int = 512,
    batch_size: int = 8,
    is_training: bool = True,
    max_boxes_per_image: int = 100
):
    """
    Build a tf.data.Dataset generator for object detection training/validation.
    """
    import tensorflow as tf
    def generator():
        for rec in records:
            img_path = rec["image_path"]
            boxes = rec.get("boxes", [])
            cls_id = rec.get("class_id", 0)
            
            # Load image
            try:
                img = load_image_as_rgb(img_path, target_size=(image_size, image_size))
                img = img.astype(np.float32) / 255.0
            except Exception:
                img = np.zeros((image_size, image_size, 3), dtype=np.float32)
                
            # Pad boxes to fixed shape (max_boxes_per_image, 4)
            padded_boxes = np.zeros((max_boxes_per_image, 4), dtype=np.float32)
            padded_classes = np.full((max_boxes_per_image,), -1, dtype=np.int32)
            
            n = min(len(boxes), max_boxes_per_image)
            if n > 0:
                padded_boxes[:n] = np.array(boxes[:n], dtype=np.float32)
                padded_classes[:n] = cls_id
                
            yield img, padded_boxes, padded_classes, n

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(max_boxes_per_image, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(max_boxes_per_image,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(records) + 1, 1000))
        
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
