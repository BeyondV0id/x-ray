# Chest X-ray Disease Detection and Localization using TensorFlow

A machine-learning research repository for detection and anatomical localization of **Pneumonia** (RSNA Pneumonia Detection Dataset) and **Tuberculosis** (TBX11K Dataset) using pure **TensorFlow 2.x / Keras** and **OpenCV**.

---

## ⚠️ IMPORTANT MEDICAL DISCLAIMER

> **"This system is an AI-assisted research tool and is not intended to provide a medical diagnosis. Results should not be used as a substitute for evaluation by a qualified medical professional."**

All disease prediction outputs use standard non-diagnostic phrasing such as *"Possible Pneumonia detected"* or *"Possible Tuberculosis detected"*.

---

## 📌 Features

- **Pure TensorFlow / Keras Stack**: Zero reliance on PyTorch or Ultralytics.
- **Deep Object Detection & Localization**: Custom RetinaNet object detection model with Feature Pyramid Network (FPN) and Focal Loss.
- **Unified Dataset Pipeline**: Robust parsing for RSNA Pneumonia CSV annotations and TBX11K XML/JSON/TXT annotations.
- **Leakage-Free Splitting**: Preserves patient grouping across `train` (70%), `val` (15%), and `test` (15%) splits.
- **Anatomical Localization**: Maps normalized bounding boxes to specific chest regions (e.g. *"Right Lower Lung"*, *"Left Upper Lung"*, *"Bilateral Mid Lung"*).
- **OpenCV Visualization**: Renders bounding boxes, confidence percentages, and lung region labels.
- **FastAPI & Web UI**: REST API server and modern dark-mode glassmorphic dashboard for interactive testing.

---

## 📂 Project Directory Structure

```
CHEST-XRAY-AI/
│
├── config/
│   └── config.yaml               # Central configuration file
│
├── data/
│   ├── raw/                      # Raw dataset storage
│   ├── processed/                # Unified train/val/test json manifests
│   ├── pneumonia/                # RSNA Pneumonia raw images & CSVs
│   └── tuberculosis/             # TBX11K raw images & annotations
│
├── docs/                         # Documentation
├── logs/
│   └── tensorboard/              # TensorBoard log events
│
├── models/
│   ├── checkpoints/              # Epoch checkpoints (.keras)
│   ├── pretrained/               # Pretrained backbone weights
│   └── production/               # Best production model, SavedModel, and TFLite
│
├── notebooks/                    # Research notebooks
│
├── results/
│   ├── validation/               # Validation reports (JSON / TXT)
│   ├── plots/                    # EDA plots (distribution, geometry)
│   ├── predictions/              # Visualized prediction outputs
│   ├── metrics/                  # Evaluation reports (mAP@50, PR curves)
│   └── dataset_samples/          # Annotation sample visualizations
│
├── src/                          # Modular Python source library
│   ├── __init__.py
│   ├── config.py                 # PyYAML config loader
│   ├── dataset.py                # RSNA & TBX11K dataset parsers & tf.data builder
│   ├── model.py                  # RetinaNet architecture, Focal Loss, Smooth L1
│   ├── metrics.py                # mAP@50, mAP@50:95, IoU, Precision & Recall
│   ├── visualization.py          # OpenCV bbox renderer & lung region estimator
│   └── utils.py                  # Logging, seed setter, GPU checker
│
├── scripts/
│   ├── 00_download_models.py     # Downloads/initializes pretrained backbone weights
│   ├── 01_setup_project.py       # Creates project directory structure & checks GPU
│   ├── 02_validate_data.py       # Validates raw RSNA and TBX11K datasets
│   ├── 03_prepare_dataset.py     # Unifies annotations and creates train/val/test splits
│   ├── 04_analyze_dataset.py     # Performs exploratory data analysis (EDA)
│   ├── 05_train.py               # Model training script with Keras callbacks
│   ├── 06_evaluate.py            # Evaluates test metrics (mAP@50, PR curves)
│   ├── 07_inference.py           # Runs inference on image or folder
│   ├── 08_export_model.py        # Exports model to SavedModel and TFLite
│   ├── 09_tensorboard.py         # Launches TensorBoard server
│   ├── check_model.py            # Quick sanity check for model signatures
│   ├── debug_detections.py       # Deep-dive debugging for detection anomalies
│   └── visualize_dataset.py      # Dataset annotation visualizer
│
├── backend/
│   └── app.py                    # FastAPI server for web REST API
│
├── frontend/
│   ├── index.html                # Web dashboard HTML layout
│   ├── style.css                 # Glassmorphic dark mode styles
│   └── app.js                    # Web UI logic & API handler
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Execution Workflow Order

Execute the scripts in the following chronological sequence:

### 1. Project Environment & Directory Setup
```bash
python scripts/01_setup_project.py
```

### 2. Download Pretrained Model Backbone
```bash
python scripts/00_download_models.py
```

### 3. Place Datasets
Download/place the raw dataset files into their respective directories:
- RSNA Pneumonia images & CSV -> `data/pneumonia/`
- TBX11K Tuberculosis images & annotations -> `data/tuberculosis/`

### 4. Validate Dataset Files & Annotations
```bash
python scripts/02_validate_data.py
```

### 5. Prepare Unified Dataset & Splits
```bash
python scripts/03_prepare_dataset.py
```

### 6. Perform Exploratory Data Analysis (EDA)
```bash
python scripts/04_analyze_dataset.py
```

### 7. Visualize Dataset Annotations (Optional Sanity Check)
```bash
python scripts/visualize_dataset.py
```

### 8. Train TensorFlow Object Detection Model
```bash
python scripts/05_train.py --epochs 30 --batch-size 8 --learning-rate 0.0001
```

### 9. Evaluate Test Performance
```bash
python scripts/06_evaluate.py
```

### 10. Run CLI Inference on Chest X-ray
```bash
python scripts/07_inference.py --image path/to/chest_xray.png --confidence 0.50
```
or on an entire directory:
```bash
python scripts/07_inference.py --folder data/processed/
```

### 11. Export Model to SavedModel & TFLite
```bash
python scripts/08_export_model.py
```

### 12. Monitor via TensorBoard
```bash
python scripts/09_tensorboard.py
```

---

## 🛠️ Debugging & Inspection Tools

- **Sanity Check Model Loading**:
  ```bash
  python scripts/check_model.py
  ```
- **Debug Low-Confidence & Overlapping Detections**:
  ```bash
  python scripts/debug_detections.py --confidence 0.30
  ```

---

## 🌐 Web Dashboard & REST API Backend

To run the interactive web app:

1. Launch FastAPI backend:
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Open your web browser at `http://localhost:8000/` or open `frontend/index.html`.
