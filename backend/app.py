import sys
import os
import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, MEDICAL_DISCLAIMER, check_gpu_status
from src.model import build_retinanet_model, decode_predictions, FocalLoss, SmoothL1Loss
from src.visualization import draw_detections, estimate_lung_region

logger = setup_logger("fastapi_backend")

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray Disease Detection & Localization API",
    description="TensorFlow-powered AI web backend for localization of Pneumonia and Tuberculosis.",
    version="1.0.0"
)

# Enable CORS for frontend web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
CONFIG = load_config()
MODEL = None

@app.on_event("startup")
def load_model_on_startup():
    global MODEL
    logger.info("Initializing TensorFlow model for FastAPI backend...")
    paths = CONFIG["paths"]
    model_path = Path(paths["models_production"]) / "best_model.keras"

    if model_path.exists():
        logger.info(f"Loading production model from: {model_path}")
        MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )
    else:
        logger.warning(f"Production model not found at {model_path}. Instantiating fresh model layout.")
        MODEL = build_retinanet_model(
            input_shape=(CONFIG["model"]["image_size"], CONFIG["model"]["image_size"], 3),
            num_classes=CONFIG["num_classes"]
        )

@app.get("/api/health")
def health_check():
    gpu_info = check_gpu_status(logger=logger)
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "gpu_acceleration": gpu_info["gpu_available"],
        "disclaimer": MEDICAL_DISCLAIMER
    }

@app.get("/api/config")
def get_configuration():
    return {
        "classes": CONFIG["classes"],
        "num_classes": CONFIG["num_classes"],
        "image_size": CONFIG["model"]["image_size"],
        "confidence_threshold": CONFIG["inference"]["confidence_threshold"],
        "disclaimer": MEDICAL_DISCLAIMER
    }

@app.post("/api/detect")
async def detect_xray(
    file: UploadFile = File(...),
    confidence: float = Query(None, description="Optional custom confidence threshold (0.0 to 1.0)")
):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    if confidence is None:
        confidence = CONFIG["inference"]["confidence_threshold"]

    # Read image contents
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file upload: {e}")

    # Resize to model input shape
    img_size = CONFIG["model"]["image_size"]
    img_resized = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    input_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # Predict
    cls_preds, box_preds = MODEL.predict(input_tensor, verbose=0)
    boxes, scores, classes = decode_predictions(
        cls_preds[0], box_preds[0],
        score_threshold=confidence,
        iou_threshold=CONFIG["inference"]["iou_threshold"]
    )

    class_names = {int(k): v for k, v in CONFIG["classes"].items()}
    detections_list = []

    for i in range(len(boxes)):
        score = float(scores[i])
        cls_id = int(classes[i])
        disease = class_names.get(cls_id, f"Class {cls_id}")
        box = boxes[i].tolist()
        lung_location = estimate_lung_region(box)

        detections_list.append({
            "disease": f"Possible {disease}",
            "disease_raw": disease,
            "class_id": cls_id,
            "confidence_percent": round(score * 100, 1),
            "confidence_score": round(score, 4),
            "bbox_normalized": [round(coord, 4) for coord in box],
            "anatomical_location": lung_location
        })

    # Render annotated image
    annotated_rgb = draw_detections(
        img_resized, boxes, scores, classes,
        confidence_threshold=confidence,
        class_names=class_names,
        draw_disclaimer=True
    )

    # Convert annotated image to Base64 PNG
    annotated_pil = Image.fromarray(annotated_rgb)
    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "filename": file.filename,
        "detection_count": len(detections_list),
        "detections": detections_list,
        "confidence_threshold_used": confidence,
        "annotated_image_base64": f"data:image/png;base64,{img_b64}",
        "disclaimer": MEDICAL_DISCLAIMER
    }

# Serve frontend directory as static files if exists
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
