import sys
import os
import io
import json
import base64
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, MEDICAL_DISCLAIMER
from src.agentic_ai import MedicalAgenticPipeline

logger = setup_logger("fastapi_pytorch_backend")

# Initialize FastAPI App
app = FastAPI(
    title="Chest X-ray Agentic AI Healthcare Platform",
    description="PyTorch GPU Deep Learning & Agentic Clinical Decision Engine for Pneumonia & Tuberculosis.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}
AGENT_PIPELINE = MedicalAgenticPipeline()

# ─────────────────────────────────────────────────────────────
# Grad-CAM Generator
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        output = self.model(x)
        score = output[0, 0]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam, torch.sigmoid(score).item()

# ─────────────────────────────────────────────────────────────
# Model Initialization
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_pytorch_models():
    global MODELS
    logger.info("Initializing PyTorch GPU Models for FastAPI Backend...")
    prod_dir = Path(CONFIG["paths"]["models_production"])

    # 1. Pneumonia EfficientNet Classification Model
    pneumonia_path = prod_dir / "best_classifier_pytorch.pth"
    if pneumonia_path.exists():
        model_p = efficientnet_b0(weights=None)
        model_p.classifier[1] = nn.Linear(model_p.classifier[1].in_features, 1)
        model_p.load_state_dict(torch.load(pneumonia_path, map_location=DEVICE))
        model_p.to(DEVICE)
        model_p.eval()
        MODELS["pneumonia_cls"] = model_p
        logger.info(f"Loaded Pneumonia Classification Model from {pneumonia_path}")

    # 2. TB EfficientNet Classification Model
    tb_path = prod_dir / "best_tbx11k_pytorch.pth"
    if tb_path.exists():
        model_tb = efficientnet_b0(weights=None)
        model_tb.classifier[1] = nn.Linear(model_tb.classifier[1].in_features, 1)
        model_tb.load_state_dict(torch.load(tb_path, map_location=DEVICE))
        model_tb.to(DEVICE)
        model_tb.eval()
        MODELS["tb_cls"] = model_tb
        logger.info(f"Loaded TB Classification Model from {tb_path}")

    # 3. TB / Pneumonia RetinaNet Detector Model
    det_path = prod_dir / "best_tbx11k_detector_pytorch.pth"
    if not det_path.exists():
        det_path = prod_dir / "best_model_pytorch.pth"
    if det_path.exists():
        state_dict = torch.load(det_path, map_location=DEVICE)
        num_classes = 3
        for k in state_dict.keys():
            if "head.classification_head.cls_logits.weight" in k:
                num_classes = state_dict[k].shape[0] // 9
                break
        model_det = retinanet_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=ResNet50_Weights.DEFAULT)
        model_det.load_state_dict(state_dict)
        model_det.to(DEVICE)
        model_det.eval()
        MODELS["detector"] = model_det
        logger.info(f"Loaded RetinaNet Detector ({num_classes} classes) from {det_path}")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "N/A",
        "models_loaded": list(MODELS.keys()),
        "disclaimer": MEDICAL_DISCLAIMER
    }

@app.get("/api/samples")
def get_sample_xrays():
    """Provides sample X-ray images for instant 1-click testing."""
    samples = []
    base_raw = Path(CONFIG["paths"]["data_raw"])

    tb_imgs = list((base_raw / "tuberculosis" / "images").glob("*.png"))[:3]
    for p in tb_imgs:
        samples.append({
            "id": f"sample_{p.stem}",
            "name": f"TB Sample ({p.name})",
            "type": "Tuberculosis",
            "path": str(p)
        })

    pn_pos = list((base_raw / "dataset1" / "val" / "1").glob("*.jpg"))[:3]
    for p in pn_pos:
        samples.append({
            "id": f"sample_{p.stem}",
            "name": f"Pneumonia Sample ({p.name})",
            "type": "Pneumonia",
            "path": str(p)
        })

    pn_norm = list((base_raw / "dataset1" / "val" / "0").glob("*.jpg"))[:3]
    for p in pn_norm:
        samples.append({
            "id": f"sample_{p.stem}",
            "name": f"Normal Healthy Sample ({p.name})",
            "type": "Normal",
            "path": str(p)
        })

    return {"samples": samples}

@app.post("/api/detect")
async def detect_xray(
    file: UploadFile = File(None),
    sample_path: str = Form(None),
    confidence: float = Form(0.3)
):
    if not file and not sample_path:
        raise HTTPException(status_code=400, detail="Provide an uploaded image file or a sample_path.")

    # Read image
    if file:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        filename = file.filename
    else:
        p = Path(sample_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Sample file not found at {sample_path}")
        pil_img = Image.open(p).convert("RGB")
        filename = p.name

    # Image Preprocessing (384x384 for EfficientNet classification)
    img_384 = pil_img.resize((384, 384), Image.BILINEAR)
    img_t = torch.from_numpy(np.array(img_384)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_norm = ((img_t - mean) / std).unsqueeze(0).to(DEVICE)

    # 1. Classification Probabilities
    pn_prob = 0.0
    tb_prob = 0.0

    if "pneumonia_cls" in MODELS:
        with torch.no_grad():
            pn_logit = MODELS["pneumonia_cls"](img_norm).squeeze()
            pn_prob = torch.sigmoid(pn_logit).item()

    if "tb_cls" in MODELS:
        with torch.no_grad():
            tb_logit = MODELS["tb_cls"](img_norm).squeeze()
            tb_prob = torch.sigmoid(tb_logit).item()

    # 2. Grad-CAM Explainable AI Heatmap
    cam_b64 = ""
    gradcam_model = MODELS.get("tb_cls") or MODELS.get("pneumonia_cls")
    if gradcam_model:
        try:
            grad_cam = GradCAM(gradcam_model, gradcam_model.features[-1])
            cam, _ = grad_cam(img_norm)
            cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)
            cam_mask = np.array(cam_pil) / 255.0

            # Generate Overlay Image
            plt_img = np.array(img_384)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("jet")
            colored_cam = (cmap(cam_mask)[:, :, :3] * 255).astype(np.uint8)
            overlay = (plt_img * 0.6 + colored_cam * 0.4).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay)
            
            buffered = io.BytesIO()
            overlay_pil.save(buffered, format="PNG")
            cam_b64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")

    # 3. Object Detection Bounding Boxes
    detections = []
    annotated_b64 = ""

    if "detector" in MODELS:
        try:
            img_512 = pil_img.resize((512, 512), Image.BILINEAR)
            det_tensor = torch.from_numpy(np.array(img_512)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                pred = MODELS["detector"]([det_tensor.to(DEVICE)])[0]

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            draw_img = img_512.copy()
            draw = ImageDraw.Draw(draw_img)

            for box, label, score in zip(boxes, labels, scores):
                if score < confidence:
                    continue
                x1, y1, x2, y2 = [float(c) for c in box]
                # Normalized coords
                nx1, ny1, nx2, ny2 = x1 / 512.0, y1 / 512.0, x2 / 512.0, y2 / 512.0
                disease_name = "Tuberculosis" if label == 2 else "Pneumonia"
                color = "#44AAFF" if label == 2 else "#FF4444"

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                tag = f"{disease_name} {score*100:.1f}%"
                draw.rectangle([x1, y1 - 16, x1 + len(tag) * 8, y1], fill=color)
                draw.text((x1 + 2, y1 - 15), tag, fill="white")

                detections.append({
                    "disease": disease_name,
                    "confidence_percent": round(score * 100, 1),
                    "confidence_score": round(score, 4),
                    "bbox_normalized": [round(ny1, 4), round(nx1, 4), round(ny2, 4), round(nx2, 4)]
                })

            buf_det = io.BytesIO()
            draw_img.save(buf_det, format="PNG")
            annotated_b64 = f"data:image/png;base64,{base64.b64encode(buf_det.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.warning(f"Detection inference failed: {e}")

    # Fallback annotated image if no detection boxes drawn
    if not annotated_b64:
        buf_orig = io.BytesIO()
        pil_img.resize((512, 512), Image.BILINEAR).save(buf_orig, format="PNG")
        annotated_b64 = f"data:image/png;base64,{base64.b64encode(buf_orig.getvalue()).decode('utf-8')}"

    # 4. Agentic AI Clinical Processing Pipeline
    primary_condition = "Normal"
    max_prob = max(pn_prob, tb_prob)
    if max_prob >= 0.5:
        primary_condition = "Tuberculosis" if tb_prob > pn_prob else "Pneumonia"

    boxes = [d["bbox_normalized"] for d in detections]
    confidences = [d["confidence_score"] for d in detections]
    agent_medical_report = AGENT_PIPELINE.run_pipeline(
        class_name=primary_condition,
        boxes=boxes,
        confidences=confidences
    )

    return {
        "status": "success",
        "filename": filename,
        "pneumonia_probability_percent": round(pn_prob * 100, 1),
        "tuberculosis_probability_percent": round(tb_prob * 100, 1),
        "primary_condition": primary_condition,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "gradcam_heatmap_base64": cam_b64,
        "medical_agent_report": agent_medical_report,
        "disclaimer": MEDICAL_DISCLAIMER
    }

# Serve frontend directory
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
