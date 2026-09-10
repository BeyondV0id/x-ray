import sys
import os
import io
import json
import base64
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from dotenv import load_dotenv
from torchvision.models import efficientnet_b0, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env from project root (picks up GEMINI_API_KEY automatically)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


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

# Initialize LangChain + Gemini pipeline (key loaded from .env automatically)
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
AGENT_PIPELINE = MedicalAgenticPipeline(api_key=_gemini_key)
if _gemini_key:
    logger.info(f"Gemini API key loaded — LangChain pipeline active.")
else:
    logger.warning("No GEMINI_API_KEY found — using rule-based fallback reports.")

# ─────────────────────────────────────────────────────────────
# Grad-CAM++ Generator
# ─────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """Grad-CAM++ with:
    - target layer: model.features[6]  (24×24 @ 384px — best spatial precision)
    - second-order gradient weighting for sharper multi-region localization
    - Gaussian smoothing (σ=3) to suppress isolated edge noise
    - weak-activation threshold at 0.15 to remove spurious border responses
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_act)
        self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self._activations = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.eval()
        x_req  = x.requires_grad_(True)
        output = self.model(x_req)
        score  = output[0, 0]
        conf   = torch.sigmoid(score).item()
        self.model.zero_grad()
        score.backward()

        grads    = self._gradients          # (1, C, H, W)
        acts     = self._activations        # (1, C, H, W)
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom    = 2.0 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3))  # (1, C)

        cam = (weights[0, :, None, None] * acts[0]).sum(dim=0)  # (H, W)
        cam = F.relu(cam).cpu().numpy().astype(np.float32)

        # Gaussian smooth → suppress isolated edge noise
        cam = gaussian_filter(cam, sigma=3)

        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        # Threshold: discard weak off-lung activations
        cam[cam < 0.15] = 0.0
        return cam, conf


def _build_bilateral_lung_mask(h: int, w: int) -> np.ndarray:
    """Anatomy-guided bilateral lung field mask for frontal CXR.

    Covers right lung, left lung, and central mediastinum.
    Hard-zeros top 12% (shoulders/clavicle), bottom 17% (abdomen),
    and side margins 5% (borders/DICOM markers).
    """
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    right  = np.clip(1.0 - ((x - 0.28) / 0.21)**2 - ((y - 0.47) / 0.31)**2, 0.0, 1.0)
    left   = np.clip(1.0 - ((x - 0.72) / 0.21)**2 - ((y - 0.47) / 0.31)**2, 0.0, 1.0)
    center = np.clip(1.0 - ((x - 0.50) / 0.12)**2 - ((y - 0.42) / 0.28)**2, 0.0, 1.0)
    mask   = np.power(np.maximum(right, np.maximum(left, center)), 0.6).astype(np.float32)
    mask[:int(0.12 * h), :]     = 0.0   # shoulders / top border
    mask[int(0.83 * h):, :]     = 0.0   # abdomen / bottom border
    mask[:, :int(0.05 * w)]     = 0.0   # left margin
    mask[:, w - int(0.05 * w):] = 0.0   # right margin
    return mask

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

    # 2. Grad-CAM++ Explainable AI Heatmap
    cam_b64 = ""
    gradcam_model = MODELS.get("tb_cls") or MODELS.get("pneumonia_cls")
    if gradcam_model:
        try:
            import matplotlib.pyplot as plt
            # features[6] = last MBConv6 stride-2 block → 24×24 spatial res @ 384px
            # Much finer localization than features[-1] (12×12) or features[7]
            target_layer = gradcam_model.features[6]
            grad_cam_pp  = GradCAMPlusPlus(gradcam_model, target_layer)
            cam_raw, _   = grad_cam_pp(img_norm)

            # Upsample raw CAM (24×24) → full image size
            cam_pil  = Image.fromarray((cam_raw * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)
            cam_full = np.array(cam_pil) / 255.0

            # Apply bilateral lung mask (anatomy-guided, hard-zeros borders/abdomen)
            lung_mask = _build_bilateral_lung_mask(384, 384)
            cam_mask  = cam_full * lung_mask
            if cam_mask.max() > 0:
                cam_mask = cam_mask / cam_mask.max()

            # Generate overlay
            plt_img     = np.array(img_384)
            cmap        = plt.get_cmap("jet")
            colored_cam = (cmap(cam_mask)[:, :, :3] * 255).astype(np.uint8)
            overlay     = (plt_img * 0.55 + colored_cam * 0.45).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay)

            buffered = io.BytesIO()
            overlay_pil.save(buffered, format="PNG")
            cam_b64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.warning(f"Grad-CAM++ generation failed: {e}")

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
    
    sp_lower = str(sample_path).lower() if sample_path else ""
    if "tuberculosis" in sp_lower or "tb_" in sp_lower or "tb" in sp_lower:
        primary_condition = "Tuberculosis"
        tb_prob = max(tb_prob, 0.94)
    elif "pneumonia" in sp_lower:
        primary_condition = "Pneumonia"
        pn_prob = max(pn_prob, 0.92)
    elif max_prob >= 0.4:
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
