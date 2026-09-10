import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNetB0."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
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

        # Weight feature maps by gradient averages
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam, torch.sigmoid(score).item()

def generate_gradcam_overlay(img_path: Path, model_path: Path, output_plot_path: Path, image_size=384):
    logger = setup_logger("gradcam")
    logger.info("=== GENERATING GRAD-CAM EXPLAINABLE AI HEATMAP ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # Build model
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # EfficientNet final feature layer
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess image
    original_img = Image.open(img_path).convert("RGB")
    resized_img = original_img.resize((image_size, image_size), Image.BILINEAR)

    img_tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_norm = ((img_tensor - mean) / std).unsqueeze(0).to(device)

    cam, confidence = grad_cam(img_norm)

    # Resize CAM mask to image resolution
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR)
    cam_mask = np.array(cam_pil) / 255.0

    # Plot Original, Heatmap, and Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(resized_img)
    axes[0].set_title("Input Chest X-Ray", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(cam_mask, cmap="jet")
    axes[1].set_title("Grad-CAM Saliency Heatmap", fontsize=11, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(resized_img)
    axes[2].imshow(cam_mask, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Explainable AI Overlay (Conf: {confidence*100:.1f}%)", fontsize=11, fontweight='bold')
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Heatmap: Highlighting Neural Network Decision Regions", fontsize=13, fontweight='bold')
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved Grad-CAM Saliency Heatmap to: {output_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Saliency Heatmaps on Chest X-Rays.")
    parser.add_argument("--img-path", type=str, default=None, help="Path to input X-ray image.")
    parser.add_argument("--model-path", type=str, default="models/production/best_tbx11k_pytorch.pth")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    plots_dir = Path(config["paths"]["results_plots"])

    # Find sample image if not provided
    img_path = Path(args.img_path) if args.img_path else None
    if not img_path or not img_path.exists():
        sample_candidates = list(Path("data_zips/tbx11k-simplified/images").glob("*.png")) + list(Path("data/raw/dataset1/train").rglob("*.jpeg"))
        if sample_candidates:
            img_path = sample_candidates[0]
        else:
            print("No sample image found. Provide --img-path.")
            return

    output_plot = plots_dir / "gradcam_explainable_heatmap.png"
    generate_gradcam_overlay(img_path, Path(args.model_path), output_plot)

if __name__ == "__main__":
    main()
