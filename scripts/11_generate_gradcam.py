import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM++ Implementation
# ──────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """Grad-CAM++ for EfficientNetB0.

    Grad-CAM++ uses second-order gradient weighting, which:
    - Better handles *multiple* discriminative regions (bilateral lung fields)
    - Is more robust when gradient magnitudes are uneven across the spatial map
    - Produces sharper, less diffuse localization than standard Grad-CAM

    Target layer: model.features[6]
      - Last MBConv6 block (stride-2) in EfficientNetB0
      - Spatial resolution: 24×24 for 384px input  (vs 12×12 for features[-1])
      - Rich semantic content + still preserves spatial precision
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor = None
        self._gradients:   torch.Tensor = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor):
        """
        Returns:
            cam  (H, W) float32  normalized [0, 1]
            conf float           sigmoid prediction confidence
        """
        self.model.eval()

        # Forward pass — keep graph for backward
        x_req = x.requires_grad_(True)
        output = self.model(x_req)
        score  = output[0, 0]
        conf   = torch.sigmoid(score).item()

        # Backward
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Grad-CAM++ weight computation
        # α^c_kij = (∂²y^c / ∂A^k_ij²) / (2·∂²y^c/∂A^k_ij² + ΣΣ A^k_ab · ∂³y^c/∂A^k_ij³)
        # Simplified closed-form (Chattopadhay et al., 2018):
        grads = self._gradients          # (1, C, H, W)
        acts  = self._activations        # (1, C, H, W)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom     = 2.0 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha     = grads_sq / denom     # (1, C, H, W)

        # Weights: sum over spatial, gated by ReLU on gradients
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3))  # (1, C)

        # Weighted combination of activation maps
        cam = (weights[0, :, None, None] * acts[0]).sum(dim=0)  # (H, W)
        cam = F.relu(cam)

        # Post-processing: Gaussian smooth → suppress isolated edge noise
        cam_np = cam.cpu().numpy().astype(np.float32)
        cam_np = gaussian_filter(cam_np, sigma=3)

        # Normalize
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Threshold weak activations to further suppress edge noise
        cam_np[cam_np < 0.15] = 0.0

        return cam_np, conf


# ──────────────────────────────────────────────────────────────────────
# Bilateral Lung Mask
# ──────────────────────────────────────────────────────────────────────
def build_bilateral_lung_mask(h: int, w: int) -> np.ndarray:
    """Build a soft mask covering both lung fields in a frontal CXR.

    Anatomy-guided parameters (normalized coordinates):
      - Left  lung center:  (cx=0.28, cy=0.47), radii (rx=0.20, ry=0.30)
      - Right lung center:  (cx=0.72, cy=0.47), radii (rx=0.20, ry=0.30)
      - Upper apex:         starts at ~15% from top
      - Lower base:         ends at ~80% from top (above diaphragm)
    Combined with a global ellipse to catch central/mediastinal regions.
    """
    y = np.linspace(0, 1, h)[:, None]  # (H, 1)
    x = np.linspace(0, 1, w)[None, :]  # (1, W)

    # Right lung (patient's right = image left)
    cx_r, cy_r, rx_r, ry_r = 0.28, 0.47, 0.21, 0.31
    right_lung = np.clip(1.0 - ((x - cx_r) / rx_r) ** 2 - ((y - cy_r) / ry_r) ** 2, 0.0, 1.0)

    # Left lung (patient's left = image right)
    cx_l, cy_l, rx_l, ry_l = 0.72, 0.47, 0.21, 0.31
    left_lung  = np.clip(1.0 - ((x - cx_l) / rx_l) ** 2 - ((y - cy_l) / ry_l) ** 2, 0.0, 1.0)

    # Central zone (mediastinum / trachea / carina)
    cx_c, cy_c, rx_c, ry_c = 0.50, 0.42, 0.12, 0.28
    central    = np.clip(1.0 - ((x - cx_c) / rx_c) ** 2 - ((y - cy_c) / ry_c) ** 2, 0.0, 1.0)

    # Combine and apply a sharpening exponent (steeper falloff than sqrt)
    mask = np.maximum(right_lung, np.maximum(left_lung, central))
    mask = np.power(mask, 0.6)  # exponent < 1 broadens slightly; increase for sharper cutoff

    # Hard-zero everything above ~12% (top border / shoulders)
    cutoff_top = int(0.12 * h)
    mask[:cutoff_top, :] = 0.0

    # Hard-zero everything below ~83% (abdomen / lower border)
    cutoff_bot = int(0.83 * h)
    mask[cutoff_bot:, :] = 0.0

    # Hard-zero left/right margins (5%) where borders/markers live
    cutoff_side = int(0.05 * w)
    mask[:, :cutoff_side]  = 0.0
    mask[:, w - cutoff_side:] = 0.0

    return mask.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────
def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────
# Main generator function
# ──────────────────────────────────────────────────────────────────────
def generate_gradcam_overlay(
    img_path: Path,
    model_path: Path,
    output_plot_path: Path,
    image_size: int = 384
):
    logger = setup_logger("gradcam_pp")
    logger.info("=== GENERATING GRAD-CAM++ EXPLAINABLE AI HEATMAP ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    model = load_model(model_path, device)

    # Target layer: features[6] = last MBConv6 stride-2 block
    # Spatial resolution: 24×24 for 384px → much finer than features[-1] (12×12)
    target_layer = model.features[6]
    logger.info("Grad-CAM++ target layer: model.features[6] (24×24 spatial resolution)")
    grad_cam_pp = GradCAMPlusPlus(model, target_layer)

    # Preprocess
    original_img = Image.open(img_path).convert("RGB")
    resized_img  = original_img.resize((image_size, image_size), Image.BILINEAR)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_arr - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Run Grad-CAM++
    cam, confidence = grad_cam_pp(img_tensor)

    # Upsample CAM to image size
    cam_pil  = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    cam_full = np.array(cam_pil) / 255.0

    # Apply bilateral lung mask
    lung_mask = build_bilateral_lung_mask(image_size, image_size)
    cam_masked = cam_full * lung_mask
    if cam_masked.max() > 0:
        cam_masked = cam_masked / cam_masked.max()

    # Colour overlays
    cmap = plt.get_cmap("jet")
    colored_raw    = (cmap(cam_full)[:, :, :3] * 255).astype(np.uint8)
    colored_masked = (cmap(cam_masked)[:, :, :3] * 255).astype(np.uint8)
    img_u8 = (img_arr * 255).astype(np.uint8)

    overlay_raw    = (img_u8 * 0.55 + colored_raw    * 0.45).astype(np.uint8)
    overlay_masked = (img_u8 * 0.55 + colored_masked * 0.45).astype(np.uint8)

    # Lung Containment Score
    total_cam_energy = cam_full.sum()
    masked_energy    = (cam_full * lung_mask).sum()
    lcs = (masked_energy / (total_cam_energy + 1e-7)) * 100.0
    logger.info(f"Lung Containment Score (LCS): {lcs:.1f}%  (target ≥ 70%)")

    # 4-panel plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(resized_img)
    axes[0].set_title("Input Chest X-Ray", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(cam_full, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM++ Raw Heatmap", fontsize=11, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(cam_masked, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"Lung-Masked CAM (LCS: {lcs:.1f}%)", fontsize=11, fontweight='bold')
    axes[2].axis("off")

    axes[3].imshow(overlay_masked)
    axes[3].set_title(f"Explainable AI Overlay (Conf: {confidence*100:.1f}%)", fontsize=11, fontweight='bold')
    axes[3].axis("off")

    plt.suptitle(
        "Grad-CAM++ · features[6] target · Bilateral Lung Mask Applied",
        fontsize=13, fontweight='bold'
    )
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Grad-CAM++ heatmap to: {output_plot_path}")
    return lcs


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ Saliency Heatmaps on Chest X-Rays.")
    parser.add_argument("--img-path",   type=str, default=None)
    parser.add_argument("--model-path", type=str, default="models/production/best_tbx11k_pytorch.pth")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--config",     type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    plots_dir = Path(config["paths"]["results_plots"])

    img_path = Path(args.img_path) if args.img_path else None
    if not img_path or not img_path.exists():
        candidates = (
            list(Path("data/raw/tuberculosis/images").glob("*.png")) +
            list(Path("data/raw/dataset1/train").rglob("*.jpeg"))
        )
        if candidates:
            img_path = candidates[0]
        else:
            print("No sample image found. Provide --img-path.")
            return

    output_plot = plots_dir / "gradcam_pp_heatmap.png"
    generate_gradcam_overlay(img_path, Path(args.model_path), output_plot, args.image_size)


if __name__ == "__main__":
    main()
