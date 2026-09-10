import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision import transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM++ (identical to script 11 — shared logic)
# ──────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self._activations = o.detach()

    def _save_grad(self, m, gi, go):
        self._gradients = go[0].detach()

    def __call__(self, x: torch.Tensor):
        self.model.eval()
        x_req  = x.requires_grad_(True)
        output = self.model(x_req)
        score  = output[0, 0]
        conf   = torch.sigmoid(score).item()
        self.model.zero_grad()
        score.backward()

        grads    = self._gradients
        acts     = self._activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom    = 2.0 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha    = grads_sq / denom
        weights  = (alpha * F.relu(grads)).sum(dim=(2, 3))

        cam = (weights[0, :, None, None] * acts[0]).sum(dim=0)
        cam = F.relu(cam).cpu().numpy().astype(np.float32)
        cam = gaussian_filter(cam, sigma=3)

        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
        cam[cam < 0.15] = 0.0
        return cam, conf


# ──────────────────────────────────────────────────────────────────────
# Bilateral lung mask
# ──────────────────────────────────────────────────────────────────────
def build_bilateral_lung_mask(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    right  = np.clip(1.0 - ((x - 0.28) / 0.21)**2 - ((y - 0.47) / 0.31)**2, 0.0, 1.0)
    left   = np.clip(1.0 - ((x - 0.72) / 0.21)**2 - ((y - 0.47) / 0.31)**2, 0.0, 1.0)
    center = np.clip(1.0 - ((x - 0.50) / 0.12)**2 - ((y - 0.42) / 0.28)**2, 0.0, 1.0)
    mask   = np.power(np.maximum(right, np.maximum(left, center)), 0.6)
    mask[:int(0.12 * h), :] = 0.0
    mask[int(0.83 * h):, :] = 0.0
    mask[:, :int(0.05 * w)] = 0.0
    mask[:, w - int(0.05 * w):] = 0.0
    return mask.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path: Path, image_size: int, device: torch.device):
    img     = Image.open(img_path).convert("RGB")
    img_res = img.resize((image_size, image_size), Image.BILINEAR)
    arr     = np.array(img_res, dtype=np.float32) / 255.0
    norm    = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor  = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return img_res, arr, tensor


# ──────────────────────────────────────────────────────────────────────
# Lung Containment Score
# ──────────────────────────────────────────────────────────────────────
def compute_lcs(cam: np.ndarray, lung_mask: np.ndarray) -> float:
    total  = cam.sum()
    inside = (cam * lung_mask).sum()
    return float(inside / (total + 1e-7)) * 100.0


# ──────────────────────────────────────────────────────────────────────
# Collect sample images
# ──────────────────────────────────────────────────────────────────────
def collect_samples(data_raw: Path, n_per_class: int = 5):
    samples = []

    # TB positive
    tb_imgs = list((data_raw / "tuberculosis" / "images").glob("*.png"))
    for p in tb_imgs[:n_per_class]:
        samples.append({"path": p, "true_label": 1, "label_name": "TB"})

    # Pneumonia positive
    for p in list((data_raw / "dataset1" / "val" / "1").glob("*.jp*g"))[:n_per_class]:
        samples.append({"path": p, "true_label": 1, "label_name": "Pneumonia"})

    # Normal
    for p in list((data_raw / "dataset1" / "val" / "0").glob("*.jp*g"))[:n_per_class]:
        samples.append({"path": p, "true_label": 0, "label_name": "Normal"})

    return samples


# ──────────────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate Grad-CAM++ quality via Lung Containment Score (LCS).")
    parser.add_argument("--model-path", type=str, default="models/production/best_tbx11k_pytorch.pth",
                        help="Path to classifier .pth file.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--n-samples",  type=int, default=5,
                        help="Number of images per class to validate.")
    parser.add_argument("--lcs-threshold", type=float, default=70.0,
                        help="Minimum acceptable mean LCS %%. Fails if below this.")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Directory to save individual CAM grids (optional).")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger("validate_gradcam")
    logger.info("=== STEP 15: GRAD-CAM++ QUALITY VALIDATION ===")

    config   = load_config(args.config)
    data_raw = Path(config["paths"]["data_raw"])
    plots_dir = Path(config["paths"]["results_plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir) if args.output_dir else plots_dir / "gradcam_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    logger.info(f"Loaded model from: {model_path}")

    # Grad-CAM++ on features[6]
    grad_cam = GradCAMPlusPlus(model, model.features[6])
    lung_mask = build_bilateral_lung_mask(args.image_size, args.image_size)
    cmap = plt.get_cmap("jet")

    # Collect test images
    samples = collect_samples(data_raw, n_per_class=args.n_samples)
    if not samples:
        logger.error("No sample images found. Check data/raw directory structure.")
        return

    logger.info(f"Running Grad-CAM++ validation on {len(samples)} images...")

    results = []
    all_lcs  = []

    for idx, s in enumerate(samples):
        img_path = s["path"]
        if not img_path.exists():
            logger.warning(f"  Skipping (not found): {img_path}")
            continue

        try:
            img_res, img_arr, img_tensor = preprocess(img_path, args.image_size, device)
            cam, conf = grad_cam(img_tensor)

            # Upsample CAM to image size
            cam_up = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (args.image_size, args.image_size), Image.BILINEAR
                )
            ) / 255.0

            # Apply mask
            cam_masked = cam_up * lung_mask
            if cam_masked.max() > 0:
                cam_masked = cam_masked / cam_masked.max()

            lcs = compute_lcs(cam_up, lung_mask)
            all_lcs.append(lcs)
            pred_label = "Abnormal" if conf >= 0.5 else "Normal"

            results.append({
                "image":     img_path.name,
                "true_label": s["label_name"],
                "prediction": pred_label,
                "confidence": round(conf * 100, 1),
                "lcs_pct":   round(lcs, 1),
            })

            logger.info(
                f"  [{idx+1:02d}] {img_path.name:<30} | "
                f"GT: {s['label_name']:<10} Pred: {pred_label:<8} "
                f"Conf: {conf*100:.1f}% | LCS: {lcs:.1f}%"
            )

            # Save 4-panel figure per image
            img_u8 = (img_arr * 255).astype(np.uint8)
            colored_masked = (cmap(cam_masked)[:, :, :3] * 255).astype(np.uint8)
            overlay = (img_u8 * 0.55 + colored_masked * 0.45).astype(np.uint8)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img_res);           axes[0].set_title("Input X-Ray"); axes[0].axis("off")
            axes[1].imshow(cam_up,  cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("Grad-CAM++ Raw"); axes[1].axis("off")
            axes[2].imshow(cam_masked, cmap="jet", vmin=0, vmax=1)
            axes[2].set_title(f"Lung-Masked (LCS: {lcs:.1f}%)"); axes[2].axis("off")
            axes[3].imshow(overlay)
            axes[3].set_title(f"Overlay | Conf: {conf*100:.1f}%"); axes[3].axis("off")
            plt.suptitle(f"{img_path.name} | GT: {s['label_name']} | Pred: {pred_label}", fontsize=12)
            plt.tight_layout()
            fig_path = out_dir / f"gradcam_{idx+1:02d}_{img_path.stem}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.error(f"  Failed on {img_path.name}: {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    if not all_lcs:
        logger.error("No valid results — cannot compute summary.")
        return

    mean_lcs   = np.mean(all_lcs)
    median_lcs = np.median(all_lcs)
    min_lcs    = np.min(all_lcs)
    max_lcs    = np.max(all_lcs)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  GRAD-CAM++ VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total images tested : {len(all_lcs)}")
    logger.info(f"  Mean  LCS           : {mean_lcs:.1f}%")
    logger.info(f"  Median LCS          : {median_lcs:.1f}%")
    logger.info(f"  Min / Max LCS       : {min_lcs:.1f}% / {max_lcs:.1f}%")
    logger.info(f"  Threshold           : {args.lcs_threshold:.1f}%")
    logger.info(f"  PASS/FAIL           : {'✓ PASS' if mean_lcs >= args.lcs_threshold else '✗ FAIL — model still focusing on edge shortcuts'}")
    logger.info("=" * 60)

    # Summary table CSV
    df_results = pd.DataFrame(results)
    csv_path = out_dir / "lcs_summary.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"  CSV results saved to: {csv_path}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(all_lcs) * 0.8), 5))
    labels  = [r["image"][:20] for r in results]
    colors  = ["#22c55e" if v >= args.lcs_threshold else "#ef4444" for v in all_lcs]
    bars    = ax.bar(range(len(all_lcs)), all_lcs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(args.lcs_threshold, color="#f59e0b", linestyle="--", linewidth=2,
               label=f"Target LCS={args.lcs_threshold:.0f}%")
    ax.axhline(mean_lcs, color="#3b82f6", linestyle="-", linewidth=2,
               label=f"Mean LCS={mean_lcs:.1f}%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Lung Containment Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Grad-CAM++ Lung Containment Score per Image", fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    chart_path = plots_dir / "gradcam_lcs_summary.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  LCS bar chart saved to: {chart_path}")

    if mean_lcs < args.lcs_threshold:
        logger.warning(
            f"⚠️  Mean LCS={mean_lcs:.1f}% is below threshold={args.lcs_threshold:.1f}%. "
            "Model may still be using shortcut features. Consider more training epochs or stronger augmentation."
        )
    else:
        logger.info(f"✓ Validation PASSED. Mean LCS={mean_lcs:.1f}% ≥ {args.lcs_threshold:.1f}%")


if __name__ == "__main__":
    main()
