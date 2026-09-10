"""
Script 14: Batch Generate Clinical Agent Reports & Grad-CAM Heatmaps for Documentation
========================================================================================
Runs test scans from Dataset1 and Tuberculosis datasets through:
1. PyTorch Classification (Pneumonia & Tuberculosis probabilities)
2. Grad-CAM Explainable AI (XAI) Heatmap visual localization
3. Medical Agentic Decision Pipeline (Prescriptions, Vitals Protocol, Red Flags, Nutrition)

Generates documentation artifacts in results/reports/ and plots in results/plots/.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger, MEDICAL_DISCLAIMER
from src.agentic_ai import MedicalAgenticPipeline

class GradCAM:
    """Grad-CAM generator for EfficientNet-B0."""
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

def load_classifier(model_path: Path, device: torch.device):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    logger = setup_logger("batch_documentation_generator")
    logger.info("=== STEP 14: BATCH GENERATE CLINICAL REPORTS & GRAD-CAM HEATMAPS ===")

    config = load_config()
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # Output directories
    plots_dir = Path(paths["results_plots"]) / "documentation"
    reports_dir = Path(paths["results_metrics"]).parent / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load classification models
    pneumonia_model_path = Path("models/production/best_classifier_pytorch.pth")
    tb_model_path = Path("models/production/best_tbx11k_pytorch.pth")

    models = {}
    if pneumonia_model_path.exists():
        models["Pneumonia"] = load_classifier(pneumonia_model_path, device)
    if tb_model_path.exists():
        models["Tuberculosis"] = load_classifier(tb_model_path, device)

    if not models:
        logger.error("No classification models found in models/production/")
        sys.exit(1)

    agent_pipeline = MedicalAgenticPipeline()

    # Collect sample test images
    base_raw = Path(paths["data_raw"])
    test_cases = [
        {"name": "Pneumonia_Test_Case_1", "type": "Pneumonia", "path": base_raw / "dataset1" / "test" / "1" / "test_1001.jpg"},
        {"name": "Pneumonia_Test_Case_2", "type": "Pneumonia", "path": base_raw / "dataset1" / "test" / "1" / "test_1002.jpg"},
        {"name": "Tuberculosis_Test_Case_1", "type": "Tuberculosis", "path": base_raw / "tuberculosis" / "images" / "tb0001.png"},
        {"name": "Normal_Healthy_Test_Case_1", "type": "Normal", "path": base_raw / "dataset1" / "test" / "0" / "test_0001.jpg"},
    ]

    # Filter existing test cases
    valid_test_cases = []
    for tc in test_cases:
        if tc["path"].exists():
            valid_test_cases.append(tc)
        else:
            # Fallback search
            if tc["type"] == "Pneumonia":
                imgs = list((base_raw / "dataset1" / "test" / "1").glob("*.jpg"))
                if imgs: valid_test_cases.append({"name": tc["name"], "type": tc["type"], "path": imgs[0]})
            elif tc["type"] == "Tuberculosis":
                imgs = list((base_raw / "tuberculosis" / "images").glob("*.png"))
                if imgs: valid_test_cases.append({"name": tc["name"], "type": tc["type"], "path": imgs[0]})
            elif tc["type"] == "Normal":
                imgs = list((base_raw / "dataset1" / "test" / "0").glob("*.jpg"))
                if imgs: valid_test_cases.append({"name": tc["name"], "type": tc["type"], "path": imgs[0]})

    logger.info(f"Processing {len(valid_test_cases)} documentation test cases...")

    documentation_reports = []
    doc_markdown_content = "# Clinical AI Agent & Grad-CAM Documentation Reports\n\n"
    doc_markdown_content += "Generated diagnostic payloads, Grad-CAM heatmaps, and evidence-based prescriptions for model documentation.\n\n"

    image_size = 384
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, tc in enumerate(valid_test_cases, 1):
        img_path = tc["path"]
        case_name = tc["name"]
        expected_type = tc["type"]

        logger.info(f"[{idx}/{len(valid_test_cases)}] Processing: {case_name} ({img_path.name})")

        original_img = Image.open(img_path).convert("RGB")
        resized_img = original_img.resize((image_size, image_size), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
        img_norm = ((img_tensor - mean) / std).unsqueeze(0).to(device)

        # Run Classification
        pn_prob = 0.0
        tb_prob = 0.0

        if "Pneumonia" in models:
            with torch.no_grad():
                pn_logit = models["Pneumonia"](img_norm).squeeze()
                pn_prob = torch.sigmoid(pn_logit).item()

        if "Tuberculosis" in models:
            with torch.no_grad():
                tb_logit = models["Tuberculosis"](img_norm).squeeze()
                tb_prob = torch.sigmoid(tb_logit).item()

        # Primary Condition
        max_p = max(pn_prob, tb_prob)
        if max_p >= 0.5:
            primary_condition = "Tuberculosis" if tb_prob > pn_prob else "Pneumonia"
        else:
            primary_condition = "Normal"

        # Grad-CAM Heatmap
        gradcam_model = models.get(primary_condition if primary_condition in models else "Pneumonia")
        cam_mask = np.zeros((image_size, image_size), dtype=np.float32)
        if gradcam_model:
            grad_cam = GradCAM(gradcam_model, gradcam_model.features[-1])
            cam, _ = grad_cam(img_norm)
            cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR)
            cam_mask = np.array(cam_pil) / 255.0

        # Plot Side-by-Side Figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(resized_img)
        axes[0].set_title(f"Input Scan: {case_name}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_mask, cmap="jet")
        axes[1].set_title("Grad-CAM Activation Map", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        plt_img = np.array(resized_img)
        cmap = plt.get_cmap("jet")
        colored_cam = (cmap(cam_mask)[:, :, :3] * 255).astype(np.uint8)
        overlay = (plt_img * 0.6 + colored_cam * 0.4).astype(np.uint8)

        axes[2].imshow(overlay)
        if primary_condition == "Normal":
            axes[2].set_title(f"Overlay: Normal Scan (Pneumonia: {pn_prob*100:.1f}%, TB: {tb_prob*100:.1f}%)", fontsize=10, fontweight="bold")
        else:
            axes[2].set_title(f"Overlay: {primary_condition} ({max_p*100:.1f}%)", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        plot_file = plots_dir / f"{case_name}_gradcam_report.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()

        # Medical Agent Report
        agent_report = agent_pipeline.run_pipeline(
            class_name=primary_condition,
            boxes=[],
            confidences=[max_p]
        )

        record = {
            "case_name": case_name,
            "image_filename": img_path.name,
            "expected_ground_truth": expected_type,
            "predicted_condition": primary_condition,
            "pneumonia_probability_percent": round(pn_prob * 100, 1),
            "tuberculosis_probability_percent": round(tb_prob * 100, 1),
            "plot_path": str(plot_file),
            "agent_report": agent_report
        }
        documentation_reports.append(record)

        # Build Markdown Entry
        spatial_data = agent_report.get("clinical_spatial_analysis", {})
        rx_data = agent_report.get("prescription_recommendation", {})
        monitor_data = agent_report.get("health_monitoring_protocol", {})

        severity_val = spatial_data.get("severity", "Normal" if primary_condition == "Normal" else "Moderate")

        doc_markdown_content += f"## Case {idx}: {case_name}\n"
        doc_markdown_content += f"- **Image**: `{img_path.name}`\n"
        doc_markdown_content += f"- **Ground Truth**: `{expected_type}` | **Predicted**: `{primary_condition}`\n"
        doc_markdown_content += f"- **Pneumonia Probability**: `{pn_prob*100:.1f}%`\n"
        doc_markdown_content += f"- **Tuberculosis Probability**: `{tb_prob*100:.1f}%`\n"
        doc_markdown_content += f"- **Severity Grade**: `{severity_val}`\n\n"
        
        doc_markdown_content += "### Prescriptions & Medications\n"
        for m in rx_data.get("medications", []):
            doc_markdown_content += f"- **{m['drug']}** ({m.get('dosage','')}): {m.get('frequency','')} for {m.get('duration','')} — *{m.get('instructions','') or m.get('clinical_rationale','')}*\n"
        doc_markdown_content += "\n"

        doc_markdown_content += "### 7-Day Monitoring Schedule & Vitals\n"
        for v in monitor_data.get("monitoring_schedule", []):
            v_check = v.get("vitals_to_check", [])
            v_str = ", ".join(v_check) if isinstance(v_check, list) else str(v_check)
            doc_markdown_content += f"- **Day {v.get('day','')} ({v.get('focus','')})**: Check `{v_str}` -> {v.get('action','')}\n"
        doc_markdown_content += "\n"

        doc_markdown_content += "### Red Flag Symptoms\n"
        for rf in monitor_data.get("emergency_red_flags", []):
            doc_markdown_content += f"- ⚠️ {rf}\n"
        doc_markdown_content += "\n---\n\n"

    # Save JSON and Markdown documentation
    json_doc_file = reports_dir / "clinical_reports_documentation.json"
    md_doc_file = reports_dir / "clinical_reports_documentation.md"

    with open(json_doc_file, "w", encoding="utf-8") as f:
        json.dump(documentation_reports, f, indent=2)

    with open(md_doc_file, "w", encoding="utf-8") as f:
        f.write(doc_markdown_content)

    logger.info(f"Saved Grad-CAM plots to: {plots_dir}")
    logger.info(f"Saved documentation JSON report to: {json_doc_file}")
    logger.info(f"Saved documentation Markdown report to: {md_doc_file}")
    logger.info("=== BATCH DOCUMENTATION GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()
