import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger
from src.agentic_ai import MedicalAgenticPipeline

def generate_html_report(report: dict, output_html_path: Path):
    """Renders a medical HTML report for the Agentic AI findings."""
    spatial = report.get("clinical_spatial_analysis", {})
    prescription = report.get("prescription_recommendation", {})
    monitoring = report.get("health_monitoring_protocol", {})
    nutrition = report.get("pulmonary_nutrition_plan", {})

    severity = spatial.get("severity", "None")
    severity_color = "#dc3545" if severity == "Severe" else ("#ffc107" if severity == "Moderate" else "#28a745")

    # Render Medications Table
    med_rows = ""
    for med in prescription.get("medications", []):
        med_rows += f"""
        <tr>
            <td><strong>{med['drug_name']}</strong></td>
            <td>{med['dosage']}</td>
            <td>{med['frequency']}</td>
            <td>{med['duration']}</td>
            <td>{med['indication']}</td>
        </tr>
        """
    if not med_rows:
        med_rows = "<tr><td colspan='5' style='text-align:center;'>No pharmacotherapy required for normal radiograph.</td></tr>"

    # Render Vitals Table
    vital_rows = ""
    for v in monitoring.get("vital_checks", []):
        vital_rows += f"""
        <tr>
            <td><strong>{v['parameter']}</strong></td>
            <td>{v['frequency']}</td>
            <td><span class="badge badge-success">{v.get('target_threshold', v.get('normal_range', 'Normal'))}</span></td>
            <td><span class="badge badge-danger">{v.get('action_threshold', 'N/A')}</span></td>
        </tr>
        """

    # Render Red Flags List
    red_flags_html = "".join([f"<li>⚠️ {rf}</li>" for rf in monitoring.get("red_flag_warning_signs", monitoring.get("red_flags", []))])

    # Render Nutrition Pillars
    nutrition_html = ""
    for p in nutrition.get("key_nutritional_pillars", []):
        sources = ", ".join(p.get("sources", []))
        nutrition_html += f"""
        <div class="card card-sub">
            <h5>🥦 {p['pillar']}</h5>
            <p><strong>Benefit:</strong> {p['benefit']}</p>
            <p><strong>Sources:</strong> {sources}</p>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic AI Medical Clinical Report</title>
    <style>
        :root {{
            --primary: #1a365d;
            --accent: #2b6cb0;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --text: #2d3748;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }}
        .card h3 {{
            margin-top: 0;
            color: var(--primary);
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 8px;
        }}
        .card-sub {{
            background: #f8fafc;
            border-left: 4px solid #3182ce;
            padding: 12px 16px;
            margin-bottom: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        }}
        .badge-severity {{
            background-color: {severity_color};
            color: white;
        }}
        .badge-success {{ background-color: #c6f6d5; color: #22543d; }}
        .badge-danger {{ background-color: #fed7d7; color: #742a2a; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            font-size: 14px;
        }}
        th {{ background-color: #edf2f7; color: #4a5568; }}
        ul.red-flags {{
            list-style-type: none;
            padding-left: 0;
        }}
        ul.red-flags li {{
            background: #fff5f5;
            color: #c53030;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-weight: 500;
        }}
        .disclaimer {{
            background-color: #fffaf0;
            border: 1px solid #feebc8;
            color: #744210;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            margin-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agentic AI Healthcare System</h1>
            <p>Integrated Chest X-Ray Vision & Autonomous Clinical Decision Support Engine</p>
        </div>

        <!-- 1. Clinical Spatial Findings -->
        <div class="card">
            <h3>🫁 1. Clinical Spatial & Severity Analysis</h3>
            <p><strong>Primary Diagnosis:</strong> {spatial.get('diagnosis')}</p>
            <p><strong>Severity Grade:</strong> <span class="badge badge-severity">{severity}</span></p>
            <p><strong>Affected Lung Regions:</strong> {', '.join(spatial.get('affected_regions', []))}</p>
            <p><strong>Opacity Area Ratio:</strong> {float(spatial.get('total_opacity_ratio', 0)) * 100:.1f}% of lung field</p>
            <p><strong>Vision Model Confidence:</strong> {float(spatial.get('primary_confidence', 0)) * 100:.1f}%</p>
        </div>

        <!-- 2. Prescription & Pharmacotherapy -->
        <div class="card">
            <h3>💊 2. Evidence-Based Prescription & Pharmacotherapy</h3>
            <p><em>{prescription.get('clinical_notes', '')}</em></p>
            <table>
                <thead>
                    <tr>
                        <th>Medication</th>
                        <th>Dosage</th>
                        <th>Frequency</th>
                        <th>Duration</th>
                        <th>Clinical Indication</th>
                    </tr>
                </thead>
                <tbody>
                    {med_rows}
                </tbody>
            </table>
        </div>

        <!-- 3. Health & Symptom Monitoring Protocol -->
        <div class="card">
            <h3>📊 3. Daily Health & Vital Signs Protocol</h3>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Frequency</th>
                        <th>Target Threshold</th>
                        <th>Action Threshold (Alert)</th>
                    </tr>
                </thead>
                <tbody>
                    {vital_rows}
                </tbody>
            </table>
            <h4>🚨 Emergency Red-Flag Warning Signs:</h4>
            <ul class="red-flags">
                {red_flags_html}
            </ul>
        </div>

        <!-- 4. Pulmonary Nutrition Plan -->
        <div class="card">
            <h3>🥗 4. Pulmonary Nutrition & Recovery Diet Plan</h3>
            <p><strong>Daily Hydration Target:</strong> {nutrition.get('daily_hydration_liters', 3.0)} Liters (Warm fluids & broths)</p>
            <p><strong>Protein Requirement:</strong> {nutrition.get('protein_target', '1.2g/kg')}</p>
            {nutrition_html}
        </div>

        <div class="disclaimer">
            ⚠️ <strong>CLINICAL DISCLAIMER:</strong> {prescription.get('disclaimer', 'Automated decision support system. Final diagnosis and prescriptions must be verified by a licensed physician.')}
        </div>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Run Agentic AI Healthcare Pipeline on Chest X-Ray findings.")
    parser.add_argument("--class-name", type=str, default="Pneumonia", help="Detected disease class name.")
    parser.add_argument("--severity", type=str, default="Moderate", choices=["Mild", "Moderate", "Severe"], help="Disease severity.")
    parser.add_argument("--config", type=str, default=None, help="Config path.")
    args = parser.parse_args()

    logger = setup_logger("run_medical_agent")
    logger.info("=== STEP 10: AGENTIC AI MEDICAL PIPELINE EXECUTION ===")

    config = load_config(args.config)
    reports_path_str = config["paths"].get("results_reports", str(Path(config["project_root"]) / "results" / "reports"))
    reports_dir = Path(reports_path_str).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Sample detection boxes for Right Lower Lobe Pneumonia
    sample_boxes = [[0.52, 0.15, 0.85, 0.45]] # [ymin, xmin, ymax, xmax]
    sample_confidences = [0.914]

    pipeline = MedicalAgenticPipeline()
    report = pipeline.run_pipeline(
        class_name=args.class_name,
        boxes=sample_boxes,
        confidences=sample_confidences,
        image_size=512
    )

    # Save JSON Report
    json_path = reports_dir / "agent_medical_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved Agentic AI JSON Report to: {json_path}")

    # Generate HTML Report
    html_path = reports_dir / "agent_medical_report.html"
    generate_html_report(report, html_path)
    logger.info(f"Generated Interactive Medical HTML Report to: {html_path}")

    logger.info("=== AGENTIC AI EXECUTION COMPLETE ===")

if __name__ == "__main__":
    main()
