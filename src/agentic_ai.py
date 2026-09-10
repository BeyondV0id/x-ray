"""
Medical Agentic AI Engine for Chest X-Ray Analysis
--------------------------------------------------
Orchestrates specialized medical sub-agents to generate:
1. Clinical Spatial & Severity Analysis (from RetinaNet/EfficientNet outputs)
2. Evidence-Based Pharmacotherapy & Prescription Recommendations
3. Daily Health & Vital Signs Monitoring Protocol (SpO2, Fever, Dyspnea)
4. Pulmonary Recovery & Anti-Inflammatory Diet Recommendations
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("src.agentic_ai")

class XRayFindingExtractor:
    """Extracts spatial, anatomical, and severity context from X-ray detection boxes."""
    
    LUNG_REGIONS = {
        "rul": "Right Upper Lobe",
        "rml": "Right Middle / Lower Lobe",
        "lul": "Left Upper Lobe",
        "lll": "Left Lower Lobe",
        "bilateral": "Bilateral Multi-Lobe Involvement"
    }

    @classmethod
    def analyze_findings(cls, class_name: str, boxes: List[List[float]], confidences: List[float], image_size: int = 512) -> Dict[str, Any]:
        """
        Analyzes bounding boxes [ymin, xmin, ymax, xmax] (normalized 0..1 or absolute).
        Returns structured clinical spatial metrics.
        """
        if not boxes or class_name.lower() in ["normal", "0_normal"]:
            return {
                "diagnosis": "Normal Chest X-Ray",
                "is_abnormal": False,
                "severity": "None",
                "affected_regions": ["No focal consolidation or opacity detected"],
                "total_opacity_ratio": 0.0,
                "box_count": 0,
                "primary_confidence": 0.99
            }

        regions_found = set()
        total_area = 0.0
        
        for box in boxes:
            # Normalize coordinates if > 1.0
            if any(c > 1.0 for c in box):
                ymin, xmin, ymax, xmax = [c / image_size for c in box]
            else:
                ymin, xmin, ymax, xmax = box

            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            area = width * height
            total_area += area

            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0

            # Anatomical mapping (X-ray image left is patient right)
            if center_x < 0.5:
                if center_y < 0.5:
                    regions_found.add(cls.LUNG_REGIONS["rul"])
                else:
                    regions_found.add(cls.LUNG_REGIONS["rml"])
            else:
                if center_y < 0.5:
                    regions_found.add(cls.LUNG_REGIONS["lul"])
                else:
                    regions_found.add(cls.LUNG_REGIONS["lll"])

        # Determine bilateral involvement
        has_right = any("Right" in r for r in regions_found)
        has_left = any("Left" in r for r in regions_found)
        if has_right and has_left:
            regions_found.add(cls.LUNG_REGIONS["bilateral"])

        # Determine severity level
        if total_area >= 0.20 or len(regions_found) >= 3 or (has_right and has_left):
            severity = "Severe"
        elif total_area >= 0.08 or len(regions_found) >= 2:
            severity = "Moderate"
        else:
            severity = "Mild"

        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.85

        return {
            "diagnosis": class_name.replace("_", " ").title(),
            "is_abnormal": True,
            "severity": severity,
            "affected_regions": list(regions_found),
            "total_opacity_ratio": round(total_area, 3),
            "box_count": len(boxes),
            "primary_confidence": round(avg_conf, 3)
        }


class PrescriptionAgent:
    """Specialized Sub-Agent for Pharmacotherapy & Prescription Recommendations."""

    @staticmethod
    def generate_prescription(finding: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = finding.get("diagnosis", "").lower()
        severity = finding.get("severity", "Mild")

        if not finding.get("is_abnormal", False):
            return {
                "medications": [],
                "clinical_notes": "No pharmacological intervention required for normal chest radiograph.",
                "disclaimer": "This system provides automated clinical decision support. Always consult a licensed medical professional."
            }

        medications = []
        
        if "pneumonia" in diagnosis or "positive" in diagnosis:
            if severity == "Mild":
                medications.append({
                    "drug_name": "Amoxicillin / Clavulanic Acid (Augmentin)",
                    "dosage": "625 mg oral",
                    "frequency": "Every 8 hours (TID)",
                    "duration": "5 - 7 Days",
                    "indication": "First-line oral antibiotic for mild community-acquired pneumonia."
                })
                medications.append({
                    "drug_name": "Azithromycin",
                    "dosage": "500 mg Day 1, then 250 mg daily",
                    "frequency": "Once daily (QD)",
                    "duration": "5 Days",
                    "indication": "Atypical coverage for respiratory pathogens."
                })
            elif severity == "Moderate":
                medications.append({
                    "drug_name": "Cefuroxime Axetil",
                    "dosage": "500 mg oral",
                    "frequency": "Every 12 hours (BID)",
                    "duration": "7 - 10 Days",
                    "indication": "Second-generation cephalosporin for moderate lower respiratory tract infection."
                })
                medications.append({
                    "drug_name": "Levofloxacin",
                    "dosage": "500 mg oral",
                    "frequency": "Once daily (QD)",
                    "duration": "7 Days",
                    "indication": "Respiratory fluoroquinolone for moderate-to-severe pneumonia."
                })
            else: # Severe
                medications.append({
                    "drug_name": "Ceftriaxone (IV/IM)",
                    "dosage": "1g - 2g IV",
                    "frequency": "Once daily (QD)",
                    "duration": "7 - 14 Days",
                    "indication": "Broad-spectrum parenteral cephalosporin for severe pneumonia requiring urgent hospitalization."
                })
                medications.append({
                    "drug_name": "Azithromycin (IV)",
                    "dosage": "500 mg IV",
                    "frequency": "Once daily (QD)",
                    "duration": "7 Days",
                    "indication": "Parenteral macrolide coverage for severe respiratory infection."
                })

            # Supportive medications for Pneumonia
            medications.append({
                "drug_name": "Paracetamol / Acetaminophen",
                "dosage": "500 mg - 650 mg oral",
                "frequency": "Every 6 hours as needed (PRN)",
                "duration": "As needed for fever > 38.0°C or pleuritic pain",
                "indication": "Antipyretic and analgesic symptom relief."
            })
            medications.append({
                "drug_name": "N-Acetylcysteine / Guaifenesin",
                "dosage": "600 mg oral",
                "frequency": "Twice daily (BID)",
                "duration": "7 Days",
                "indication": "Mucolytic expectorant to promote airway secretion clearance."
            })

        elif "tb" in diagnosis or "tuberculosis" in diagnosis:
            medications.append({
                "drug_name": "RIPE Anti-Tubercular Regimen (Rifampin + Isoniazid + Pyrazinamide + Ethambutol)",
                "dosage": "Weight-based standardized combination tablet",
                "frequency": "Once daily under DOTS (Directly Observed Therapy)",
                "duration": "2 Months Intensive Phase + 4 Months Continuation Phase",
                "indication": "Standard WHO first-line antitubercular therapy."
            })
            medications.append({
                "drug_name": "Pyridoxine (Vitamin B6)",
                "dosage": "25 mg - 50 mg oral",
                "frequency": "Once daily (QD)",
                "duration": "Duration of Isoniazid therapy",
                "indication": "Prevention of Isoniazid-induced peripheral neuropathy."
            })

        return {
            "medications": medications,
            "clinical_notes": f"Pharmacotherapy tailored for {severity} {finding.get('diagnosis')}. Verify renal function and allergy profile prior to administration.",
            "disclaimer": "DISCLAIMER: Automated recommendations generated for clinical decision support. Final prescribing rights rest with licensed physicians."
        }


class HealthMonitoringAgent:
    """Specialized Sub-Agent for Daily Health & Vital Signs Protocol."""

    @staticmethod
    def generate_monitoring_plan(finding: Dict[str, Any]) -> Dict[str, Any]:
        severity = finding.get("severity", "Mild")
        is_abnormal = finding.get("is_abnormal", False)

        if not is_abnormal:
            return {
                "vital_checks": [
                    {"parameter": "Body Temperature", "frequency": "Once Daily", "normal_range": "36.5°C - 37.5°C"},
                    {"parameter": "Pulse Oximetry (SpO2)", "frequency": "Optional / As needed", "normal_range": "95% - 100%"}
                ],
                "red_flags": ["Persistent cough > 2 weeks", "Unexplained fever", "Chest pain"],
                "follow_up_days": 14
            }

        freq_spo2 = "Every 4 Hours" if severity == "Severe" else ("Every 6 Hours" if severity == "Moderate" else "Twice Daily")
        
        return {
            "vital_checks": [
                {
                    "parameter": "Pulse Oximetry (SpO2)",
                    "frequency": freq_spo2,
                    "target_threshold": ">= 94% on Room Air",
                    "action_threshold": "< 92% requires immediate supplemental O2 and urgent medical evaluation"
                },
                {
                    "parameter": "Body Temperature",
                    "frequency": "Every 6 Hours",
                    "target_threshold": "< 37.5°C",
                    "action_threshold": "> 38.5°C refractory to antipyretics"
                },
                {
                    "parameter": "Respiratory Rate (BPM)",
                    "frequency": "Every 6 Hours",
                    "target_threshold": "12 - 20 breaths/min",
                    "action_threshold": "> 25 breaths/min (Tachypnea)"
                },
                {
                    "parameter": "Symptom Dyspnea Score",
                    "frequency": "Twice Daily",
                    "target_threshold": "Scale 0-10 (Mild breathlessness <= 3)",
                    "action_threshold": "Score >= 6 or sudden progression"
                }
            ],
            "red_flag_warning_signs": [
                "Oxygen saturation (SpO2) dropping below 92%",
                "Coughing up blood or rust-colored sputum (Hemoptysis)",
                "Inability to complete full sentences due to shortness of breath",
                "Confusion, altered mental state, or severe lethargy",
                "Pleuritic chest pain increasing sharply on deep inspiration"
            ],
            "follow_up_schedule": {
                "repeat_xray_days": 14 if severity == "Mild" else 7,
                "physician_reevaluation": "Within 48-72 hours for treatment response check"
            }
        }


class PulmonaryNutritionAgent:
    """Specialized Sub-Agent for Pulmonary Nutrition & Anti-Inflammatory Diet."""

    @staticmethod
    def generate_diet_plan(finding: Dict[str, Any]) -> Dict[str, Any]:
        severity = finding.get("severity", "Mild")
        is_abnormal = finding.get("is_abnormal", False)

        if not is_abnormal:
            return {
                "daily_hydration_liters": 2.5,
                "protein_target": "1.0g / kg body weight",
                "recommended_foods": ["Balanced Mediterranean diet", "Fresh fruits", "Whole grains"],
                "foods_to_avoid": ["Excess refined sugars", "Ultra-processed foods"],
                "supplements": ["Daily Multivitamin (Optional)"]
            }

        hydration = 3.5 if severity == "Severe" else 3.0
        
        return {
            "daily_hydration_liters": hydration,
            "hydration_guideline": "Consume warm fluids (herbal teas, warm water, broths) throughout the day to thin airway secretions and facilitate mucus clearance.",
            "protein_target": "1.2g - 1.5g / kg body weight daily (Essential for pulmonary tissue repair and immune cell synthesis)",
            "key_nutritional_pillars": [
                {
                    "pillar": "Anti-Inflammatory Omega-3 Fatty Acids",
                    "sources": ["Salmon", "Flaxseeds", "Chia seeds", "Walnuts", "Extra virgin olive oil"],
                    "benefit": "Reduces lung tissue inflammatory cytokine activity."
                },
                {
                    "pillar": "Antioxidants & Immune Support (Vitamin C & Zinc)",
                    "sources": ["Citrus fruits", "Bell peppers", "Berries", "Spinach", "Pumpkin seeds"],
                    "benefit": "Scavenges reactive oxygen species in inflamed pulmonary alveoli."
                },
                {
                    "pillar": "Vitamin D3 (Immune Regulation)",
                    "sources": ["Sunlight exposure", "Fortified eggs/milk", "Vitamin D3 2000 IU supplement"],
                    "benefit": "Enhances antimicrobial peptide expression in respiratory epithelium."
                }
            ],
            "foods_to_avoid": [
                "Excess dairy products if they exacerbate thick bronchial mucus",
                "High-sodium processed meals (causes fluid retention and respiratory burden)",
                "Carbonated beverages & gas-forming foods (increases abdominal pressure on diaphragm)",
                "Refined sugars (suppresses neutrophil phagocytic activity)"
            ],
            "sample_meal_plan": {
                "breakfast": "Oatmeal topped with walnuts, flaxseeds, and fresh blueberries + Warm lemon ginger tea",
                "lunch": "Grilled salmon or tofu salad with spinach, olive oil dressing, and quinoa",
                "afternoon_snack": "Bone broth or vegetable soup + Pumpkin seeds",
                "dinner": "Steamed chicken breast/lentils with garlic broccoli and sweet potato"
            }
        }


class MedicalAgenticPipeline:
    """Orchestrator for the Complete Healthcare Agentic AI Pipeline."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def run_pipeline(self, class_name: str, boxes: List[List[float]], confidences: List[float], image_size: int = 512) -> Dict[str, Any]:
        """Runs all specialized medical sub-agents and produces a comprehensive report."""
        logger.info(f"Running Agentic AI Pipeline for detected class '{class_name}' with {len(boxes)} bounding boxes...")

        # 1. Spatial & Severity Analysis
        spatial_analysis = XRayFindingExtractor.analyze_findings(class_name, boxes, confidences, image_size)

        # 2. Pharmacotherapy Agent
        prescription = PrescriptionAgent.generate_prescription(spatial_analysis)

        # 3. Health Monitoring Agent
        monitoring_plan = HealthMonitoringAgent.generate_monitoring_plan(spatial_analysis)

        # 4. Pulmonary Nutrition Agent
        diet_plan = PulmonaryNutritionAgent.generate_diet_plan(spatial_analysis)

        # Combine into complete agentic clinical report
        report = {
            "agentic_ai_version": "1.0.0-medical",
            "clinical_spatial_analysis": spatial_analysis,
            "prescription_recommendation": prescription,
            "health_monitoring_protocol": monitoring_plan,
            "pulmonary_nutrition_plan": diet_plan,
            "overall_summary": (
                f"Agentic AI analysis completed for {spatial_analysis['diagnosis']}. "
                f"Severity: {spatial_analysis['severity']}. Affected Regions: {', '.join(spatial_analysis['affected_regions'])}."
            )
        }

        logger.info("Agentic AI Medical Report successfully generated.")
        return report

