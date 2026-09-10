/* ─────────────────────────────────────────────────────────────
   Frontend Application JavaScript - PulmoVision AI Platform
   ───────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "http://127.0.0.1:8000/api";

    // DOM Elements
    const statusBadge = document.getElementById("statusBadge");
    const gpuBadge = document.getElementById("gpuBadge");
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const confThreshold = document.getElementById("confThreshold");
    const confVal = document.getElementById("confVal");
    const sampleGrid = document.getElementById("sampleGrid");
    const exportPdfBtn = document.getElementById("exportPdfBtn");

    // Results Elements
    const emptyState = document.getElementById("emptyState");
    const analysisContent = document.getElementById("analysisContent");
    const pnProbVal = document.getElementById("pnProbVal");
    const pnProbBar = document.getElementById("pnProbBar");
    const tbProbVal = document.getElementById("tbProbVal");
    const tbProbBar = document.getElementById("tbProbBar");
    const diagStatus = document.getElementById("diagStatus");
    const severityBadge = document.getElementById("severityBadge");
    const annotatedImage = document.getElementById("annotatedImage");
    const gradcamImage = document.getElementById("gradcamImage");
    const findingsContainer = document.getElementById("findingsContainer");
    const medicationGrid = document.getElementById("medicationGrid");
    const vitalsScheduleGrid = document.getElementById("vitalsScheduleGrid");
    const redFlagList = document.getElementById("redFlagList");
    const pillarsGrid = document.getElementById("pillarsGrid");
    const mealGrid = document.getElementById("mealGrid");

    let selectedFile = null;
    let selectedSamplePath = null;
    let activeRole = "doctor";

    // ─────────────────────────────────────────────────────────────
    // 1. Health Check & Initial Load
    // ─────────────────────────────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            const data = await res.json();
            if (statusBadge) {
                statusBadge.textContent = "Backend: Online";
                statusBadge.classList.remove("pulse");
                statusBadge.style.color = "#10b981";
            }

            if (gpuBadge) {
                if (data.gpu_name && data.gpu_name !== "N/A") {
                    gpuBadge.textContent = `GPU: ${data.gpu_name}`;
                    gpuBadge.style.color = "#10b981";
                } else {
                    gpuBadge.textContent = "Compute: CPU";
                    gpuBadge.style.color = "#94a3b8";
                }
            }
        } catch (e) {
            if (statusBadge) {
                statusBadge.textContent = "Backend: Offline";
                statusBadge.style.color = "#f43f5e";
            }
        }
    }

    async function loadSamples() {
        try {
            const res = await fetch(`${API_BASE}/samples`);
            const data = await res.json();
            sampleGrid.innerHTML = "";

            data.samples.forEach((s, idx) => {
                const card = document.createElement("div");
                card.className = "sample-card";
                card.innerHTML = `
                    <span>${s.name}</span>
                    <span class="sample-type ${s.type}">${s.type}</span>
                `;
                card.addEventListener("click", () => {
                    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
                    card.classList.add("active");
                    selectedSamplePath = s.path;
                    selectedFile = null;
                    analyzeBtn.disabled = false;
                    dropZone.querySelector(".primary-text").textContent = `Selected: ${s.name}`;
                    // Instant 1-click analysis on sample select
                    runInference();
                });
                sampleGrid.appendChild(card);
            });
        } catch (e) {
            sampleGrid.innerHTML = '<div class="sample-card">Failed to load samples</div>';
        }
    }

    checkHealth();
    loadSamples();

    // ─────────────────────────────────────────────────────────────
    // 2. Drag & Drop & Controls
    // ─────────────────────────────────────────────────────────────
    confThreshold.addEventListener("input", (e) => {
        confVal.textContent = `${e.target.value}%`;
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        selectedSamplePath = null;
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        dropZone.querySelector(".primary-text").textContent = `Selected: ${file.name}`;
        analyzeBtn.disabled = false;
    }

    // Role Switcher Buttons
    document.querySelectorAll(".role-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".role-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeRole = btn.dataset.role;
            applyRoleView();
        });
    });

    function applyRoleView() {
        const tabPrescription = document.querySelector('[data-tab="tab-prescription"]');
        const tabGradcam = document.querySelector('[data-tab="tab-gradcam"]');

        if (activeRole === "patient") {
            // Patient view defaults to 7-Day Vitals & Nutrition
            switchTab("tab-vitals");
        } else if (activeRole === "student") {
            // Radiology Fellow defaults to Grad-CAM Heatmaps
            switchTab("tab-gradcam");
        } else {
            // Doctor defaults to Clinical Findings & Prescriptions
            switchTab("tab-findings");
        }
    }

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(tab => {
        tab.addEventListener("click", () => {
            switchTab(tab.dataset.tab);
        });
    });

    function switchTab(tabId) {
        document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

        const targetBtn = document.querySelector(`[data-tab="${tabId}"]`);
        const targetPane = document.getElementById(tabId);
        if (targetBtn && targetPane) {
            targetBtn.classList.add("active");
            targetPane.classList.add("active");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 3. Inference Analysis Submission
    // ─────────────────────────────────────────────────────────────
    async function runInference() {
        if (!selectedFile && !selectedSamplePath) return;

        analyzeBtn.disabled = true;
        btnSpinner.classList.remove("hidden");

        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        } else {
            formData.append("sample_path", selectedSamplePath);
        }
        formData.append("confidence", (parseFloat(confThreshold.value) / 100).toString());

        try {
            const res = await fetch(`${API_BASE}/detect`, {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            if (data.status === "success") {
                renderResults(data);
                // Auto switch to Detailed Prescriptions tab if pathology detected
                if (data.primary_condition !== "Normal") {
                    switchTab("tab-prescription");
                }
            } else {
                alert("Analysis Error: " + (data.detail || "Unknown error"));
            }
        } catch (e) {
            alert("Failed to connect to backend server: " + e.message);
        } finally {
            analyzeBtn.disabled = false;
            btnSpinner.classList.add("hidden");
        }
    }

    analyzeBtn.addEventListener("click", runInference);

    // ─────────────────────────────────────────────────────────────
    // 4. Render Deep Learning & Medical Agent Payload
    // ─────────────────────────────────────────────────────────────
    function renderResults(data) {
        emptyState.classList.add("hidden");
        analysisContent.classList.remove("hidden");
        exportPdfBtn.classList.remove("hidden");

        // Probabilities & Diagnosis
        const pnPct = data.pneumonia_probability_percent;
        const tbPct = data.tuberculosis_probability_percent;
        pnProbVal.textContent = `${pnPct}%`;
        pnProbBar.style.width = `${pnPct}%`;
        tbProbVal.textContent = `${tbPct}%`;
        tbProbBar.style.width = `${tbPct}%`;

        diagStatus.textContent = data.primary_condition;
        const agentRep = data.medical_agent_report || {};
        const clinSummary = agentRep.clinical_summary || {};
        const spatialData = agentRep.clinical_spatial_analysis || agentRep.anatomical_findings || {};
        const rxData = agentRep.prescription_recommendation || agentRep.prescription || agentRep.prescriptions || {};
        const monitorData = agentRep.health_monitoring_protocol || agentRep.monitoring || agentRep.health_monitoring || {};
        const dietData = agentRep.pulmonary_nutrition_plan || agentRep.diet || agentRep.pulmonary_nutrition || {};

        if (data.primary_condition === "Normal") {
            severityBadge.textContent = "Normal / Healthy";
            severityBadge.style.background = "rgba(16, 185, 129, 0.2)";
            severityBadge.style.color = "#10b981";
        } else {
            const sev = spatialData.severity || spatialData.severity_grade || "Moderate";
            severityBadge.textContent = `High Risk (${sev})`;
            severityBadge.style.background = "rgba(244, 63, 94, 0.2)";
            severityBadge.style.color = "#f43f5e";
        }

        // Tab 1: AI Images & Narrative Clinical Summary
        annotatedImage.src = data.annotated_image_base64;
        gradcamImage.src = data.gradcam_heatmap_base64 || data.annotated_image_base64;

        const clinSummaryBox = document.getElementById("clinicalSummaryBox");
        if (clinSummaryBox) {
            const narrative = clinSummary.summary || clinSummary.impression || "AI visual findings evaluated.";
            const whoClass = clinSummary.who_classification || clinSummary.icd10_code || "";
            clinSummaryBox.innerHTML = `
                <div style="font-weight: 600; font-size: 14px; color: var(--primary-cyan); margin-bottom: 4px;">Radiological Impression & AI Narrative</div>
                <p style="font-size: 13px; color: var(--text-primary); line-height: 1.5;">${narrative}</p>
                ${whoClass ? `<div style="font-size: 12px; color: var(--primary-teal); margin-top: 6px;"><strong>Classification:</strong> ${whoClass}</div>` : ''}
            `;
        }

        findingsContainer.innerHTML = "";
        if (data.detections.length === 0) {
            findingsContainer.innerHTML = '<div class="finding-card">No localized high-confidence lesion bounding boxes detected above threshold.</div>';
        } else {
            data.detections.forEach(d => {
                const fCard = document.createElement("div");
                fCard.className = `finding-card ${d.disease}`;
                fCard.innerHTML = `
                    <strong>${d.disease} Detection</strong> (${d.confidence_percent}% confidence)
                    <br><span class="text-muted">Normalized Bounding Box Coordinates: [${d.bbox_normalized.join(", ")}]</span>
                `;
                findingsContainer.appendChild(fCard);
            });
        }

        // Tab 2: Detailed Evidence-Based Prescriptions
        const rxMetaSubtitle = document.getElementById("rxMetaSubtitle");
        if (rxMetaSubtitle) {
            rxMetaSubtitle.textContent = clinSummary.who_classification ? `Protocol: ${clinSummary.who_classification}` : "WHO / ATS / IDSA Standard Treatment Regimen";
        }

        const rxInfoBanner = document.getElementById("rxInfoBanner");
        if (rxInfoBanner) {
            const icd = clinSummary.icd10_code ? `ICD-10: <strong>${clinSummary.icd10_code}</strong> | ` : '';
            const urg = clinSummary.urgency ? `Urgency: <strong style="color: ${clinSummary.urgency === 'Emergency' ? '#f43f5e' : '#f59e0b'}">${clinSummary.urgency}</strong> | ` : '';
            const hosp = clinSummary.hospitalization_recommended ? '<strong style="color: #f43f5e">Hospitalization Recommended</strong>' : 'Outpatient Care Protocol';
            rxInfoBanner.innerHTML = `<div style="padding: 10px 14px; background: rgba(20,184,166,0.1); border: 1px solid var(--border-accent); border-radius: var(--radius-sm); font-size: 13px; margin-bottom: 16px;">${icd}${urg}${hosp}</div>`;
        }

        medicationGrid.innerHTML = "";
        const rxList = rxData.medications || [];
        if (rxList.length === 0) {
            medicationGrid.innerHTML = `<div class="med-card" style="grid-column: 1 / -1;"><div class="med-name">No Antimicrobial Pharmacotherapy Needed</div><div class="med-instructions">${rxData.clinical_notes || 'Patient chest X-ray does not indicate active consolidation or bacterial infection.'}</div></div>`;
        } else {
            rxList.forEach(m => {
                const mCard = document.createElement("div");
                mCard.className = "med-card";
                const drugTitle = m.drug_name || m.drug || "Medication";
                const dosage = m.dosage || "";
                const freq = m.frequency || "";
                const route = m.route || "Oral";
                const dur = m.duration || "";
                const indication = m.indication || m.instructions || m.clinical_rationale || "";
                const monitor = m.monitoring || "";

                mCard.innerHTML = `
                    <div class="med-name">${drugTitle} ${dosage ? `<span style="font-size: 13px; color: var(--text-primary); font-weight: 500;">— ${dosage}</span>` : ''}</div>
                    <div class="med-meta"><strong>Frequency:</strong> ${freq} | <strong>Route:</strong> ${route} | <strong>Duration:</strong> ${dur}</div>
                    ${indication ? `<div class="med-instructions" style="margin-top: 4px;"><strong>Indication & Action:</strong> ${indication}</div>` : ''}
                    ${monitor ? `<div class="med-instructions" style="margin-top: 4px; color: #fcd34d;"><strong>Monitoring & Safety:</strong> ${monitor}</div>` : ''}
                `;
                medicationGrid.appendChild(mCard);
            });
        }

        const contraList = document.getElementById("contraindicationsList");
        if (contraList) {
            contraList.innerHTML = "";
            const contras = rxData.contraindications_to_check || ["Verify allergy history before administration", "Check baseline renal and liver panel (LFT/eGFR)"];
            contras.forEach(c => {
                const li = document.createElement("li");
                li.style.fontSize = "13px";
                li.style.color = "var(--text-secondary)";
                li.style.marginBottom = "4px";
                li.textContent = c;
                contraList.appendChild(li);
            });
        }

        // Tab 3: Patient Recovery Dashboard (Vitals & Red Flags)
        redFlagList.innerHTML = "";
        const redFlags = monitorData.emergency_red_flags || monitorData.red_flag_warning_signs || [];
        redFlags.forEach(rf => {
            const li = document.createElement("li");
            li.style.fontSize = "13px";
            li.style.lineHeight = "1.4";
            li.textContent = rf;
            redFlagList.appendChild(li);
        });

        vitalsScheduleGrid.innerHTML = "";
        const vitalsChecks = monitorData.vital_checks || [];
        const vSched = monitorData.monitoring_schedule || [];
        
        if (vitalsChecks.length > 0) {
            vitalsChecks.forEach(vc => {
                const vCard = document.createElement("div");
                vCard.className = "med-card";
                vCard.innerHTML = `
                    <div class="med-name">📊 ${vc.parameter}</div>
                    <div class="med-meta">Frequency: <strong>${vc.frequency}</strong> | Target: <strong style="color: var(--primary-teal)">${vc.normal_range}</strong></div>
                    <div class="med-instructions" style="color: #f43f5e; margin-top: 4px;">Escalation Threshold: ${vc.action_threshold}</div>
                `;
                vitalsScheduleGrid.appendChild(vCard);
            });
        } else if (vSched.length > 0) {
            vSched.forEach(v => {
                const vCard = document.createElement("div");
                vCard.className = "med-card";
                const vCheck = Array.isArray(v.vitals_to_check) ? v.vitals_to_check.join(", ") : (v.vitals_to_check || '');
                vCard.innerHTML = `
                    <div class="med-name">Day ${v.day}: ${v.focus}</div>
                    <div class="med-meta">Check: ${vCheck}</div>
                    <div class="med-instructions">Action: ${v.action}</div>
                `;
                vitalsScheduleGrid.appendChild(vCard);
            });
        }

        const homeCareList = document.getElementById("homeCareList");
        if (homeCareList) {
            homeCareList.innerHTML = "";
            const hCare = monitorData.home_care_instructions || [
                "Maintain strict adherence to prescribed medication timing",
                "Rest with head elevated 30 degrees to optimize pulmonary expansion",
                "Log SpO2 and body temperature morning and evening",
                "Contact physician immediately if fever >38.5°C persists over 72 hours"
            ];
            hCare.forEach(hc => {
                const li = document.createElement("li");
                li.style.fontSize = "13px";
                li.style.color = "var(--text-secondary)";
                li.style.marginBottom = "4px";
                li.textContent = hc;
                homeCareList.appendChild(li);
            });
        }

        // Tab 4: Pulmonary Nutrition & Hydration
        const hydVal = dietData.hydration_target || (dietData.daily_hydration_liters ? `${dietData.daily_hydration_liters} Liters/day` : "3.0 Liters/day");
        document.getElementById("hydrationVal").textContent = hydVal;
        document.getElementById("proteinVal").textContent = dietData.protein_target || "1.2–1.5 g/kg/day";

        pillarsGrid.innerHTML = "";
        const pillars = dietData.key_nutritional_pillars || dietData.nutrition_pillars || [];
        pillars.forEach(p => {
            const pCard = document.createElement("div");
            pCard.className = "med-card";
            const sources = Array.isArray(p.sources) ? p.sources.join(", ") : (p.sources || '');
            pCard.innerHTML = `
                <div class="med-name">${p.pillar}</div>
                ${sources ? `<div class="med-meta">Food Sources: ${sources}</div>` : ''}
                <div class="med-instructions">${p.benefit || p.description || ''}</div>
            `;
            pillarsGrid.appendChild(pCard);
        });

        mealGrid.innerHTML = "";
        const mealPlan = dietData.sample_meal_plan || dietData.suggested_meal_plan || {};
        if (Array.isArray(mealPlan)) {
            mealPlan.forEach(m => {
                const mCard = document.createElement("div");
                mCard.className = "med-card";
                mCard.innerHTML = `
                    <div class="med-name">${m.meal}</div>
                    <div class="med-instructions">${Array.isArray(m.options) ? m.options.join(", ") : m.options}</div>
                `;
                mealGrid.appendChild(mCard);
            });
        } else if (typeof mealPlan === "object") {
            Object.entries(mealPlan).forEach(([mealName, options]) => {
                const mCard = document.createElement("div");
                mCard.className = "med-card";
                const optText = Array.isArray(options) ? options.join(" + ") : options;
                mCard.innerHTML = `
                    <div class="med-name">${mealName.toUpperCase()}</div>
                    <div class="med-instructions">${optText}</div>
                `;
                mealGrid.appendChild(mCard);
            });
        }
    }

    // Export PDF / Print
    exportPdfBtn.addEventListener("click", () => {
        window.print();
    });
});
