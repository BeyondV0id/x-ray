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
            statusBadge.textContent = "Backend: Online";
            statusBadge.classList.remove("pulse");
            statusBadge.style.color = "#10b981";

            if (data.gpu_name && data.gpu_name !== "N/A") {
                gpuBadge.textContent = `GPU: ${data.gpu_name}`;
                gpuBadge.style.color = "#10b981";
            } else {
                gpuBadge.textContent = "Compute: CPU";
                gpuBadge.style.color = "#94a3b8";
            }
        } catch (e) {
            statusBadge.textContent = "Backend: Offline";
            statusBadge.style.color = "#f43f5e";
        }
    }

    async function loadSamples() {
        try {
            const res = await fetch(`${API_BASE}/samples`);
            const data = await res.json();
            sampleGrid.innerHTML = "";

            data.samples.forEach(s => {
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
    analyzeBtn.addEventListener("click", async () => {
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
            } else {
                alert("Analysis Error: " + (data.detail || "Unknown error"));
            }
        } catch (e) {
            alert("Failed to connect to backend server: " + e.message);
        } finally {
            analyzeBtn.disabled = false;
            btnSpinner.classList.add("hidden");
        }
    });

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
        const agentRep = data.medical_agent_report;

        if (data.primary_condition === "Normal") {
            severityBadge.textContent = "Normal / Healthy";
            severityBadge.style.background = "rgba(16, 185, 129, 0.2)";
            severityBadge.style.color = "#10b981";
        } else {
            const sev = agentRep.anatomical_findings.severity_grade || "Moderate";
            severityBadge.textContent = `High Risk (${sev})`;
            severityBadge.style.background = "rgba(244, 63, 94, 0.2)";
            severityBadge.style.color = "#f43f5e";
        }

        // Tab 1 & 2: Images & Findings
        annotatedImage.src = data.annotated_image_base64;
        gradcamImage.src = data.gradcam_heatmap_base64 || data.annotated_image_base64;

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

        // Tab 3: Prescriptions
        medicationGrid.innerHTML = "";
        const rxList = agentRep.prescriptions.medications || [];
        rxList.forEach(m => {
            const mCard = document.createElement("div");
            mCard.className = "med-card";
            mCard.innerHTML = `
                <div class="med-name">${m.drug} (${m.dosage})</div>
                <div class="med-meta">Frequency: ${m.frequency} | Duration: ${m.duration}</div>
                <div class="med-instructions">Indication & Notes: ${m.instructions}</div>
            `;
            medicationGrid.appendChild(mCard);
        });

        // Tab 4: Vitals Protocol & Red Flags
        redFlagList.innerHTML = "";
        const redFlags = agentRep.health_monitoring.emergency_red_flags || [];
        redFlags.forEach(rf => {
            const li = document.createElement("li");
            li.textContent = rf;
            redFlagList.appendChild(li);
        });

        vitalsScheduleGrid.innerHTML = "";
        const vSched = agentRep.health_monitoring.monitoring_schedule || [];
        vSched.forEach(v => {
            const vCard = document.createElement("div");
            vCard.className = "med-card";
            vCard.innerHTML = `
                <div class="med-name">Day ${v.day}: ${v.focus}</div>
                <div class="med-meta">Check: ${v.vitals_to_check.join(", ")}</div>
                <div class="med-instructions">Instructions: ${v.action}</div>
            `;
            vitalsScheduleGrid.appendChild(vCard);
        });

        // Tab 5: Nutrition & Hydration
        document.getElementById("hydrationVal").textContent = agentRep.pulmonary_nutrition.hydration_target || "3.0 Liters/day";
        document.getElementById("proteinVal").textContent = agentRep.pulmonary_nutrition.protein_target || "1.2-1.5 g/kg/day";

        pillarsGrid.innerHTML = "";
        const pillars = agentRep.pulmonary_nutrition.nutrition_pillars || [];
        pillars.forEach(p => {
            const pCard = document.createElement("div");
            pCard.className = "med-card";
            pCard.innerHTML = `
                <div class="med-name">${p.pillar}</div>
                <div class="med-instructions">${p.description}</div>
            `;
            pillarsGrid.appendChild(pCard);
        });

        mealGrid.innerHTML = "";
        const meals = agentRep.pulmonary_nutrition.suggested_meal_plan || [];
        meals.forEach(m => {
            const mCard = document.createElement("div");
            mCard.className = "med-card";
            mCard.innerHTML = `
                <div class="med-name">${m.meal}</div>
                <div class="med-instructions">${m.options.join(", ")}</div>
            `;
            mealGrid.appendChild(mCard);
        });

        applyRoleView();
    }

    // Export PDF / Print
    exportPdfBtn.addEventListener("click", () => {
        window.print();
    });
});
