document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "http://localhost:8000";

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const confThreshold = document.getElementById("confThreshold");
    const confVal = document.getElementById("confVal");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnSpinner = document.getElementById("btnSpinner");

    const statusBadge = document.getElementById("statusBadge");
    const gpuBadge = document.getElementById("gpuBadge");
    const medicalDisclaimer = document.getElementById("medicalDisclaimer");

    const placeholderState = document.getElementById("placeholderState");
    const activeResults = document.getElementById("activeResults");
    const annotatedImage = document.getElementById("annotatedImage");
    const detectionCountBadge = document.getElementById("detectionCountBadge");
    const findingsContainer = document.getElementById("findingsContainer");

    let selectedFile = null;

    // 1. Backend Health Check
    async function checkBackendHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();

            if (data.status === "online") {
                statusBadge.textContent = "TF Online";
                statusBadge.classList.add("online");
            } else {
                statusBadge.textContent = "Offline";
            }

            if (data.gpu_acceleration) {
                gpuBadge.textContent = "GPU Active";
            } else {
                gpuBadge.textContent = "CPU Mode";
            }

            if (data.disclaimer) {
                medicalDisclaimer.innerHTML = `<strong>Medical Research Disclaimer:</strong> ${data.disclaimer}`;
            }
        } catch (e) {
            statusBadge.textContent = "Offline";
            gpuBadge.textContent = "Server Disconnected";
            console.warn("Backend connection check failed:", e);
        }
    }

    checkBackendHealth();

    // 2. Confidence Slider Sync
    confThreshold.addEventListener("input", (e) => {
        confVal.textContent = `${e.target.value}%`;
    });

    // 3. File Input & Drag and Drop Handling
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        const dropText = dropZone.querySelector(".primary-text");
        dropText.textContent = `Selected: ${file.name}`;
        analyzeBtn.disabled = false;
    }

    // 4. Inference API Call
    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        btnSpinner.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", selectedFile);

        const confValDecimal = parseFloat(confThreshold.value) / 100.0;

        try {
            const res = await fetch(`${API_BASE}/api/detect?confidence=${confValDecimal}`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(`Server returned HTTP ${res.status}`);
            }

            const data = await res.json();
            renderResults(data);
        } catch (err) {
            alert(`Inference failed: ${err.message}. Make sure backend (backend/app.py) is running.`);
            console.error(err);
        } finally {
            analyzeBtn.disabled = false;
            btnSpinner.classList.add("hidden");
        }
    });

    function renderResults(data) {
        placeholderState.classList.add("hidden");
        activeResults.classList.remove("hidden");

        // Display annotated base64 image
        annotatedImage.src = data.annotated_image_base64;
        detectionCountBadge.textContent = `${data.detection_count} Finding(s)`;

        // Clear previous cards
        findingsContainer.innerHTML = "";

        if (data.detection_count === 0) {
            findingsContainer.innerHTML = `
                <div class="card-detection">
                    <div class="disease-title">No Abnormalities Detected</div>
                    <div class="location-tag">No regions exceeded the ${confThreshold.value}% confidence threshold.</div>
                </div>
            `;
            return;
        }

        data.detections.forEach((det, idx) => {
            const isPneu = det.disease_raw === "Pneumonia";
            const cardClass = isPneu ? "pneumonia" : "tuberculosis";

            const cardHtml = `
                <div class="card-detection ${cardClass}">
                    <div class="card-header">
                        <span class="disease-title">${det.disease}</span>
                        <span class="conf-score">${det.confidence_percent}%</span>
                    </div>
                    <div class="location-tag">📍 ${det.anatomical_location}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${det.confidence_percent}%;"></div>
                    </div>
                </div>
            `;
            findingsContainer.insertAdjacentHTML("beforeend", cardHtml);
        });
    }
});
