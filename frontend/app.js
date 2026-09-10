document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "http://localhost:8000";

    // ===== DOM References (existing) =====
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

    // ===== DOM References (clinical workflow) =====
    const clinicalSection = document.getElementById("clinicalSection");
    const severityBadge = document.getElementById("severityBadge");
    const severitySummary = document.getElementById("severitySummary");
    const prescriptionContent = document.getElementById("prescriptionContent");
    const rxRoleIndicator = document.getElementById("rxRoleIndicator");
    const rxTitle = document.getElementById("rxTitle");
    const fuTitle = document.getElementById("fuTitle");
    const followupTimeline = document.getElementById("followupTimeline");
    const clinicalNotesPanel = document.getElementById("clinicalNotesPanel");
    const lifestylePanel = document.getElementById("lifestylePanel");
    const lifestyleContent = document.getElementById("lifestyleContent");
    const dietPanel = document.getElementById("dietPanel");
    const dietContent = document.getElementById("dietContent");
    const educationPanel = document.getElementById("educationPanel");
    const educationContent = document.getElementById("educationContent");
    const generateReportBtn = document.getElementById("generateReportBtn");
    const printReportBtn = document.getElementById("printReportBtn");

    // ===== State =====
    let selectedFile = null;
    let currentRole = "doctor";
    let lastAnalysisData = null;

    // =========================================================================
    //  PRESCRIPTION & DIET DATABASE
    // =========================================================================
    const PRESCRIPTION_DB = {
        "Pneumonia": {
            severity(confidence) {
                if (confidence >= 85) return { level: "Severe", color: "#ef4444", icon: "\u{1F534}", cssClass: "severe" };
                if (confidence >= 65) return { level: "Moderate", color: "#f59e0b", icon: "\u{1F7E1}", cssClass: "moderate" };
                return { level: "Mild", color: "#10b981", icon: "\u{1F7E2}", cssClass: "mild" };
            },
            medications: {
                Mild: [
                    { name: "Amoxicillin", dose: "500mg", route: "Oral", frequency: "Three times daily", duration: "5-7 days", cls: "Antibiotic", clsColor: "#3b82f6", instruction: "Take with or after food. Complete the full course even if you feel better." },
                    { name: "Paracetamol", dose: "500-1000mg", route: "Oral", frequency: "Every 4-6 hrs (PRN)", duration: "As needed", cls: "Antipyretic", clsColor: "#8b5cf6", instruction: "Take for fever or body pain. Do not exceed 4g per day." },
                    { name: "Guaifenesin", dose: "200-400mg", route: "Oral", frequency: "Every 4 hours", duration: "As needed", cls: "Expectorant", clsColor: "#06b6d4", instruction: "Helps loosen mucus. Drink plenty of water." }
                ],
                Moderate: [
                    { name: "Amoxicillin-Clavulanate", dose: "625mg", route: "Oral", frequency: "Three times daily", duration: "7-10 days", cls: "Antibiotic", clsColor: "#3b82f6", instruction: "Take at the start of a meal. Complete the full course." },
                    { name: "Azithromycin", dose: "500mg D1, 250mg D2-5", route: "Oral", frequency: "Once daily", duration: "5 days", cls: "Antibiotic", clsColor: "#3b82f6", instruction: "Take 1 hour before or 2 hours after meals." },
                    { name: "Paracetamol", dose: "500-1000mg", route: "Oral", frequency: "Every 4-6 hrs (PRN)", duration: "As needed", cls: "Antipyretic", clsColor: "#8b5cf6", instruction: "For fever and pain management." },
                    { name: "Salbutamol Inhaler", dose: "2 puffs (100mcg/puff)", route: "Inhalation", frequency: "Every 4-6 hrs (PRN)", duration: "As needed", cls: "Bronchodilator", clsColor: "#06b6d4", instruction: "Use spacer for better delivery. Shake well before use." }
                ],
                Severe: [
                    { name: "Ceftriaxone", dose: "1-2g", route: "Intravenous", frequency: "Once daily", duration: "7-14 days", cls: "Antibiotic", clsColor: "#3b82f6", instruction: "Administered by healthcare professional in hospital." },
                    { name: "Azithromycin", dose: "500mg", route: "IV \u2192 Oral", frequency: "Once daily", duration: "5-7 days", cls: "Antibiotic", clsColor: "#3b82f6", instruction: "Transition to oral when clinically stable." },
                    { name: "Oxygen Therapy", dose: "2-6 L/min", route: "Nasal Cannula", frequency: "Continuous", duration: "Until SpO\u2082 > 94%", cls: "Supportive", clsColor: "#10b981", instruction: "Maintain oxygen saturation above 94%." },
                    { name: "IV Normal Saline", dose: "1000ml", route: "Intravenous", frequency: "Per hydration needs", duration: "As needed", cls: "Supportive", clsColor: "#10b981", instruction: "For fluid resuscitation and hydration." },
                    { name: "Paracetamol", dose: "1000mg", route: "IV/Oral", frequency: "Every 6 hours", duration: "As needed", cls: "Antipyretic", clsColor: "#8b5cf6", instruction: "For high fever management." }
                ]
            },
            follow_up: [
                { time: "48-72 Hours", action: "Clinical reassessment \u2014 vitals, O\u2082 saturation, symptom progression", icon: "\u{1F504}", urgent: true },
                { time: "1 Week", action: "Symptom review \u2014 assess antibiotic response and improvement", icon: "\u{1F4CB}", urgent: false },
                { time: "2 Weeks", action: "Follow-up consultation \u2014 review lab results if ordered", icon: "\u{1FA7A}", urgent: false },
                { time: "4-6 Weeks", action: "Follow-up chest X-ray \u2014 confirm infiltrate resolution", icon: "\u{1F4F8}", urgent: false }
            ],
            patient_info: {
                simple_explanation: "Pneumonia is a lung infection that fills the air sacs with fluid or pus, making breathing difficult. It causes cough, fever, and chest pain. With proper treatment, most people recover fully.",
                lifestyle: [
                    "\u{1F6CF}\uFE0F Get plenty of rest \u2014 your body needs energy to fight the infection",
                    "\u{1F4A7} Drink lots of fluids (water, warm soups, herbal teas) to stay hydrated",
                    "\u{1F48A} Take ALL prescribed antibiotics even if you feel better \u2014 stopping early causes resistance",
                    "\u{1F6AD} Avoid smoking and secondhand smoke completely",
                    "\u{1F34E} Eat nutritious foods to support your immune system",
                    "\u{1F637} Cover your mouth when coughing to prevent spreading",
                    "\u{1F321}\uFE0F Monitor your temperature \u2014 fever above 103\u00B0F needs urgent attention"
                ],
                warning_signs: [
                    "Difficulty breathing or shortness of breath getting worse",
                    "High fever (above 103\u00B0F / 39.4\u00B0C) not responding to medication",
                    "Chest pain that worsens with breathing",
                    "Coughing up blood",
                    "Confusion or disorientation",
                    "Bluish color of lips or fingernails"
                ]
            },
            diet: {
                overview: "A nutrient-rich diet is crucial for pneumonia recovery. Focus on high-protein foods to repair lung tissue, vitamin C for immune function, and adequate hydration to thin mucus.",
                categories: [
                    {
                        title: "\u2705 Recommended Foods",
                        icon: "\u{1F966}",
                        type: "recommend",
                        items: [
                            "Lean proteins \u2014 chicken soup, eggs, fish, lentils (dal), paneer",
                            "Vitamin C rich \u2014 oranges, lemons, amla, guava, bell peppers",
                            "Warm fluids \u2014 chicken broth, vegetable soup, ginger tea, turmeric milk",
                            "Whole grains \u2014 oats, brown rice, whole wheat roti",
                            "Probiotic foods \u2014 yogurt (curd), buttermilk to restore gut flora during antibiotics",
                            "Zinc-rich foods \u2014 pumpkin seeds, chickpeas, nuts, spinach"
                        ]
                    },
                    {
                        title: "\u274C Foods to Avoid",
                        icon: "\u{1F6AB}",
                        type: "avoid",
                        items: [
                            "Cold beverages and ice cream \u2014 can irritate throat",
                            "Processed & fried foods \u2014 increase inflammation",
                            "Excessive dairy (if causing mucus) \u2014 milk, cheese in large amounts",
                            "Sugary foods \u2014 weaken immune response",
                            "Alcohol \u2014 dehydrates and interacts with antibiotics",
                            "Caffeinated drinks \u2014 can lead to dehydration"
                        ]
                    }
                ],
                meal_plan: [
                    { time: "Morning", desc: "Warm turmeric milk + oatmeal with honey and nuts" },
                    { time: "Mid-Morning", desc: "Fresh orange juice or amla juice + 2 boiled eggs" },
                    { time: "Lunch", desc: "Chicken soup / dal + brown rice + steamed vegetables + curd" },
                    { time: "Snack", desc: "Warm ginger-lemon tea + handful of pumpkin seeds" },
                    { time: "Dinner", desc: "Grilled fish / paneer + whole wheat roti + vegetable stew" },
                    { time: "Bedtime", desc: "Warm milk with a pinch of turmeric and pepper" }
                ],
                key_nutrients: [
                    { name: "Vitamin C", highlight: true },
                    { name: "Zinc", highlight: true },
                    { name: "Protein", highlight: true },
                    { name: "Vitamin A", highlight: false },
                    { name: "Omega-3", highlight: false },
                    { name: "Probiotics", highlight: true },
                    { name: "Iron", highlight: false }
                ]
            },
            educational: {
                definition: "Pneumonia is an infection of the lung parenchyma caused by bacteria, viruses, or fungi, leading to inflammation and consolidation of the alveolar spaces.",
                pathophysiology: "Infectious agents reach the alveoli via inhalation or aspiration, triggering an inflammatory cascade. Neutrophils and exudate fill the alveoli, impairing gas exchange. Stages: Congestion \u2192 Red hepatization \u2192 Grey hepatization \u2192 Resolution.",
                risk_factors: ["Age > 65 or < 5 years", "Immunocompromised states (HIV, chemotherapy)", "Chronic lung disease (COPD, asthma)", "Smoking history", "Recent viral upper respiratory infection", "Aspiration risk (dysphagia, GERD)", "Hospitalization / mechanical ventilation"],
                key_signs: ["Fever with productive cough (purulent sputum)", "Tachypnea and dyspnea", "Crackles / rales on auscultation", "Dullness to percussion over affected area", "Decreased breath sounds", "Pleuritic chest pain", "Elevated WBC count with left shift"],
                differentials: ["Pulmonary edema (cardiogenic)", "Pulmonary embolism", "Lung malignancy", "Tuberculosis", "Organizing pneumonia (COP)"],
                references: [
                    "Harrison\u2019s Principles of Internal Medicine, Ch. 121 \u2014 Pneumonia",
                    "IDSA/ATS Guidelines for CAP Management (2019)",
                    "Radiopaedia: Pneumonia imaging patterns",
                    "UpToDate: Community-acquired pneumonia in adults"
                ]
            }
        },

        "Tuberculosis": {
            severity(confidence) {
                if (confidence >= 85) return { level: "Advanced", color: "#ef4444", icon: "\u{1F534}", cssClass: "severe" };
                if (confidence >= 65) return { level: "Moderate", color: "#f59e0b", icon: "\u{1F7E1}", cssClass: "moderate" };
                return { level: "Early/Suspected", color: "#10b981", icon: "\u{1F7E2}", cssClass: "mild" };
            },
            medications: {
                "Early/Suspected": [
                    { name: "Isoniazid (H)", dose: "5 mg/kg (max 300mg)", route: "Oral", frequency: "Once daily", duration: "6 months", cls: "Anti-TB", clsColor: "#f59e0b", instruction: "Take on empty stomach, 1 hour before meals." },
                    { name: "Rifampicin (R)", dose: "10 mg/kg (max 600mg)", route: "Oral", frequency: "Once daily", duration: "6 months", cls: "Anti-TB", clsColor: "#f59e0b", instruction: "Take on empty stomach. Urine may turn orange-red \u2014 this is normal." },
                    { name: "Pyridoxine (Vit B6)", dose: "25mg", route: "Oral", frequency: "Once daily", duration: "6 months", cls: "Supplement", clsColor: "#8b5cf6", instruction: "Prevents nerve damage from Isoniazid." }
                ],
                Moderate: [
                    { name: "Isoniazid (H)", dose: "5 mg/kg (max 300mg)", route: "Oral", frequency: "Once daily", duration: "2mo intensive + 4mo continuation", cls: "Anti-TB (DOTS)", clsColor: "#f59e0b", instruction: "Take on empty stomach under DOTS supervision." },
                    { name: "Rifampicin (R)", dose: "10 mg/kg (max 600mg)", route: "Oral", frequency: "Once daily", duration: "2mo intensive + 4mo continuation", cls: "Anti-TB (DOTS)", clsColor: "#f59e0b", instruction: "Take on empty stomach. Orange urine is expected." },
                    { name: "Pyrazinamide (Z)", dose: "25 mg/kg", route: "Oral", frequency: "Once daily", duration: "2 months (intensive)", cls: "Anti-TB (DOTS)", clsColor: "#f59e0b", instruction: "Take with meals if stomach upset occurs." },
                    { name: "Ethambutol (E)", dose: "15 mg/kg", route: "Oral", frequency: "Once daily", duration: "2 months (intensive)", cls: "Anti-TB (DOTS)", clsColor: "#f59e0b", instruction: "Report any vision changes immediately." },
                    { name: "Pyridoxine (Vit B6)", dose: "25mg", route: "Oral", frequency: "Once daily", duration: "6 months", cls: "Supplement", clsColor: "#8b5cf6", instruction: "Prevents peripheral neuropathy." }
                ],
                Advanced: [
                    { name: "Isoniazid (H)", dose: "5 mg/kg (max 300mg)", route: "Oral", frequency: "Once daily", duration: "2+4 months (DOTS supervised)", cls: "Anti-TB (DOTS)", clsColor: "#ef4444", instruction: "Strict DOTS compliance essential." },
                    { name: "Rifampicin (R)", dose: "10 mg/kg (max 600mg)", route: "Oral", frequency: "Once daily", duration: "2+4 months (DOTS supervised)", cls: "Anti-TB (DOTS)", clsColor: "#ef4444", instruction: "Take on empty stomach." },
                    { name: "Pyrazinamide (Z)", dose: "25 mg/kg", route: "Oral", frequency: "Once daily", duration: "2 months intensive", cls: "Anti-TB (DOTS)", clsColor: "#ef4444", instruction: "Monitor uric acid levels." },
                    { name: "Ethambutol (E)", dose: "15 mg/kg", route: "Oral", frequency: "Once daily", duration: "2 months intensive", cls: "Anti-TB (DOTS)", clsColor: "#ef4444", instruction: "Monthly eye exam recommended." },
                    { name: "Streptomycin", dose: "15 mg/kg", route: "Intramuscular", frequency: "Once daily", duration: "2 months (if resistance suspected)", cls: "Anti-TB", clsColor: "#ef4444", instruction: "Administered by healthcare worker. Report hearing changes." },
                    { name: "Pyridoxine (Vit B6)", dose: "50mg", route: "Oral", frequency: "Once daily", duration: "Throughout treatment", cls: "Supplement", clsColor: "#8b5cf6", instruction: "Higher dose due to advanced disease neuropathy risk." }
                ]
            },
            follow_up: [
                { time: "2 Weeks", action: "Sputum AFB smear \u2014 baseline comparison and monitoring", icon: "\u{1F52C}", urgent: true },
                { time: "2 Months", action: "End of intensive phase \u2014 sputum culture, LFTs, transition to continuation phase", icon: "\u{1F9EA}", urgent: true },
                { time: "3 Months", action: "Chest X-ray \u2014 assess radiological improvement", icon: "\u{1F4F8}", urgent: false },
                { time: "5 Months", action: "Sputum re-examination \u2014 confirm culture conversion", icon: "\u{1F52C}", urgent: false },
                { time: "6 Months", action: "Treatment completion \u2014 final sputum, final X-ray, outcome assessment", icon: "\u2705", urgent: true }
            ],
            patient_info: {
                simple_explanation: "Tuberculosis (TB) is a bacterial infection that mainly affects your lungs. The bacteria spread through the air when an infected person coughs. With proper treatment for 6 months, TB can be completely cured.",
                lifestyle: [
                    "\u{1F48A} NEVER miss or skip your TB medications \u2014 incomplete treatment leads to drug-resistant TB",
                    "\u{1F3E5} Follow the DOTS program \u2014 take medicines under direct supervision as advised",
                    "\u{1F35D} Eat a high-protein, nutritious diet (eggs, dal, milk, fruits, vegetables)",
                    "\u{1F637} Cover your mouth when coughing; use a mask in crowded places for first 2-3 weeks",
                    "\u{1F32C}\uFE0F Keep your living space well-ventilated with open windows",
                    "\u{1F6AD} Completely avoid smoking and alcohol during treatment",
                    "\u{1F6CF}\uFE0F Get adequate rest, especially during the intensive phase",
                    "\u{1F4A7} Stay well hydrated \u2014 drink at least 8-10 glasses of water daily"
                ],
                warning_signs: [
                    "Coughing up blood (hemoptysis)",
                    "Persistent high fever not responding to medication",
                    "Severe weight loss or loss of appetite",
                    "Yellowing of skin or eyes (possible liver toxicity from drugs)",
                    "Vision changes (if on Ethambutol)",
                    "Numbness or tingling in hands/feet",
                    "Severe nausea or vomiting preventing oral medication"
                ]
            },
            diet: {
                overview: "TB patients require a high-calorie, high-protein diet to combat weight loss and support immune recovery. Anti-TB drugs can cause appetite loss and liver strain, making nutrition critical.",
                categories: [
                    {
                        title: "\u2705 Recommended Foods",
                        icon: "\u{1F966}",
                        type: "recommend",
                        items: [
                            "High-protein \u2014 eggs (3-4/day), chicken, fish, paneer, soy, dal, rajma",
                            "Calorie-dense \u2014 ghee, peanut butter, bananas, dry fruits, milkshakes",
                            "Iron-rich \u2014 spinach, beetroot, pomegranate, jaggery, red meat",
                            "Vitamin A & D \u2014 carrots, sweet potatoes, fortified milk, sunlight exposure",
                            "Whole grains \u2014 ragi, jowar, wheat, brown rice for sustained energy",
                            "Antioxidant-rich \u2014 green tea, turmeric, garlic, ginger, berries"
                        ]
                    },
                    {
                        title: "\u274C Foods to Avoid",
                        icon: "\u{1F6AB}",
                        type: "avoid",
                        items: [
                            "Alcohol \u2014 STRICTLY PROHIBITED (liver toxicity with Isoniazid & Rifampicin)",
                            "Tobacco & smoking \u2014 damages lungs and slows recovery",
                            "Refined sugar & white bread \u2014 empty calories, weakens immunity",
                            "Trans fats & deep fried foods \u2014 increase inflammation",
                            "Canned/processed foods \u2014 high sodium, low nutrition",
                            "Excessive tea/coffee with meals \u2014 reduces iron absorption"
                        ]
                    }
                ],
                meal_plan: [
                    { time: "Early AM", desc: "Soaked almonds (5-6) + 1 glass warm milk with turmeric" },
                    { time: "Breakfast", desc: "3 egg omelette + whole wheat toast + fresh fruit juice" },
                    { time: "Mid-Morning", desc: "Banana milkshake with peanut butter + handful of dry fruits" },
                    { time: "Lunch", desc: "Dal + rice + chicken/fish curry + green salad + curd" },
                    { time: "Snack", desc: "Sprouted chana chaat + pomegranate + green tea" },
                    { time: "Dinner", desc: "Paneer/soy curry + roti + spinach sabzi + buttermilk" },
                    { time: "Bedtime", desc: "Warm milk with honey and a pinch of turmeric" }
                ],
                key_nutrients: [
                    { name: "Protein", highlight: true },
                    { name: "Calories", highlight: true },
                    { name: "Iron", highlight: true },
                    { name: "Vitamin A", highlight: true },
                    { name: "Vitamin D", highlight: false },
                    { name: "Zinc", highlight: false },
                    { name: "B-Complex", highlight: true },
                    { name: "Antioxidants", highlight: false }
                ]
            },
            educational: {
                definition: "Tuberculosis (TB) is a chronic granulomatous infectious disease caused by Mycobacterium tuberculosis, primarily affecting the lungs but capable of disseminating to virtually any organ system.",
                pathophysiology: "M. tuberculosis is transmitted via airborne droplet nuclei (1-5\u03BCm). Upon reaching the alveoli, bacilli are phagocytosed by alveolar macrophages. Cell-mediated immunity leads to granuloma formation with central caseating necrosis (Ghon focus). The Ghon complex (Ghon focus + draining lymph node) represents primary TB. Reactivation occurs when immune surveillance weakens.",
                risk_factors: ["HIV/AIDS co-infection (strongest risk factor)", "Immunosuppressive therapy", "Close contact with active TB case", "Malnutrition and poverty", "Diabetes mellitus", "Silicosis", "Healthcare workers", "Crowded living conditions", "Substance abuse"],
                key_signs: ["Chronic cough > 2 weeks (most important symptom)", "Hemoptysis", "Night sweats", "Significant weight loss / anorexia", "Low-grade evening rise of fever", "Fatigue and malaise", "Upper lobe cavitary lesions on CXR"],
                differentials: ["Lung malignancy (especially upper lobe mass)", "Fungal infection (Histoplasmosis, Aspergillosis)", "Non-tuberculous mycobacterial infection", "Sarcoidosis", "Lung abscess", "Community-acquired pneumonia"],
                references: [
                    "Harrison\u2019s Principles of Internal Medicine, Ch. 173 \u2014 Tuberculosis",
                    "WHO End TB Strategy (2015) & Updated Guidelines",
                    "RNTCP (Revised National TB Control Programme) Guidelines",
                    "Radiopaedia: Pulmonary TB imaging features",
                    "Robbins & Cotran Pathologic Basis of Disease \u2014 Granulomatous Inflammation"
                ]
            }
        }
    };

    // =========================================================================
    //  1. BACKEND HEALTH CHECK
    // =========================================================================
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

    // =========================================================================
    //  2. CONFIDENCE SLIDER
    // =========================================================================
    confThreshold.addEventListener("input", (e) => {
        confVal.textContent = `${e.target.value}%`;
    });

    // =========================================================================
    //  3. FILE INPUT & DRAG-DROP
    // =========================================================================
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

    // =========================================================================
    //  4. ROLE MANAGEMENT
    // =========================================================================
    document.querySelectorAll(".role-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".role-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentRole = btn.dataset.role;
            if (lastAnalysisData && lastAnalysisData.detection_count > 0) {
                renderClinicalWorkflow(lastAnalysisData);
            }
        });
    });

    // =========================================================================
    //  5. INFERENCE API CALL
    // =========================================================================
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

    // =========================================================================
    //  6. RESULTS RENDERING (original + clinical workflow trigger)
    // =========================================================================
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
            clinicalSection.classList.add("hidden");
            lastAnalysisData = data;
            return;
        }

        data.detections.forEach((det) => {
            const isPneu = det.disease_raw === "Pneumonia";
            const cardClass = isPneu ? "pneumonia" : "tuberculosis";

            const cardHtml = `
                <div class="card-detection ${cardClass}">
                    <div class="card-header">
                        <span class="disease-title">${det.disease}</span>
                        <span class="conf-score">${det.confidence_percent}%</span>
                    </div>
                    <div class="location-tag">\u{1F4CD} ${det.anatomical_location}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${det.confidence_percent}%;"></div>
                    </div>
                </div>
            `;
            findingsContainer.insertAdjacentHTML("beforeend", cardHtml);
        });

        lastAnalysisData = data;
        renderClinicalWorkflow(data);
    }

    // =========================================================================
    //  7. CLINICAL WORKFLOW ENGINE
    // =========================================================================
    function renderClinicalWorkflow(data) {
        clinicalSection.classList.remove("hidden");

        // Aggregate unique diseases with highest confidence
        const diseaseMap = {};
        data.detections.forEach(det => {
            const key = det.disease_raw;
            if (!diseaseMap[key] || det.confidence_percent > diseaseMap[key].confidence_percent) {
                diseaseMap[key] = det;
            }
        });

        const diseases = Object.values(diseaseMap);
        const primaryDetection = diseases.reduce((a, b) =>
            a.confidence_percent > b.confidence_percent ? a : b
        );
        const primaryDb = PRESCRIPTION_DB[primaryDetection.disease_raw];
        if (!primaryDb) return;

        const primarySeverity = primaryDb.severity(primaryDetection.confidence_percent);

        // Render all sections
        renderSeverity(primaryDetection, primarySeverity, data.detection_count);
        renderPrescriptions(diseases);
        renderFollowUp(diseases);
        renderDiet(diseases);

        // Role-specific panels
        clinicalNotesPanel.classList.toggle("hidden", currentRole !== "doctor");
        lifestylePanel.classList.toggle("hidden", currentRole !== "patient");
        educationPanel.classList.toggle("hidden", currentRole !== "student");

        // Diet panel visible for all roles
        dietPanel.classList.remove("hidden");

        if (currentRole === "patient") {
            renderPatientInfo(diseases);
        }
        if (currentRole === "student") {
            renderEducation(diseases);
        }

        updateRoleUI();
    }

    // ---- Severity ----
    function renderSeverity(detection, severity, totalCount) {
        severityBadge.textContent = `${severity.icon} ${severity.level}`;
        severityBadge.className = `severity-badge ${severity.cssClass}`;

        const roleContextMap = {
            doctor: `<strong>Primary Finding:</strong> ${detection.disease} at <strong>${detection.confidence_percent}%</strong> confidence.<br>
                     <strong>Severity Assessment:</strong> ${severity.level} \u2014 ${totalCount} region(s) of concern identified.<br>
                     <strong>Location:</strong> ${detection.anatomical_location}.<br>
                     <em>Severity estimated from AI confidence level. Correlate with clinical presentation.</em>`,
            patient: `Our AI analysis found signs of <strong>${detection.disease_raw}</strong> in your X-ray with <strong>${detection.confidence_percent}%</strong> confidence.<br>
                      The condition appears to be <strong>${severity.level}</strong>.<br>
                      <em>Please consult your doctor for proper diagnosis and treatment.</em>`,
            student: `<strong>AI Detection:</strong> ${detection.disease} \u2014 ${detection.confidence_percent}% confidence (${severity.level}).<br>
                      <strong>Region:</strong> ${detection.anatomical_location}. Total findings: ${totalCount}.<br>
                      <em>Note: AI confidence is used as a proxy for severity for educational demonstration.</em>`
        };

        severitySummary.innerHTML = roleContextMap[currentRole] || roleContextMap.doctor;
    }

    // ---- Prescriptions ----
    function renderPrescriptions(diseases) {
        let html = "";

        diseases.forEach((det, dIdx) => {
            const db = PRESCRIPTION_DB[det.disease_raw];
            if (!db) return;

            const severity = db.severity(det.confidence_percent);
            const meds = db.medications[severity.level] || db.medications[Object.keys(db.medications)[0]];

            html += `<div class="disease-rx-group" style="animation-delay: ${dIdx * 0.1}s">`;
            html += `<div class="disease-rx-title">
                        <h3>${severity.icon} ${det.disease_raw}</h3>
                        <span class="disease-rx-severity" style="background: ${severity.color}22; color: ${severity.color};">${severity.level}</span>
                     </div>`;
            html += `<div class="rx-cards">`;

            meds.forEach((med, idx) => {
                const isDoctor = currentRole === "doctor";
                html += `
                <div class="rx-card" id="rx-${dIdx}-${idx}" style="animation-delay: ${idx * 0.06}s">
                    <div class="rx-card-header">
                        <div class="rx-drug-info">
                            <span class="rx-drug-name">${med.name}</span>
                            <span class="rx-dose">${med.dose}</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px">
                            <span class="rx-class-badge" style="background: ${med.clsColor}22; color: ${med.clsColor}; border: 1px solid ${med.clsColor}44;">${med.cls}</span>
                            ${isDoctor ? `<label class="rx-toggle"><input type="checkbox" checked onchange="document.getElementById('rx-${dIdx}-${idx}').classList.toggle('excluded')"> Include</label>` : ""}
                        </div>
                    </div>
                    <div class="rx-details">
                        <div class="rx-detail-item"><span class="rx-detail-label">Route</span><span class="rx-detail-value">${med.route}</span></div>
                        <div class="rx-detail-item"><span class="rx-detail-label">Frequency</span><span class="rx-detail-value">${med.frequency}</span></div>
                        <div class="rx-detail-item"><span class="rx-detail-label">Duration</span><span class="rx-detail-value">${med.duration}</span></div>
                    </div>
                    ${currentRole === "patient" ? `<div class="rx-patient-instruction">\u{1F4AC} ${med.instruction}</div>` : ""}
                </div>`;
            });

            html += `</div></div>`;
        });

        prescriptionContent.innerHTML = html;
    }

    // ---- Follow-Up Timeline ----
    function renderFollowUp(diseases) {
        let allFollowUps = [];

        diseases.forEach(det => {
            const db = PRESCRIPTION_DB[det.disease_raw];
            if (!db) return;
            db.follow_up.forEach(fu => {
                allFollowUps.push({ ...fu, disease: det.disease_raw });
            });
        });

        let html = "";
        allFollowUps.forEach((fu, idx) => {
            const isPneu = fu.disease === "Pneumonia";
            const tagColor = isPneu ? "background:rgba(255,107,0,0.12);color:#ff6b00;" : "background:rgba(0,230,118,0.12);color:#00e676;";

            html += `
            <div class="timeline-item" style="animation-delay: ${idx * 0.08}s">
                <div class="timeline-dot ${fu.urgent ? "urgent" : ""}"></div>
                <div class="timeline-content">
                    <div class="timeline-time">
                        ${fu.icon} ${fu.time}
                        ${fu.urgent ? '<span class="urgent-tag">URGENT</span>' : ""}
                    </div>
                    <div class="timeline-action">${fu.action}</div>
                    <span class="timeline-disease-tag" style="${tagColor}">${fu.disease}</span>
                </div>
            </div>`;
        });

        followupTimeline.innerHTML = html;
    }

    // ---- Diet Recommendation Agent ----
    function renderDiet(diseases) {
        let html = "";

        diseases.forEach(det => {
            const db = PRESCRIPTION_DB[det.disease_raw];
            if (!db || !db.diet) return;
            const diet = db.diet;

            // Overview
            html += `<div class="diet-overview"><h3>\u{1F37D}\uFE0F ${det.disease_raw} \u2014 Nutrition Plan</h3><p>${diet.overview}</p></div>`;

            // Food categories grid
            html += `<div class="diet-grid">`;
            diet.categories.forEach((cat, cIdx) => {
                html += `<div class="diet-category" style="animation-delay: ${cIdx * 0.1}s">`;
                html += `<div class="diet-category-title">${cat.icon} ${cat.title}</div>`;
                html += `<div class="diet-items">`;
                cat.items.forEach(item => {
                    html += `<div class="diet-item ${cat.type}">${item}</div>`;
                });
                html += `</div></div>`;
            });
            html += `</div>`;

            // Meal plan
            html += `<div class="diet-meal-plan">`;
            html += `<h4>\u{1F374} Suggested Daily Meal Plan</h4>`;
            diet.meal_plan.forEach(meal => {
                html += `<div class="meal-row">
                    <span class="meal-time">${meal.time}</span>
                    <span class="meal-desc">${meal.desc}</span>
                </div>`;
            });
            html += `</div>`;

            // Key nutrients
            html += `<div class="diet-nutrients">`;
            diet.key_nutrients.forEach(n => {
                html += `<span class="nutrient-tag ${n.highlight ? "highlight" : ""}">${n.name}</span>`;
            });
            html += `</div>`;
        });

        dietContent.innerHTML = html;
    }

    // ---- Patient Info ----
    function renderPatientInfo(diseases) {
        let html = "";

        diseases.forEach(det => {
            const db = PRESCRIPTION_DB[det.disease_raw];
            if (!db) return;
            const info = db.patient_info;

            html += `<div class="patient-explanation"><h3>\u{1F4A1} What is ${det.disease_raw}?</h3><p>${info.simple_explanation}</p></div>`;

            html += `<h3 style="margin-bottom:12px;font-size:1rem;">\u{1F49A} Lifestyle Tips for Recovery</h3>`;
            html += `<div class="lifestyle-list">`;
            info.lifestyle.forEach((tip, i) => {
                html += `<div class="lifestyle-item" style="animation-delay: ${i * 0.05}s">${tip}</div>`;
            });
            html += `</div>`;

            html += `<div class="warning-section">`;
            html += `<div class="warning-title">\u{1F6A8} When to Seek Emergency Care</div>`;
            html += `<div class="warning-list">`;
            info.warning_signs.forEach((sign, i) => {
                html += `<div class="warning-item" style="animation-delay: ${i * 0.05}s">${sign}</div>`;
            });
            html += `</div></div>`;
        });

        lifestyleContent.innerHTML = html;
    }

    // ---- Education ----
    function renderEducation(diseases) {
        let html = "";

        diseases.forEach(det => {
            const db = PRESCRIPTION_DB[det.disease_raw];
            if (!db) return;
            const edu = db.educational;

            html += `<div class="edu-disease-header"><h3>\u{1F4D6} ${det.disease_raw}</h3></div>`;

            html += section("\u{1F9EC} Definition", `<p class="edu-text">${edu.definition}</p>`);
            html += section("\u{1FA7B} Pathophysiology", `<p class="edu-text">${edu.pathophysiology}</p>`);
            html += section("\u26A0\uFE0F Risk Factors", listItems(edu.risk_factors));
            html += section("\u{1FA7A} Key Clinical Signs", listItems(edu.key_signs));
            html += section("\u{1F504} Differential Diagnoses", listItems(edu.differentials));
            html += section("\u{1F4DA} Study References", edu.references.map(r => `<div class="edu-ref">${r}</div>`).join(""));
        });

        educationContent.innerHTML = html;

        function section(title, content) {
            return `<div class="edu-section"><div class="edu-section-title">${title}</div>${content}</div>`;
        }
        function listItems(arr) {
            return `<ul class="edu-list">${arr.map(i => `<li>${i}</li>`).join("")}</ul>`;
        }
    }

    // ---- Role UI updates ----
    function updateRoleUI() {
        const roleLabels = {
            doctor: { rx: "Prescription", fu: "Follow-Up Plan", indicator: "\u{1FA7A} Doctor View", reportLabel: "\u{1F4CB} Generate Clinical Report" },
            patient: { rx: "My Medications", fu: "My Appointments", indicator: "\u{1F3E5} Patient View", reportLabel: "\u{1F4CB} Download My Report" },
            student: { rx: "Treatment Protocol", fu: "Follow-Up Guidelines", indicator: "\u{1F4DA} Study Mode", reportLabel: "\u{1F4CB} Generate Study Notes" }
        };

        const labels = roleLabels[currentRole] || roleLabels.doctor;
        rxTitle.textContent = labels.rx;
        fuTitle.textContent = labels.fu;
        rxRoleIndicator.textContent = labels.indicator;
        generateReportBtn.innerHTML = `<span>\u{1F4CB}</span> ${labels.reportLabel}`;
    }

    // =========================================================================
    //  8. REPORT GENERATION
    // =========================================================================
    generateReportBtn.addEventListener("click", () => generateReport(false));
    printReportBtn.addEventListener("click", () => generateReport(true));

    function generateReport(autoPrint) {
        if (!lastAnalysisData || lastAnalysisData.detection_count === 0) {
            alert("No analysis data available. Please run AI inference first.");
            return;
        }

        const data = lastAnalysisData;
        const diseases = [];
        const diseaseMap = {};
        data.detections.forEach(det => {
            if (!diseaseMap[det.disease_raw] || det.confidence_percent > diseaseMap[det.disease_raw].confidence_percent) {
                diseaseMap[det.disease_raw] = det;
            }
        });
        Object.values(diseaseMap).forEach(d => diseases.push(d));

        const primaryDet = diseases.reduce((a, b) => a.confidence_percent > b.confidence_percent ? a : b);
        const primaryDb = PRESCRIPTION_DB[primaryDet.disease_raw];
        const severity = primaryDb.severity(primaryDet.confidence_percent);
        const severityLevel = severity.level;
        const meds = primaryDb.medications[severityLevel] || primaryDb.medications[Object.keys(primaryDb.medications)[0]];
        const clinicalNotes = document.getElementById("clinicalNotes")?.value || "";
        const diet = primaryDb.diet;

        const now = new Date();

        const reportHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clinical Report \u2014 ${primaryDet.disease_raw}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 820px; margin: 0 auto; padding: 40px 30px; color: #1a1a1a; line-height: 1.6; }
        .header { text-align: center; border-bottom: 3px solid #3b82f6; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { font-size: 1.6rem; color: #1e293b; margin-bottom: 4px; }
        .header .subtitle { color: #64748b; font-size: 0.85rem; }
        .section { margin-bottom: 28px; }
        .section h2 { font-size: 1.1rem; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 14px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.88rem; }
        th, td { padding: 10px 12px; text-align: left; border: 1px solid #e2e8f0; }
        th { background: #f1f5f9; font-weight: 600; color: #334155; }
        .severity-label { display: inline-block; padding: 4px 14px; border-radius: 12px; font-weight: 700; font-size: 0.82rem; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 8px; font-size: 0.78rem; font-weight: 600; }
        .disclaimer { background: #fef3c7; border: 1px solid #f59e0b; padding: 14px 18px; border-radius: 10px; font-size: 0.82rem; margin-top: 30px; line-height: 1.5; }
        .signature { margin-top: 50px; display: flex; justify-content: space-between; }
        .sig-block { width: 240px; text-align: center; }
        .sig-line { border-top: 1px solid #333; margin-bottom: 6px; padding-top: 8px; font-size: 0.85rem; color: #64748b; }
        .diet-section ul { list-style: none; padding: 0; }
        .diet-section li { padding: 4px 0; font-size: 0.88rem; }
        .diet-section li:before { content: '\u2022 '; color: #10b981; font-weight: bold; }
        .diet-section .avoid li:before { color: #ef4444; }
        @media print { body { padding: 20px; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>\u{1FAC1} Chest X-ray AI \u2014 Clinical Report</h1>
        <p class="subtitle">Generated: ${now.toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" })} at ${now.toLocaleTimeString("en-IN")} | Role: ${currentRole.charAt(0).toUpperCase() + currentRole.slice(1)}</p>
    </div>

    <div class="section">
        <h2>\u{1F50D} Findings Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Primary Finding</td><td><strong>${primaryDet.disease}</strong></td></tr>
            <tr><td>AI Confidence</td><td>${primaryDet.confidence_percent}%</td></tr>
            <tr><td>Severity</td><td><span class="severity-label" style="background:${severity.color}18;color:${severity.color};">${severity.icon} ${severity.level}</span></td></tr>
            <tr><td>Anatomical Location</td><td>${primaryDet.anatomical_location}</td></tr>
            <tr><td>Total Detections</td><td>${data.detection_count}</td></tr>
            <tr><td>Confidence Threshold Used</td><td>${data.confidence_threshold_used}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>\u{1F48A} Prescribed Medications</h2>
        <table>
            <tr><th>Medication</th><th>Dose</th><th>Route</th><th>Frequency</th><th>Duration</th></tr>
            ${meds.map(m => `<tr><td>${m.name}</td><td>${m.dose}</td><td>${m.route}</td><td>${m.frequency}</td><td>${m.duration}</td></tr>`).join("")}
        </table>
    </div>

    <div class="section">
        <h2>\u{1F4C5} Follow-Up Schedule</h2>
        <table>
            <tr><th>Timeline</th><th>Action Required</th><th>Priority</th></tr>
            ${primaryDb.follow_up.map(f => `<tr><td><strong>${f.time}</strong></td><td>${f.action}</td><td>${f.urgent ? '<span class="badge" style="background:#fef3c7;color:#b45309;">Urgent</span>' : "Routine"}</td></tr>`).join("")}
        </table>
    </div>

    <div class="section diet-section">
        <h2>\u{1F957} Diet Recommendations</h2>
        <p style="margin-bottom:12px;font-size:0.9rem;color:#475569;">${diet.overview}</p>
        ${diet.categories.map(cat => `
            <h3 style="font-size:0.95rem;margin:12px 0 6px;color:${cat.type === "recommend" ? "#059669" : "#dc2626"};">${cat.title}</h3>
            <ul class="${cat.type}">
                ${cat.items.map(i => `<li>${i}</li>`).join("")}
            </ul>
        `).join("")}
    </div>

    ${clinicalNotes ? `
    <div class="section">
        <h2>\u{1F4DD} Clinical Notes</h2>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;font-size:0.9rem;white-space:pre-wrap;">${clinicalNotes}</div>
    </div>` : ""}

    <div class="disclaimer">
        <strong>\u26A0\uFE0F Medical Research Disclaimer:</strong> This report is AI-generated for research and educational purposes only. 
        It is not a substitute for evaluation by a qualified medical professional. All findings, prescriptions, and dietary recommendations 
        should be reviewed and validated by a licensed physician before any clinical action is taken.
    </div>

    <div class="signature">
        <div class="sig-block">
            <div class="sig-line">Physician Signature</div>
        </div>
        <div class="sig-block">
            <div class="sig-line">Date</div>
        </div>
    </div>
</body>
</html>`;

        const reportWindow = window.open("", "_blank");
        reportWindow.document.write(reportHTML);
        reportWindow.document.close();

        if (autoPrint) {
            reportWindow.onload = () => reportWindow.print();
        }
    }
});
