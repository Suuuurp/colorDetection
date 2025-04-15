// script.js
// REVISED - Debugging ROI Selection

// Get references to HTML elements
const videoElement = document.getElementById('webcamVideo');
const outputCanvas = document.getElementById('outputCanvas'); // Visible canvas
const outputCtx = outputCanvas.getContext('2d');
const resultsDiv = document.getElementById('results');
const modeStatusSpan = document.getElementById('modeStatus');

// --- Create Offscreen Canvas ---
const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d');
let offscreenCanvasInitialized = false;

// --- State Variables ---
let isProcessing = false;
let drawPoseSkeleton = false; // Keep variable even if drawing is removed
let currentDetections = [];
// let currentPoseLandmarks = null; // No longer needed
let currentMode = "Initializing";

// --- ROI Selection State ---
let isSelectingROI = false;
let roiStartX = 0;
let roiStartY = 0;
let roiEndX = 0;
let roiEndY = 0;
let manualROI = null; // Stores {x, y, w, h} after selection

// --- Webcam Access ---
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
            console.log("Webcam stream started."); // LOG
            updateStatus("Webcam active. Starting detection...");
            // Set canvas sizes
            outputCanvas.width = videoElement.videoWidth;
            outputCanvas.height = videoElement.videoHeight;
            offscreenCanvas.width = videoElement.videoWidth;
            offscreenCanvas.height = videoElement.videoHeight;
            offscreenCanvasInitialized = true;
            // Add Event Listeners
            console.log("Attaching mouse listeners to outputCanvas"); // LOG
            outputCanvas.addEventListener('mousedown', handleMouseDown);
            outputCanvas.addEventListener('mousemove', handleMouseMove);
            outputCanvas.addEventListener('mouseup', handleMouseUp);
            outputCanvas.addEventListener('mouseleave', handleMouseLeave);
            window.addEventListener('keydown', handleKeyDown); // Keep for logging ('l')
            requestAnimationFrame(processFrameLoop); // Start Loop
        };
    } catch (error) { /* ... error handling ... */ }
}

// --- Frame Processing Loop ---
async function processFrameLoop() {
    if (!offscreenCanvasInitialized || videoElement.paused || videoElement.ended) {
        requestAnimationFrame(processFrameLoop); return;
    }

    // 1. Draw video to offscreen first
    offscreenCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    offscreenCtx.drawImage(videoElement, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    // 2. If processing, skip backend request, draw last overlays, and loop
    if (isProcessing) {
        drawOverlays(offscreenCtx); // Draw last known overlays on offscreen
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputCtx.drawImage(offscreenCanvas, 0, 0); // Copy to visible
        requestAnimationFrame(processFrameLoop);
        return;
    }

    // 3. Get frame data from offscreen canvas
    const frameData = offscreenCanvas.toDataURL('image/jpeg', 0.7);

    // 4. Prepare request data
    const requestData = { image_data: frameData };
    // --- ROI SEND CHECK ---
    // console.log(`ROI Check: Mode='${currentMode}', manualROI=`, manualROI); // DEBUG
    if (currentMode === "ROI Selection" && manualROI) {
         requestData.manual_roi = manualROI;
         // console.log("===> Adding manual_roi to requestData:", requestData.manual_roi); // DEBUG
    }

    // 5. Send to backend
    isProcessing = true;
    let results = null;
    try {
        // console.log(">>> Sending Request:", JSON.stringify(requestData).substring(0, 200) + "..."); // DEBUG
        const response = await fetch('/process_frame', { /* ... */
             method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestData) });
        if (!response.ok) { const e = await response.json().catch(()=>({})); throw new Error(`HTTP ${response.status}: ${e.error||'Server Error'}`); }
        results = await response.json();
        // console.log("Backend Response:", results); // DEBUG

        // 6. Update State
        // --- Explicit Mode Log ---
        const previousMode = currentMode;
        currentMode = results.mode || "Error";
        if(currentMode !== previousMode) {
            console.log(`MODE CHANGED from ${previousMode} to ${currentMode}`); // LOG MODE CHANGE
        }
        // --- End Explicit Mode Log ---

        currentDetections = results.detections || [];
        // currentPoseLandmarks = null; // No longer needed
        if (currentMode === "Human Detected" && manualROI) { manualROI = null; }

        // 7. Update Text Display based on received results
        updateResultsText(currentDetections, currentMode);

    } catch (error) { /* ... error handling ... */ }
     finally {
        isProcessing = false;
    }

     // Draw overlays onto OFFSCREEN canvas based on the state updated above
    drawOverlays(offscreenCtx);

    // Copy final composed image from OFFSCREEN to VISIBLE canvas
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    outputCtx.drawImage(offscreenCanvas, 0, 0);

    // Loop
    requestAnimationFrame(processFrameLoop);
}


// --- Combined Overlay Drawing Function (takes context as argument) ---
function drawOverlays(ctx) {
    ctx.save();
    drawDetections(ctx, currentDetections); // Draw detection boxes

    // --- Draw ROI selection rectangles ---
    // Draw finalized green box FIRST (if applicable)
    if (!isSelectingROI && currentMode === "ROI Selection" && manualROI) {
         console.log("Drawing finalized green ROI box:", manualROI); // DEBUG
         ctx.strokeStyle = "lime";
         ctx.lineWidth = 4; // Thick final ROI box
         ctx.strokeRect(manualROI.x, manualROI.y, manualROI.w, manualROI.h);
     }
     // Draw red dragging box SECOND (if applicable) - Drawn in mousemove now

    ctx.restore();
}

// --- Drawing Helpers ---
function drawDetections(ctx, detections) { // Keep thick lines
    if (!detections || !Array.isArray(detections)) { return; }
    detections.forEach((det) => {
        if (/* ... standard checks ... */
            det && typeof det === 'object' && det.roi && Array.isArray(det.roi) && det.roi.length === 4 &&
            det.draw_color && Array.isArray(det.draw_color) && det.draw_color.length === 3 &&
            det.roi.every(val => typeof val === 'number' && !isNaN(val)) &&
            det.draw_color.every(val => typeof val === 'number' && !isNaN(val)) )
        {
            const [x, y, w, h] = det.roi;
            const [b, g, r] = det.draw_color;
            if (w > 0 && h > 0) {
                ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.lineWidth = 6; // Keep detection boxes VERY thick
                ctx.beginPath(); ctx.rect(x, y, w, h); ctx.stroke();
            }
        }
    });
}

// function drawPose(...) // REMOVED

// --- Update UI Text ---
function updateResultsText(detections, mode) { /* ... remains the same ... */
    let modeText = mode || "Unknown"; let instructionText = "";
    if (mode === "ROI Selection" && !manualROI && !isSelectingROI) {
        instructionText = " - Click & drag to select ROI.";
    }
    modeStatusSpan.textContent = ` ${modeText}${instructionText}`;
    // console.log("Updating text for mode:", mode); // DEBUG
    if (!detections || !Array.isArray(detections) || detections.length === 0) { /* ... no detections message ... */ return; }
    let htmlContent = "<ul>";
    detections.forEach((det, index) => { /* ... build list item ... */
        const part = det.part || 'Unknown Part'; const colorName = det.color_name || 'Processing...';
        const hex = det.hex || '...'; const avgBgr = det.avg_bgr || 'N/A';
        const method = det.method || '...';
        htmlContent += `<li><b>${part}:</b> ${colorName} (${hex}) <small>(AvgBGR:${avgBgr}) [${method}]</small></li>`;
    });
    htmlContent += "</ul>"; resultsDiv.innerHTML = htmlContent;
    // console.log("Updated resultsDiv innerHTML."); // DEBUG
}
function updateStatus(message, isError = false) { /* ... remains the same ... */
      modeStatusSpan.textContent = ` ${message}`; /* ... */ resultsDiv.innerHTML = "";
}


// --- Mouse Event Handlers for ROI ---
function handleMouseDown(event) {
    console.log("[Mouse Down] Mode:", currentMode); // LOG MODE ON CLICK
    if (currentMode === "ROI Selection") { // Check the updated currentMode
        console.log("==> Starting ROI Selection Process"); // LOG START
        isSelectingROI = true;
        manualROI = null; // Clear previous final ROI when starting new one
        const rect = outputCanvas.getBoundingClientRect();
        roiStartX = event.clientX - rect.left; roiStartY = event.clientY - rect.top;
        roiEndX = roiStartX; roiEndY = roiStartY;
        console.log("    Start Coords:", roiStartX, roiStartY); // LOG Coords
    } else {
        console.log("==> Mouse Down Ignored (Mode is Human Detected)"); // LOG IGNORE
    }
}

function handleMouseMove(event) {
    if (isSelectingROI) { // Only draw if actively selecting
        const rect = outputCanvas.getBoundingClientRect();
        roiEndX = event.clientX - rect.left;
        roiEndY = event.clientY - rect.top;

        // --- Draw the red dragging rectangle Directly on VISIBLE canvas ---
        // Redraw the *latest* frame from offscreen first to clear old red box
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputCtx.drawImage(offscreenCanvas, 0, 0);
        // Draw other potentially existing overlays (e.g., previous green box if any)
        // This ensures the red box is on the very top during drag
        drawOverlays(outputCtx); // Call helper to draw static stuff first

        // Now draw the red box on top
        console.log(`    Dragging ROI to ${roiEndX}, ${roiEndY}`); // LOG DRAG
        outputCtx.strokeStyle = "red";
        outputCtx.lineWidth = 2; // Keep drag box reasonably thin
        outputCtx.strokeRect(roiStartX, roiStartY, roiEndX - roiStartX, roiEndY - roiStartY);
        // --- End immediate feedback drawing ---
    }
}

function handleMouseUp(event) {
    console.log("[Mouse Up] isSelectingROI:", isSelectingROI); // LOG STATE ON UP
    if (isSelectingROI) {
        isSelectingROI = false; // Stop dragging indicator FIRST
        const rect = outputCanvas.getBoundingClientRect();
        roiEndX = event.clientX - rect.left; roiEndY = event.clientY - rect.top; // Get final coords
        const x = Math.min(roiStartX, roiEndX); const y = Math.min(roiStartY, roiEndY);
        const w = Math.abs(roiStartX - roiEndX); const h = Math.abs(roiStartY - roiEndY);

        if (w > 10 && h > 10) {
             manualROI = { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
             console.log("==> ROI Set (Final):", manualROI); // LOG FINAL ROI
        } else {
             manualROI = null; // Clear ROI if too small
             console.log("==> ROI too small/discarded on Mouse Up"); // LOG DISCARD
        }
        // Immediately redraw overlays to show green box or clear red box
         outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
         outputCtx.drawImage(offscreenCanvas, 0, 0); // Use last offscreen image
         drawOverlays(outputCtx); // Draw overlays based on new state
    }
}

function handleMouseLeave(event) {
     if (isSelectingROI) {
        console.log("Mouse left canvas while selecting, cancelling ROI."); // LOG CANCEL
        isSelectingROI = false; manualROI = null; // Reset state
        // Redraw immediately to clear red box
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputCtx.drawImage(offscreenCanvas, 0, 0);
        drawOverlays(outputCtx); // Redraw without the dragging box
     }
}


// --- Keyboard Event Handler ---
async function handleKeyDown(event) {
    const key = event.key.toLowerCase();
    // if (key === 'p') { ... } // REMOVED 'p' key handling
    if (key === 'l') { // Keep logging
        console.log("Attempting to log detections:", currentDetections); // LOG
        if (currentDetections && currentDetections.length > 0) { /* ... logging fetch ... */ }
        else { console.log("Nothing to log."); } // LOG
    }
}

// --- Start Webcam Setup ---
console.log("Initializing script..."); // LOG
updateStatus("Requesting webcam access...");
setupWebcam();