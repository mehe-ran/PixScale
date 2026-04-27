// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const btnText = document.getElementById('btn-text');
const btnIcon = document.getElementById('btn-icon');

const sliderContainer = document.getElementById('slider-container');
const sliderHandle = document.getElementById('slider-handle');
const beforeImg = document.getElementById('before-img');
const afterImg = document.getElementById('after-img');

// --- Slider Interaction Logic ---
let isDragging = false;

// Mouse events
sliderContainer.addEventListener('mousedown', (e) => { isDragging = true; updateSlider(e.clientX); });
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', (e) => { if (isDragging) updateSlider(e.clientX); });

// Touch events for mobile/trackpad
sliderContainer.addEventListener('touchstart', (e) => { isDragging = true; updateSlider(e.touches[0].clientX); });
window.addEventListener('touchend', () => { isDragging = false; });
window.addEventListener('touchmove', (e) => { if (isDragging) updateSlider(e.touches[0].clientX); });

function updateSlider(clientX) {
    const rect = sliderContainer.getBoundingClientRect();
    let x = clientX - rect.left;

    // Constrain to container boundaries
    x = Math.max(0, Math.min(x, rect.width));

    const percentage = (x / rect.width) * 100;

    // Move the handle
    sliderHandle.style.left = `${percentage}%`;

    // Clip the top image (Original) to reveal the bottom image (Upscaled)
    beforeImg.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;
}

// --- API Communication Logic ---
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 1. Stage the UI for Processing
    const objectURL = URL.createObjectURL(file);
    beforeImg.src = objectURL;

    // Temporarily blur the original on the "Output" side as a placeholder
    afterImg.src = objectURL;
    afterImg.style.filter = "blur(10px) brightness(0.5)";

    // Update Button State
    uploadBtn.classList.add('system-processing');
    btnText.innerText = "Processing...";
    btnIcon.innerText = "memory";

    // Reset slider to middle
    sliderHandle.style.left = '50%';
    beforeImg.style.clipPath = 'inset(0 50% 0 0)';

    // 2. Transmit to FastAPI
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8000/upscale/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Neural Engine failed to process payload.");

        // 3. Render Result
        const blob = await response.blob();
        afterImg.src = URL.createObjectURL(blob);
        afterImg.style.filter = "none"; // Remove blur

    } catch (error) {
        console.error(error);
        alert("System Error: Failed to process image through AI core.");
        afterImg.src = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="; // Reset
    } finally {
        // Restore Button State
        uploadBtn.classList.remove('system-processing');
        btnText.innerText = "Upload Image";
        btnIcon.innerText = "upload";
        fileInput.value = ''; // Clear input
    }
});