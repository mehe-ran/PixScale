# CORE | Neural Upscaling Engine

CORE is a full stack, AI powered image super resolution engine. It utilizes a custom Compact SRResNet architecture to dynamically upscale low resolution inputs by a factor of 4x, recovering lost textures and sharp edges.

## Architecture & Stack
* **Neural Network:** PyTorch (Compact SRResNet)
* **Hardware Acceleration:** Optimized for Apple Silicon (MPS)
* **Backend:** FastAPI (Python)
* **Frontend:** Vanilla JS, HTML, Tailwind CSS (Industrial Chrome UI)
* **Training Data:** Custom dataloader trained on a 12GB high-resolution 4K image archive utilizing 96px random patch cropping.

## The Pipeline
1. **Inference API:** The FastAPI backend receives image bytes and normalizes the tensor data.
2. **Super Resolution:** The model processes the `[0, 1]` tensor through multiple residual blocks, using a sub-pixel convolution upsample block to generate the 4x output.
3. **UI Integration:** The frontend renders the result via a custom-built clip-path slider for real-time before/after comparison.

## Local Setup
Ensure you have Python 3.x installed and a virtual environment active.

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the inference server:
   `uvicorn main:app --reload`
3. Access the web interface at `http://127.0.0.1:8000`