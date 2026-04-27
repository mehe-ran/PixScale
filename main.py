import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter # <-- Added ImageFilter here
import torchvision.transforms as transforms
from model import CompactSRResNet

app = FastAPI(title="CORE API")

# allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# hardware setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "compact_srresnet.pth"

# load model globally into memory
print("loading ai model into memory...")
model = CompactSRResNet(scale_factor=4).to(device)

# intercept and translate legacy weights to the new clean architecture
try:
    old_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for key, value in old_state_dict.items():
        new_key = key.replace('upsample.upsample_block.', 'upsample.')
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
except:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.eval()
print("model loaded. api is ready.")

# the ai endpoint
@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    # read uploaded image bytes
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # preprocess: keep it standard [0, 1]
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # postprocess: clamp to ensure valid pixels and convert back
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)

    # 1. Kill the color static
    output_img = output_img.filter(ImageFilter.MedianFilter(size=3))
    # 2. Sharpen the edges
    output_img = output_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # ---------------------------

    # save to byte stream for http response
    img_byte_arr = io.BytesIO()
    output_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

# mount the frontend folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")