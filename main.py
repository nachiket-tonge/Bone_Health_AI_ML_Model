from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import base64
import cv2
import numpy as np

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

model.load_state_dict(torch.load("resnet50_fracture_best.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Transform
# -----------------------------
IMG_SIZE = 256

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "API is working 🚀"}

# -----------------------------
# Grad-CAM
# -----------------------------
def generate_heatmap(img_tensor, model):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]   # [C, H, W]
    fmap = features[0].detach().cpu().numpy()[0]     # [C, H, W]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    fh.remove()
    bh.remove()

    return heatmap, heatmap_color


# -----------------------------
# Bounding Box from Heatmap
# -----------------------------
def get_bounding_box_from_heatmap(heatmap_gray, threshold=180):
    """
    heatmap_gray: single-channel 0-255 heatmap
    returns: x, y, w, h or None
    """
    _, thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # ignore tiny noise regions
    if cv2.contourArea(largest_contour) < 50:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


# -----------------------------
# Draw Bounding Box
# -----------------------------
def draw_bounding_box_on_image(image_pil, bbox):
    image_np = np.array(image_pil)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            image_np,
            "Fracture Region",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return image_np


# -----------------------------
# Encode image to base64
# -----------------------------
def encode_image_to_base64(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")


# -----------------------------
# Predict API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))

        img_tensor = transform(image).unsqueeze(0)

        with torch.enable_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "Fracture Detected" if pred == 1 else "No Fracture"

        # Grad-CAM
        heatmap_gray, heatmap_color = generate_heatmap(img_tensor, model)

        bbox = None
        boxed_image_np = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

        if pred == 1:
            bbox = get_bounding_box_from_heatmap(heatmap_gray, threshold=180)
            boxed_image_np = draw_bounding_box_on_image(image_resized, bbox)

        heatmap_base64 = encode_image_to_base64(heatmap_color)
        boxed_image_base64 = encode_image_to_base64(boxed_image_np)

        response = {
            "prediction": label,
            "confidence": float(confidence),
            "heatmap": heatmap_base64,
            "boxed_image": boxed_image_base64,
            "bounding_box": None
        }

        if bbox is not None:
            x, y, w, h = bbox
            response["bounding_box"] = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }

        return response

    except Exception as e:
        return {"error": str(e)}