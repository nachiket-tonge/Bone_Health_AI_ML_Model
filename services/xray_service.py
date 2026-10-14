import io

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from PIL import Image


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/xray_detector.pt"


# -----------------------------
# Load Model Once
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

IMG_SIZE = checkpoint["img"]
THRESHOLD = checkpoint["threshold"]


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()


# -----------------------------
# Image Transform
# -----------------------------

transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# X-Ray Detection
# -----------------------------

def detect_xray(image_bytes: bytes):
    """
    Determine whether the uploaded image
    is classified as an X-ray image.
    """

    # Convert bytes to PIL image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():

        output = model(image_tensor)

        probability = torch.sigmoid(
            output
        ).item()

    is_xray = probability >= THRESHOLD

    return {
        "is_xray": bool(is_xray),
        "confidence": float(probability),
        "threshold": float(THRESHOLD)
    }
