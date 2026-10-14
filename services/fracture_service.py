from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import cv2
import numpy as np

from utils.gradcam import (
    generate_heatmap,
    get_bounding_box_from_heatmap,
    draw_bounding_box_on_image,
    encode_image
)

IMG_SIZE = 256

# -----------------------------
# Load Model (Loads only once)
# -----------------------------
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

model.load_state_dict(
    torch.load(
        "models/resnet50_fracture_best.pth",
        map_location="cpu"
    )
)

model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict_fracture(image_bytes: bytes):
    """
    Predict fracture from uploaded X-ray image.
    """

    image = Image.open(
    io.BytesIO(image_bytes)
).convert("RGB")

    image_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.enable_grad():

        output = model(img_tensor)

        probs = F.softmax(output, dim=1)

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[0][pred].item()

    label = (
        "Fracture Detected"
        if pred == 1
        else "No Fracture"
    )

    heatmap_gray, heatmap_color = generate_heatmap(
        img_tensor,
        model,
        IMG_SIZE
    )

    bbox = None

    boxed_image = cv2.cvtColor(
        np.array(image_resized),
        cv2.COLOR_RGB2BGR
    )

    if pred == 1:

        bbox = get_bounding_box_from_heatmap(
            heatmap_gray,
            threshold=180
        )

        boxed_image = draw_bounding_box_on_image(
            image_resized,
            bbox
        )

    response = {

        "prediction": label,

        "confidence": float(confidence),

        "heatmap": encode_image(
            heatmap_color
        ),

        "boxed_image": encode_image(
            boxed_image
        ),

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