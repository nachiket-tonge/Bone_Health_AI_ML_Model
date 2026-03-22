from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Fracture Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once (VERY IMPORTANT)
MODEL_PATH = "models/fracture_model_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_image(image):
    img = cv2.resize(image, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Grad-CAM function
# -------------------------------
def generate_gradcam(image):

    img_array = preprocess_image(image)

    last_conv_layer = model.get_layer("conv5_block3_3_conv")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)

    # Preprocess
    img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        label = "Fracture Detected"
        confidence = float(1 - prediction)
    else:
        label = "No Fracture"
        confidence = float(prediction)

    # Grad-CAM
    heatmap_img = generate_gradcam(image)

    # Convert heatmap to base64 (for frontend)
    _, buffer = cv2.imencode('.jpg', heatmap_img)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "heatmap": heatmap_base64
    }