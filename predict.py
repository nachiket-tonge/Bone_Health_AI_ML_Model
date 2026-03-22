import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "models/fracture_model_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction < 0.5:
        return "Fracture Detected", 1 - prediction
    else:
        return "No Fracture", prediction
