import joblib
import pandas as pd


# -----------------------------
# Load Model Once
# -----------------------------

checkpoint = joblib.load(
    "models/bmd/bmd_pipeline.joblib"
)

model = checkpoint["model"]
feature_names = checkpoint["columns"]


# -----------------------------
# Prediction
# -----------------------------

def predict_bmd(data: dict):
    """
    Predict Bone Mineral Density.
    """

    # Create DataFrame
    df = pd.DataFrame([data])

    # Ensure columns are in training order
    df = df[feature_names]

    # Predict using the sklearn pipeline
    prediction = model.predict(df)[0]

    return {
        "predicted_bmd": round(float(prediction), 4)
    }