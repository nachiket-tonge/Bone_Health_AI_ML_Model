import joblib
import pandas as pd

checkpoint = joblib.load("models/bmd/bmd_pipeline.joblib")
model = checkpoint["model"]

tests = [
    {
        "age": 25,
        "weight_kg": 55,
        "height_cm": 165,
        "sex": "F",
        "fracture": "no fracture",
        "medication": "No medication"
    },
    {
        "age": 35,
        "weight_kg": 70,
        "height_cm": 175,
        "sex": "M",
        "fracture": "no fracture",
        "medication": "No medication"
    },
    {
        "age": 65,
        "weight_kg": 58,
        "height_cm": 160,
        "sex": "F",
        "fracture": "fracture",
        "medication": "Glucocorticoids"
    },
    {
        "age": 70,
        "weight_kg": 75,
        "height_cm": 172,
        "sex": "M",
        "fracture": "fracture",
        "medication": "Glucocorticoids"
    }
]

df = pd.DataFrame(tests)

predictions = model.predict(df)

for data, prediction in zip(tests, predictions):
    print(data)
    print("Predicted BMD:", prediction)
    print()