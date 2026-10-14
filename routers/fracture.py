from fastapi import APIRouter, UploadFile, File
from services.xray_service import detect_xray
from services.fracture_service import predict_fracture

router = APIRouter(
    tags=["Fracture Detection"]
)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        # -----------------------------
        # Read image ONCE
        # -----------------------------

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "error": "Uploaded image is empty."
            }

        # -----------------------------
        # Step 1: Check X-Ray
        # -----------------------------

        xray_result = detect_xray(image_bytes)

        if not xray_result["is_xray"]:
            return {
                "error": "INVALID_XRAY_IMAGE",
                "message": (
                    "The uploaded image does not appear to be "
                    "a suitable bone X-ray. Please upload a "
                    "clear bone X-ray image."
                ),
                "confidence": xray_result["confidence"]
            }

        # -----------------------------
        # Step 2: Fracture Prediction
        # -----------------------------

        result = predict_fracture(image_bytes)

        return result

    except Exception as e:
        return {
            "error": str(e)
        }