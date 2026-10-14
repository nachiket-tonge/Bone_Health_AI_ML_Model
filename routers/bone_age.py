from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.bone_age_service import predict_bone_age

router = APIRouter()


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    gender: str = Form(...)
):
    """
    Predict Bone Age

    Form Data:
        image : X-ray Image
        gender : Male / Female
    """

    try:
        result = await predict_bone_age(image, gender)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )