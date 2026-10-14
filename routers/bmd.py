from fastapi import APIRouter
from pydantic import BaseModel

from services.bmd_service import predict_bmd

router = APIRouter(
    prefix="/bmd",
    tags=["Bone Mineral Density"]
)


class BMDRequest(BaseModel):
    age: int
    sex: str
    fracture: str
    weight_kg: float
    height_cm: float
    medication: str



@router.post("/predict")
def predict(request: BMDRequest):

    result = predict_bmd(request.model_dump())

    return result