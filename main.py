from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.fracture import router as fracture_router
from routers.bmd import router as bmd_router
from routers.bone_age import router as bone_age_router

app = FastAPI(
    title="Bone Health AI API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fracture_router)
app.include_router(bmd_router)

app.include_router(
    bone_age_router,
    prefix="/bone-age",
    tags=["Bone Age"]
)

@app.get("/")
def home():
    return {
        "message": "Bone Health AI API is Running 🚀"
    }