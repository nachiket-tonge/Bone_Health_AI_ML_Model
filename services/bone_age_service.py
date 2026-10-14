import io

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image


# =====================================================
# Bone Age Model
# =====================================================

class BoneAgeResNet(nn.Module):

    def __init__(self, backbone="resnet50"):

        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0
        )

        self.gender_fc = nn.Sequential(

            nn.Linear(1, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU()

        )

        self.head = nn.Sequential(

            nn.Linear(2048 + 32, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(256, 1)

        )

    def forward(self, image, gender):

        image_features = self.backbone(image)

        gender_features = self.gender_fc(gender)

        features = torch.cat(
            [image_features, gender_features],
            dim=1
        )

        return self.head(features)

# =====================================================
# Load Checkpoint
# =====================================================

CHECKPOINT_PATH = "models/bone_age/bone_age_resnet50_full.pth"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

IMG_SIZE = checkpoint["img_size"]
AGE_MEAN = checkpoint["age_mean"]
AGE_STD = checkpoint["age_std"]

model = BoneAgeResNet(
    backbone=checkpoint["backbone"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# =====================================================
# Image Transform
# =====================================================

transform = T.Compose([

    T.Resize((IMG_SIZE, IMG_SIZE)),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =====================================================
# Gender Mapping
# =====================================================

def preprocess_gender(gender: str):

    gender = gender.strip().lower()

    if gender == "male":
        return torch.tensor([[1.0]])

    if gender == "female":
        return torch.tensor([[0.0]])

    raise ValueError(
        "Gender must be Male or Female."
    )


# =====================================================
# Prediction Function
# =====================================================

async def predict_bone_age(image, gender):

    # ---------- Read Image ----------
    image_bytes = await image.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    # ---------- Gender ----------
    gender_tensor = preprocess_gender(gender)

    # ---------- Prediction ----------
    with torch.no_grad():

        prediction = model(
            image,
            gender_tensor
        )

    # ---------- Denormalize ----------
    predicted_age = (
        prediction.item() * AGE_STD
    ) + AGE_MEAN

    predicted_age = round(
        predicted_age,
        2
    )

    predicted_years = round(
        predicted_age / 12,
        2
    )

    # ---------- Response ----------
    return {

        "success": True,

        "prediction": {

            "bone_age_months": predicted_age,

            "bone_age_years": predicted_years,

            "unit": "months"

        }

    }