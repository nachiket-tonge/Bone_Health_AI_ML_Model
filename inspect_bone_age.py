import torch

ckpt = torch.load(
    "models/bone_age/bone_age_resnet50_full.pth",
    map_location="cpu",
    weights_only=False
)

for k in ckpt["model_state_dict"]:
    print(k)