import torch

MODEL_PATH = "models/bmd/bmd_model.pth"

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    for key in checkpoint:
        print("-", key)

    print("\nTypes:")
    for key, value in checkpoint.items():
        print(key, "->", type(value))