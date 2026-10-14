import torch

checkpoint = torch.load(
    "models/bone_age/bone_age_resnet50_full.pth",
    map_location="cpu",
    weights_only=False
)

state = checkpoint["model_state_dict"]

print("\nLast Layers:\n")

for key in state.keys():
    if "fc" in key or "head" in key:
        print(key, state[key].shape)