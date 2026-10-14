import torch

checkpoint = torch.load(
    "models/bmd/bmd_model.pth",
    map_location="cpu",
    weights_only=False
)

print("\n======= LABEL ENCODERS =======\n")

encoders = checkpoint["label_encoders"]

for name, encoder in encoders.items():

    print(name)

    print("Classes:")

    print(list(encoder.classes_))

    print("-" * 40)

print("\n======= FEATURE ORDER =======")

print(checkpoint["feature_names"])