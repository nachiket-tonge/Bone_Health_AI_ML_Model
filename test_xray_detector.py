from pathlib import Path

from services.xray_service import detect_xray


# Change this to the image you want to test
IMAGE_PATH = Path("test_images/test.jpg")


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


result = detect_xray(image_bytes)

print("X-Ray Detector Result:")
print(result)