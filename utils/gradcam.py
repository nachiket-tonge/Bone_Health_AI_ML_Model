import base64
import cv2
import numpy as np


def generate_heatmap(img_tensor, model, img_size):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    fmap = features[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_size, img_size))

    if cam.max() != 0:
        cam /= cam.max()

    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    fh.remove()
    bh.remove()

    return heatmap, heatmap_color


def get_bounding_box_from_heatmap(
        heatmap_gray,
        threshold=180
):
    _, thresh = cv2.threshold(
        heatmap_gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_DILATE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 50:
        return None

    return cv2.boundingRect(largest)


def draw_bounding_box_on_image(
        image_pil,
        bbox
):

    image_np = np.array(image_pil)

    image_np = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    if bbox is not None:
        x, y, w, h = bbox

        cv2.rectangle(
            image_np,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            image_np,
            "Fracture Region",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return image_np


def encode_image(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)

    return base64.b64encode(
        buffer
    ).decode("utf-8")