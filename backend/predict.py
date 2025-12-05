# backend/predict.py
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from backend.model import load_model

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMG_SIZE = 224

# Default preprocessing for input images
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def prepare_img_bytes(img_bytes: bytes):
    """Convert raw image bytes to PIL + normalized tensor."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # [1, C, H, W]
    return img, img_tensor


def _make_gradcam(model, target_layers, device: str):
    """
    Instantiate GradCAM in a way that is compatible with multiple
    versions of pytorch-grad-cam.
    """
    use_cuda_flag = (device is not None) and (device != "cpu")

    # Try legacy signature with use_cuda
    try:
        cam = GradCAM(model=model, target_layers=target_layers,
                      use_cuda=use_cuda_flag)
        return cam
    except TypeError:
        pass

    # Try newer signature with device
    try:
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            device=torch.device(device if device is not None else "cpu"),
        )
        return cam
    except TypeError:
        pass

    # Try (less common) use_gpu
    try:
        cam = GradCAM(model=model, target_layers=target_layers,
                      use_gpu=use_cuda_flag)
        return cam
    except Exception:
        pass

    # Fallback: bare constructor
    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        return cam
    except Exception as e:
        raise RuntimeError(
            "Unable to construct GradCAM with this pytorch-grad-cam version. "
            "Consider installing a compatible version, e.g.: "
            "pip install grad-cam==1.4.5. "
            f"Original error: {e}"
        )


def predict_and_explain(
    model_path: str,
    img_bytes: bytes,
    device: str = "cpu",
    labels_map: dict | None = None,
):
    """
    Run a forward pass + Grad-CAM and return:
      - predicted index
      - human-readable label
      - raw probabilities
      - percentages for Normal / Pneumonia
      - text summary for probabilities
      - heatmap legend text
      - Grad-CAM overlay (JPG bytes)
    """

    # Load model
    model = load_model(model_path, device=device)
    model.to(device)
    model.eval()

    # Prepare input
    orig_img, img_tensor = prepare_img_bytes(img_bytes)
    img_tensor = img_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    # ----- Human-readable label -----
    # Default mapping assuming class order: [NORMAL, PNEUMONIA]
    default_map = {
        0: "Normal chest X-ray",
        1: "Pneumonia chest X-ray",
    }

    if labels_map is None:
        label_name = default_map.get(pred_idx, str(pred_idx))
    else:
        # labels_map is class_name -> index (like {"NORMAL":0, "PNEUMONIA":1})
        # invert it:
        inv = {v: k for k, v in labels_map.items()}
        raw_label = inv.get(pred_idx, str(pred_idx))
        # If raw_label is numeric, try default_map; otherwise use string
        try:
            li = int(raw_label)
            label_name = default_map.get(li, str(li))
        except Exception:
            label_name = raw_label

    # ----- Percentages & summary text -----
    normal_pct = None
    pneumonia_pct = None
    summary_text = None

    if len(probs) >= 2:
        # Assuming index 0 = NORMAL, index 1 = PNEUMONIA
        normal_pct = float(probs[0] * 100.0)
        pneumonia_pct = float(probs[1] * 100.0)

        summary_text = (
            f"{normal_pct:.1f}% refers to a Normal chest radiograph and "
            f"{pneumonia_pct:.1f}% refers to Pneumonia."
        )

    # ----- Grad-CAM heatmap -----
    target_layers = [model.features]  # DenseNet features block
    cam = _make_gradcam(model=model, target_layers=target_layers,
                        device=device)

    grayscale_cam = cam(input_tensor=img_tensor)[0]  # HxW

    # Overlay CAM on original image (resized)
    rgb_img = np.array(
        orig_img.resize((IMG_SIZE, IMG_SIZE))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Encode overlay as JPEG
    import cv2
    _, overlay_jpg = cv2.imencode(
        ".jpg",
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
    )
    overlay_bytes = overlay_jpg.tobytes()

    # Legend / guidance text for heatmap
    heatmap_legend = (
        "Blue areas → low importance; "
        "Red/yellow → high importance (regions the model focused on)."
    )

    return {
        "pred_idx": pred_idx,
        "label": label_name,
        "probs": probs.tolist(),         # raw probabilities (0–1)
        "percentages": {                 # convenience percentages
            "normal": normal_pct,
            "pneumonia": pneumonia_pct,
        },
        "summary": summary_text,         # "94.6% refers to Normal ... "
        "heatmap_legend": heatmap_legend,
        "overlay": overlay_bytes,        # JPG bytes; app.py will base64 this
    }
