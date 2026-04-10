import numpy as np
import torch
import cv2

from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Integrated Gradients
from captum.attr import IntegratedGradients


def get_target_layer(model):
    return model.layer4[-1]


# ---------------------------
# Focus Score (same, correct)
# ---------------------------
def compute_focus_score(cam: np.ndarray) -> float:
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    mean_val = np.mean(cam)
    max_val = np.max(cam)

    focus_score = max_val - mean_val
    return float(np.clip(focus_score, 0.0, 1.0))


# ---------------------------
# Integrated Gradients
# ---------------------------
def generate_integrated_gradients(model, input_tensor):
    ig = IntegratedGradients(model)

    # target = predicted class
    input_tensor.requires_grad = True
    output = model(input_tensor)
    target_class = output.argmax(dim=1).item()

    attributions = ig.attribute(input_tensor, target=target_class)

    attr = attributions.squeeze().detach().cpu().numpy()
    attr = np.mean(attr, axis=0)

    attr = attr - attr.min()
    attr = attr / (attr.max() + 1e-8)

    return attr


def generate_xai_maps(model, input_tensor, image_np):
    """
    Returns:
    {
        "gradcam++": image,
        "scorecam": image,
        "integrated_gradients": image,
        "focus_scores": {...}
    }
    """

    target_layer = get_target_layer(model)

    cam_pp = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    cam_pp_map = cam_pp(input_tensor=input_tensor)[0]
    cam_pp_img = show_cam_on_image(image_np, cam_pp_map, use_rgb=True)
    focus_pp = compute_focus_score(cam_pp_map)

    score_cam = ScoreCAM(model=model, target_layers=[target_layer])
    score_map = score_cam(input_tensor=input_tensor)[0]
    score_img = show_cam_on_image(image_np, score_map, use_rgb=True)
    focus_scorecam = compute_focus_score(score_map)

    ig_map = generate_integrated_gradients(model, input_tensor)

    ig_img = (ig_map * 255).astype(np.uint8)
    ig_img = cv2.applyColorMap(ig_img, cv2.COLORMAP_JET)
    ig_img = cv2.cvtColor(ig_img, cv2.COLOR_BGR2RGB)

    focus_ig = compute_focus_score(ig_map)

    return {
        "gradcam_pp": cam_pp_img,
        "scorecam": score_img,
        "integrated_gradients": ig_img,
        "focus_scores": {
            "gradcam_pp": focus_pp,
            "scorecam": focus_scorecam,
            "integrated_gradients": focus_ig
        }
    }
