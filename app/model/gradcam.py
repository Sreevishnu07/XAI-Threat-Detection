import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model):
    return model.layer4[-1]


def compute_focus_score(cam: np.ndarray) -> float:
    """
    Energy-based focus score:
    Measures how concentrated the attention is
    """

    # Normalize CAM to [0, 1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Sharpness idea: high peaks = focused
    mean_val = np.mean(cam)
    max_val = np.max(cam)

    # Focus = how peaked the attention is
    focus_score = max_val - mean_val

    # Clamp to [0,1]
    focus_score = float(np.clip(focus_score, 0.0, 1.0))

    return focus_score

def generate_gradcam(model, input_tensor, image_np):
    """
    Returns:
    - cam_image (visualization)
    - focus_score (numeric attention quality)
    """

    target_layer = get_target_layer(model)

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    focus_score = compute_focus_score(grayscale_cam)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    return cam_image, focus_score
