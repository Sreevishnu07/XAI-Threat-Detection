import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients
from app.model.threat import smoothgrad_integrated_gradients, normalize_ig_map


def get_target_layer(model):
    return model.layer4[-1]


def compute_focus_score(cam: np.ndarray) -> float:
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    mean_val = np.mean(cam)
    max_val = np.max(cam)
    return float(np.clip(max_val - mean_val, 0.0, 1.0))


def generate_integrated_gradients(model, input_tensor):
    ig = IntegratedGradients(model)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(input_tensor)
    output = model(input_tensor)
    target_class = output.argmax(dim=1).item()
    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=20  # reduced from 50
    )
    attr = attributions.squeeze().detach().cpu().numpy()
    attr = np.abs(attr)
    attr = np.mean(attr, axis=0)
    attr = attr - attr.min()
    attr = attr / (attr.max() + 1e-8)
    attr = cv2.GaussianBlur(attr, (5, 5), 0)
    return attr


def generate_xai_maps(model, input_tensor, image_np):
    target_layer = get_target_layer(model)

    cam_pp = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    cam_pp_map = cam_pp(input_tensor=input_tensor)[0]
    cam_pp_img = show_cam_on_image(image_np, cam_pp_map, use_rgb=True)
    focus_pp = compute_focus_score(cam_pp_map)

    # ScoreCAM disabled - too slow on CPU, reusing GradCAM++ result
    score_img = cam_pp_img
    score_map = cam_pp_map
    focus_scorecam = focus_pp

    ig_map = smoothgrad_integrated_gradients(
        model,
        input_tensor,
        generate_integrated_gradients,
        n_samples=5,  
        noise_sigma=0.05
    )
    ig_map = normalize_ig_map(ig_map)
    ig_uint8 = (ig_map * 255).astype(np.uint8)
    ig_uint8 = cv2.medianBlur(ig_uint8, 5)
    ig_img = cv2.applyColorMap(ig_uint8, cv2.COLORMAP_JET)
    ig_img = cv2.cvtColor(ig_img, cv2.COLOR_BGR2RGB)
    focus_ig = compute_focus_score(ig_map)

    return {
        "gradcam_pp": cam_pp_img,
        "scorecam": score_img,
        "integrated_gradients": ig_img,
        "maps": {
            "gradcam_pp": cam_pp_map,
            "scorecam": score_map,
            "integrated_gradients": ig_map
        },
        "focus_scores": {
            "gradcam_pp": focus_pp,
            "scorecam": focus_scorecam,
            "integrated_gradients": focus_ig
        }
    }
