import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def compute_focus_score(cam):
    # cam values are between 0 and 1
    return float(cam.max() - cam.mean())


def generate_gradcam(model, input_tensor, image_np):
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    focus_score = compute_focus_score(grayscale_cam)

    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    return visualization, focus_score
