import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

try:
    class_idx = json.load(urllib.request.urlopen(LABELS_URL))
except Exception:
    class_idx = {str(i): ["", f"class_{i}"] for i in range(1000)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(DEVICE)
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


def predict(model, image_tensor):
    """
    Returns:
    {
        label,
        confidence,
        class_id
    }
    """

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_id = predicted.item()
    label = class_idx.get(str(class_id), ["", f"class_{class_id}"])[1]

    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "class_id": class_id
    }


def get_model_output(model, image_tensor):
    """
    Use this when gradients are required (e.g., Integrated Gradients)
    """
    return model(image_tensor)
