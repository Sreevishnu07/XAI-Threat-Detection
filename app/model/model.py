import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

# Load ImageNet class labels once
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
class_idx = json.load(urllib.request.urlopen(LABELS_URL))


def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
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
    image = transform(image).unsqueeze(0)
    return image


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)

        probs = torch.nn.functional.softmax(output, dim=1)  # 🔥 probabilities
        confidence, predicted = torch.max(probs, 1)

        label = class_idx[str(predicted.item())][1]

    return {
        "label": label,
        "confidence": round(confidence.item(), 4)
    }
