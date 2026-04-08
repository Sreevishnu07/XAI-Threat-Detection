import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

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
    return transform(image).unsqueeze(0)

def get_top_k_predictions(probs, k=5):
    top_probs, top_idxs = torch.topk(probs, k)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        label = class_idx[str(idx.item())][1]
        results.append({
            "label": label,
            "confidence": prob.item()
        })

    return results


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)

        # Top-5 predictions
        top_k = get_top_k_predictions(probs, k=5)

        # Top-1 (for display)
        top_label = top_k[0]["label"]
        top_conf = top_k[0]["confidence"]

    return {
        "label": top_label,
        "confidence": round(top_conf, 4),
        "top_k": [
            {
                "label": item["label"],
                "confidence": round(item["confidence"], 4)
            }
            for item in top_k
        ]
    }
