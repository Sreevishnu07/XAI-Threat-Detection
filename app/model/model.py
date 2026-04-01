import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()
