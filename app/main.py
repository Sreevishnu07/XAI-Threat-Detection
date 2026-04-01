from fastapi import FastAPI
from PIL import Image
from app.model.model import load_model, preprocess_image, predict

app = FastAPI()

model = load_model()

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/test-model")
def test_model():
    img = Image.new("RGB", (224, 224))  # dummy image
    tensor = preprocess_image(img)
    pred = predict(model, tensor)
    return {"prediction": pred}
