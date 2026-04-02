from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model.model import load_model, preprocess_image, predict

app = FastAPI()

model = load_model()


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = preprocess_image(image)

        result = predict(model, tensor)

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        return {"error": str(e)}
