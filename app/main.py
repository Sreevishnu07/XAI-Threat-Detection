from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

import numpy as np
import cv2
import base64

from app.model.model import load_model, preprocess_image, predict
from app.model.gradcam import generate_gradcam
from app.model.threat import compute_threat_score, get_threat_level

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        tensor = preprocess_image(image)
        result = predict(model, tensor)

        return {
            "filename": file.filename,
            "prediction": result
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-xai")
async def predict_xai(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image")

        tensor = preprocess_image(image)
        result = predict(model, tensor)

        confidence = result["confidence"]
        label = result["label"]  

        image_resized = image.resize((224, 224))
        image_np = np.array(image_resized) / 255.0

        cam_image, focus_score = generate_gradcam(model, tensor, image_np)


        threat_score = compute_threat_score(confidence, focus_score, label)  # 🔥 FIX 2
        threat_level = get_threat_level(threat_score)

        _, buffer = cv2.imencode(".jpg", cam_image)
        cam_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "prediction": result,
            "gradcam": cam_base64,
            "focus_score": round(focus_score, 4),
            "threat_score": round(threat_score, 4),
            "threat_level": threat_level
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
