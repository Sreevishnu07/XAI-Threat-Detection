import hashlib
import json
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import cv2
import base64

from app.model.model import load_model, preprocess_image, predict
from app.model.gradcam import generate_xai_maps
from app.model.threat import (
    compute_threat_score,
    get_threat_level,
    get_trust_level,
    compute_uncertainty
)
from app.model.explainer import generate_explanation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

try:
    redis_client = redis.Redis(host="xai-redis", port=6379, decode_responses=True)
    redis_client.ping()
except:
    redis_client = None


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

        image_hash = hashlib.md5(contents).hexdigest()

        if redis_client:
            cached = redis_client.get(image_hash)
            if cached:
                data = json.loads(cached)
                data["cache"] = True
                return data

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

        xai_results = generate_xai_maps(model, tensor, image_np)

        gradcam_pp = xai_results["gradcam_pp"]
        scorecam = xai_results["scorecam"]
        ig = xai_results["integrated_gradients"]
        focus_scores = xai_results["focus_scores"]

        maps = xai_results["maps"]

        threat_score, consistency = compute_threat_score(
            confidence,
            focus_scores,
            label,
            maps["gradcam_pp"],
            maps["scorecam"],
            maps["integrated_gradients"]
        )

        threat_level = get_threat_level(threat_score)
        trust_level = get_trust_level(consistency)

        uncertainty = compute_uncertainty(confidence, consistency, label)

        explanation = generate_explanation(
            label,
            confidence,
            focus_scores,
            consistency,
            threat_level,
            uncertainty
        )

        def encode(img):
            _, buffer = cv2.imencode(".jpg", img)
            return base64.b64encode(buffer).decode("utf-8")

        response_data = {
            "prediction": result,

            "xai": {
                "gradcam_pp": encode(gradcam_pp),
                "scorecam": encode(scorecam),
                "integrated_gradients": encode(ig)
            },

            "focus_scores": {
                k: round(v, 4) for k, v in focus_scores.items()
            },

            "threat_score": round(threat_score, 4),
            "threat_level": threat_level,

            "consistency": round(consistency, 4),
            "trust_level": trust_level,

            "uncertainty": uncertainty,

            "explanation": explanation,

            "cache": False
        }

        if redis_client:
            redis_client.setex(image_hash, 300, json.dumps(response_data))

        return response_data

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
