from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.model.model import load_model, preprocess_image, predict

app = FastAPI()

# OPTIONAL but helps avoid weird issues
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
        contents = await file.read()

        # 🔥 DEBUG CHECK
        if contents is None or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = preprocess_image(image)

        result = predict(model, tensor)

        return {
            "filename": file.filename,
            "prediction": result
        }

    except Exception as e:
        return {"error": str(e)}
