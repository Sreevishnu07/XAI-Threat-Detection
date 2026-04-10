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

        xai_results = generate_xai_maps(model, tensor, image_np)

        gradcam_pp = xai_results["gradcam_pp"]
        scorecam = xai_results["scorecam"]
        ig = xai_results["integrated_gradients"]
        focus_scores = xai_results["focus_scores"]

        threat_score = compute_threat_score(confidence, focus_scores, label)
        threat_level = get_threat_level(threat_score)

        def encode(img):
            _, buffer = cv2.imencode(".jpg", img)
            return base64.b64encode(buffer).decode("utf-8")

        return {
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
            "threat_level": threat_level
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
