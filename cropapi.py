from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from predict_with_prevention import predict_disease

app = FastAPI()

@app.post("/predict_with_prevention")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert("RGB")

        disease, confidence, tips, top_preds = predict_disease(img)

        return {
            "prediction": disease,
            "confidence": round(confidence, 2),
            "preventions": tips,
            "top_predictions": [
                {
                    "class": cls,
                    "confidence": round(conf, 2)
                }
                for cls, conf in top_preds
            ]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )