# backend/app.py
import os
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.predict import predict_and_explain

app = FastAPI(
    title="CXR Prediction API",
    version="0.1.0",
)

# CORS – allow your frontend dev server(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path and device (override via environment variables if needed)
MODEL_PATH = os.environ.get("MODEL_PATH", "backend/weights/best.pt")
DEVICE = os.environ.get("DEVICE", "cpu")


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # basic file-type guard
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")

    content = await file.read()

    # Ensure model file exists
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Model weights not found at '{MODEL_PATH}'. Train a model or update MODEL_PATH.",
        )

    try:
        res = predict_and_explain(
            MODEL_PATH,
            content,
            device=DEVICE,
            labels_map=None,  # or pass a dict if you want custom label mapping
        )

        overlay_b64 = base64.b64encode(res["overlay"]).decode("utf-8")

        return {
            "label": res["label"],                     # e.g. "Normal chest X-ray"
            "pred_idx": res["pred_idx"],               # numeric class index
            "probs": res["probs"],                     # raw probabilities (for debugging)
            "percentages": res["percentages"],         # percentages per class (0–100)
          
            "summary": res["summary"],                 # natural-language summary
            "heatmap_legend": res["heatmap_legend"],   # explains colors on heatmap
            "overlay_b64": overlay_b64,                # Grad-CAM overlay as base64 JPEG
        }

    except HTTPException:
        # pass through HTTPExceptions unchanged
        raise
    except Exception as e:
        # convert other exceptions into a clean 500
        raise HTTPException(status_code=500, detail=str(e))
