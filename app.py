import io
import os
import numpy as np
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMG_SIZE = (160, 160)   # 🔥 faster than 224
THRESHOLD = 0.5

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────
# LOAD MODEL (ON STARTUP)
# ─────────────────────────────────────────────

print("🚀 Loading model...")
model = load_model("best_model_phase2.keras")
print("✅ Model loaded")

# 🔥 Warmup (VERY IMPORTANT)
dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
model.predict(dummy)
print("🔥 Model warmed up")

# ─────────────────────────────────────────────
# SERVE UI
# ─────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"

    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return HTMLResponse("<h2>Dashboard not found ❌</h2>")

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "CheXNet DenseNet-121"}

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("📥 File received")

        # Load image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Resize
        img = img.resize(IMG_SIZE)
        img = np.array(img).astype(np.float32)

        # Normalize (IMPORTANT)
        img = img / 255.0
        img = (img - MEAN) / STD

        # Ensure 3 channels
        if img.shape[-1] != 3:
            img = np.stack([img]*3, axis=-1)

        # Add batch dimension
        inp = np.expand_dims(img, axis=0)

        # Safety check
        if inp.shape != (1, IMG_SIZE[0], IMG_SIZE[1], 3):
            return JSONResponse(
                {"success": False, "error": "Invalid input shape"},
                status_code=500
            )

        # Prediction
        pred = model.predict(inp, verbose=0)

        prob_tb = float(pred[0][0])

        # NaN safety
        if np.isnan(prob_tb):
            prob_tb = 0.0

        prob_nor = 1 - prob_tb
        is_tb = prob_tb >= THRESHOLD
        conf = prob_tb if is_tb else prob_nor

        print("✅ Prediction done:", prob_tb)

        return JSONResponse({
            "success": True,
            "is_tb": is_tb,
            "prob_tb": round(prob_tb * 100, 1),
            "prob_normal": round(prob_nor * 100, 1),
            "confidence": round(conf * 100, 1),
            "label": "HIGH RISK OF TUBERCULOSIS" if is_tb else "NORMAL — No TB Detected",
            "finding": (
                f"TB probability: {prob_tb*100:.1f}%" if is_tb
                else f"Normal probability: {prob_nor*100:.1f}%"
            ),
            "gradcam_b64": None,   # disabled for speed
            "bbox": None,
            "filename": file.filename,
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )