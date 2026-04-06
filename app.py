"""
=============================================================
  TB Detection - Fast Prediction Only (No GradCAM)
  Model  : DenseNet-121
  Deploy : Render free tier
=============================================================
"""

import io, os, base64, cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import gdown
import threading, time, requests

# ───────────────── CONFIG ─────────────────
IMG_SIZE  = (224, 224)
THRESHOLD = 0.35
MEAN      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD       = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH     = "model.keras"
GDRIVE_FILE_ID = "11uhh09WzNMH3ZbXDhAPNtVO4fn5o3DXE"
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

# Disable GPU on Render
tf.config.set_visible_devices([], 'GPU')

# ───────────────── APP ─────────────────
app = FastAPI(title="TB Detection API")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

# Serve static JS files
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

model = None
model_ready = False

# ───────────────── KEEP ALIVE ─────────────────
def keep_alive():
    """Ping server every 14 mins to prevent Render sleep."""
    time.sleep(60)  # wait 1 min after startup first
    while True:
        try:
            requests.get(
                "https://x-ray-detection-model.onrender.com/health",
                timeout=10
            )
            print("♻️ Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        time.sleep(14 * 60)

threading.Thread(target=keep_alive, daemon=True).start()

# ───────────────── MODEL DOWNLOAD ─────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        if size_mb > 10:  # valid model file
            print(f"✅ Model exists ({size_mb:.1f} MB)")
            return
        else:
            print("⚠️ Model file too small, re-downloading...")
            os.remove(MODEL_PATH)

    print("⬇️ Downloading model from Google Drive...")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            MODEL_PATH,
            quiet=False,
            fuzzy=True
        )
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"✅ Downloaded! Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise e

# ───────────────── STARTUP ─────────────────
@app.on_event("startup")
def startup():
    global model, model_ready
    try:
        model_ready = False
        print("🚀 Server starting...")
        download_model()

        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Warmup run — makes first prediction fast
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)

        model_ready = True

        print("✅ Model loaded and warmed up!")
        print(f"   Input  : {model.input_shape}")
        print(f"   Output : {model.output_shape}")

    except Exception as e:
        print(f"❌ Startup error: {e}")
        model = None
        model_ready = False

# ───────────────── SERVE UI ─────────────────
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    # Try templates folder first, then root
    for path in [
        os.path.join(BASE_DIR, "templates", "dashboard.html"),
        os.path.join(BASE_DIR, "dashboard.html"),
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return HTMLResponse("""
        <h2 style="font-family:Arial;padding:40px">
            TB Detection API is running ✅<br><br>
            <small>POST /predict — upload X-ray<br>
            GET  /health  — check status</small>
        </h2>
    """)

# ───────────────── HEALTH ─────────────────
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "model"       : "DenseNet-121",
        "threshold"   : THRESHOLD,
    }

# ───────────────── PREPROCESS ─────────────────
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.expand_dims(img, axis=0)

# ───────────────── PREDICT ─────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check model loaded and warmed up
        if not model_ready or model is None:
            return JSONResponse(
                {"success": False, "error": "Model still loading, retry in 10 seconds"},
                status_code=503
            )

        print(f"📥 Received: {file.filename}")

        # Read & validate image
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return JSONResponse(
                {"success": False, "error": "Invalid image file"},
                status_code=400
            )

        # Preprocess
        img_rgb = np.array(pil_img)
        display = cv2.resize(img_rgb, IMG_SIZE)
        inp     = preprocess(display)

        # Predict
        pred    = model.predict(inp, verbose=0)
        prob_tb = float(pred[0][0])

        # Safety check
        if not np.isfinite(prob_tb):
            prob_tb = 0.0

        prob_normal = 1.0 - prob_tb
        is_tb       = prob_tb >= THRESHOLD
        confidence  = prob_tb if is_tb else prob_normal

        print(f"   TB: {prob_tb*100:.1f}% | Normal: {prob_normal*100:.1f}% → {'TB' if is_tb else 'NORMAL'}")

        # Finding text
        if is_tb:
            finding = (
                f"Findings suggestive of active pulmonary TB. "
                f"TB probability: {prob_tb*100:.1f}%. "
                f"Please consult a specialist immediately."
            )
            label = "HIGH RISK OF TUBERCULOSIS"
        else:
            finding = (
                f"No significant pulmonary abnormalities detected. "
                f"Normal probability: {prob_normal*100:.1f}%. "
                f"Routine follow-up recommended."
            )
            label = "NORMAL — No TB Detected"

        return JSONResponse({
            "success"    : True,
            "is_tb"      : is_tb,
            "prob_tb"    : round(prob_tb     * 100, 1),
            "prob_normal": round(prob_normal * 100, 1),
            "confidence" : round(confidence  * 100, 1),
            "label"      : label,
            "finding"    : finding,
            "gradcam_b64": None,   # disabled for speed
            "bbox"       : None,
            "filename"   : file.filename,
        })

    except Exception as e:
        import traceback
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

# ───────────────── RUN ─────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)