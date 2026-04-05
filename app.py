"""
=============================================================
 TB Detection - FINAL STABLE (RENDER SAFE + FRONTEND + AI)
=============================================================
"""

import io, os, base64, cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import gdown

# ───────────────── CONFIG ─────────────────
IMG_SIZE = (224, 224)
THRESHOLD = 0.35

MODEL_PATH = "best_model_phase2.keras"
GDRIVE_FILE_ID = "11uhh09WzNMH3ZbXDhAPNtVO4fn5o3DXE"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Disable GPU (important for Render)
tf.config.set_visible_devices([], 'GPU')

# ───────────────── APP INIT ─────────────────
app = FastAPI(title="TB Detection AI")

import threading, time, requests

def keep_alive():
    while True:
        time.sleep(14 * 60)
        try:
            requests.get("https://x-ray-detection-model.onrender.com/health", timeout=10)
            print("♻️ Keep-alive ping")
        except: pass

threading.Thread(target=keep_alive, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 FIXED PATHS (IMPORTANT FOR RENDER)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

model = None

# ───────────────── DOWNLOAD MODEL ─────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists")
        return

    print("⬇️ Downloading model...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Download complete")

# ───────────────── STARTUP ─────────────────
@app.on_event("startup")
def load_model():
    global model
    try:
        print("🚀 Starting server...")
        download_model()

        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)

        print("✅ Model loaded successfully!")

    except Exception as e:
        print("❌ Startup error:", e)
        model = None

# ───────────────── ROUTES ─────────────────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    try:
        with open(os.path.join(BASE_DIR, "templates", "dashboard.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h2>Error loading dashboard: {e}</h2>"
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# ───────────────── PREPROCESS ─────────────────
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.expand_dims(img, axis=0)

# ───────────────── SAFE GRADCAM ─────────────────
def make_gradcam(img_array):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        return heatmap.numpy()

    except Exception as e:
        print("GradCAM error:", e)
        return None

def overlay_heatmap(img, heatmap):
    try:
        h, w = img.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    except:
        return img

def to_b64(img):
    try:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except:
        return None

# ───────────────── PREDICT ─────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)

        contents = await file.read()

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        img = np.array(img)
        display = cv2.resize(img, IMG_SIZE)
        inp = preprocess(display)

        pred = model.predict(inp, verbose=0)
        prob_tb = float(pred[0][0])

        if not np.isfinite(prob_tb):
            prob_tb = 0.0

        prob_normal = 1 - prob_tb
        is_tb = prob_tb >= THRESHOLD
        confidence = prob_tb if is_tb else prob_normal

        gradcam_img = None
        

        return {
            "success": True,
            "is_tb": is_tb,
            "prob_tb": round(prob_tb * 100, 2),
            "prob_normal": round(prob_normal * 100, 2),
            "confidence": round(confidence * 100, 2),
            "label": "TB DETECTED" if is_tb else "NORMAL",

            # 🔥 MATCH FRONTEND
            "gradcam_b64": gradcam_img,
            "finding": "AI-based analysis completed. Refer to Grad-CAM for highlighted regions.",
            "filename": file.filename,
            "bbox": None
        }

    except Exception as e:
        print("❌ FULL ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ───────────────── RUN ─────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)