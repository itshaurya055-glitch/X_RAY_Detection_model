"""
=============================================================
  TB Detection - FastAPI Backend (Render Deployment)
  Serves dashboard.html + prediction API
=============================================================
  Local  : python app.py  → http://localhost:8000
  Render : Auto-deployed from GitHub
=============================================================
"""

import io, os, base64, cv2, numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import matplotlib; matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "models", "best_model_phase2.keras")
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.35
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="TB Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── Load model at startup ──
import os
import requests
import tensorflow as tf

MODEL_URL = "https://drive.google.com/uc?id=11uhh09WzNMH3ZbXDhAPNtVO4fn5o3DXE"
MODEL_PATH = "model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇ Downloading model...")

        response = requests.get(MODEL_URL, stream=True, timeout=60)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

        print("✅ Model downloaded!")

# 🔥 Run this BEFORE loading model
download_model()

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ── Serve dashboard.html at root ──
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>dashboard.html not found</h2>", status_code=404)


# ── Helpers ──
def preprocess(img_rgb):
    img = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    return np.expand_dims((img - MEAN) / STD, axis=0)


def make_gradcam(img_array):
    densenet    = model.get_layer("densenet121")
    last_conv   = densenet.get_layer("conv5_block16_2_conv")
    inner_model = tf.keras.Model(inputs=densenet.input,
                                 outputs=[last_conv.output, densenet.output])
    gap       = model.get_layer("global_average_pooling2d")
    drop1     = model.get_layer("dropout")
    dense1    = model.get_layer("dense")
    bn        = model.get_layer("batch_normalization")
    drop2     = model.get_layer("dropout_1")
    out_layer = model.get_layer("tb_prediction")

    with tf.GradientTape() as tape:
        img_t = tf.cast(img_array, tf.float32)
        conv_out, dn_out = inner_model(img_t, training=False)
        tape.watch(conv_out)
        x = gap(dn_out)
        x = drop1(x, training=False)
        x = dense1(x)
        x = bn(x, training=False)
        x = drop2(x, training=False)
        preds = out_layer(x)
        loss  = preds[:, 0]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = np.squeeze((conv_out[0] @ pooled[..., tf.newaxis]).numpy())
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(img_rgb, heatmap, alpha=0.45):
    h, w  = img_rgb.shape[:2]
    hm    = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(heatmap, (w, h))),
                cv2.COLORMAP_JET)
    sup   = cv2.addWeighted(
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1-alpha,
                hm, alpha, 0)
    return cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)


def get_bbox(heatmap, w, h, thr=0.5):
    binary = (cv2.resize(heatmap, (w, h)) > thr).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return {"left":  round(x/w*100, 1), "top":    round(y/h*100, 1),
            "width": round(bw/w*100, 1), "height": round(bh/h*100, 1)}


def to_b64(img_rgb):
    buf = io.BytesIO()
    Image.fromarray(img_rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Predict endpoint ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_rgb  = np.array(
            Image.open(io.BytesIO(await file.read())).convert("RGB"))
        display  = cv2.resize(img_rgb, IMG_SIZE)
        inp      = preprocess(display)

        prob_tb  = float(model.predict(inp, verbose=0)[0][0])
        prob_nor = 1 - prob_tb
        is_tb    = prob_tb >= THRESHOLD
        conf     = prob_tb if is_tb else prob_nor

        if is_tb:
            heatmap = make_gradcam(inp)
            gradcam_b64 = to_b64(overlay_heatmap(display, heatmap))
            bbox = get_bbox(heatmap, *IMG_SIZE)
        else:
            gradcam_b64 = None
            bbox = None

        finding = (
            f"Findings highly suggestive of active pulmonary TB. "
            f"TB probability: {prob_tb*100:.1f}%. Consult a specialist immediately."
            if is_tb else
            f"No significant pulmonary abnormalities detected. "
            f"Normal probability: {prob_nor*100:.1f}%. Routine follow-up recommended."
        )

        return JSONResponse({
            "success"     : True,
            "is_tb"       : is_tb,
            "prob_tb"     : round(prob_tb * 100, 1),
            "prob_normal" : round(prob_nor * 100, 1),
            "confidence"  : round(conf * 100, 1),
            "label"       : "HIGH RISK OF TUBERCULOSIS" if is_tb else "NORMAL — No TB Detected",
            "finding"     : finding,
            "gradcam_b64" : gradcam_b64,
            "bbox"        : bbox,
            "filename"    : file.filename,
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok", "model": "CheXNet DenseNet-121"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)