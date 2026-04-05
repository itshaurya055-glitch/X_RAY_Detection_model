"""
=============================================================
  TB Detection - FastAPI Backend (Render Deployment)
  Fixed: GradCAM, preprocessing, prediction bugs
=============================================================
"""

import io, os, base64, cv2, numpy as np, requests
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import matplotlib; matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
IMG_SIZE  = (224, 224)
THRESHOLD = 0.35
MEAN      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD       = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Google Drive model URL ──
# Replace FILE_ID with your actual Google Drive file ID
GDRIVE_FILE_ID = "11uhh09WzNMH3ZbXDhAPNtVO4fn5o3DXE"
MODEL_PATH     = "best_model_phase2.keras"
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="TB Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─────────────────── DOWNLOAD MODEL ──────────────────────────

def download_model_gdrive(file_id: str, dest: str):
    """Download large file from Google Drive handling confirmation token."""
    if os.path.exists(dest):
        print(f"✅ Model already exists: {dest}")
        return

    print("⬇️  Downloading model from Google Drive...")
    session = requests.Session()
    url     = "https://drive.google.com/uc?export=download"

    # First request to get confirmation token for large files
    response = session.get(url, params={"id": file_id}, stream=True)

    # Find confirmation token in cookies or response text
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(url,
                               params={"id": file_id, "confirm": token},
                               stream=True)

    # Save file
    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"   Downloaded: {total/1024/1024:.1f} MB", end="\r")

    print(f"\n✅ Model downloaded! Size: {total/1024/1024:.1f} MB")


# Download and load model
download_model_gdrive(GDRIVE_FILE_ID, MODEL_PATH)

print("📦 Loading model...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")


# ─────────────────── SERVE DASHBOARD ─────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve dashboard.html — for local testing only.
       On Netlify+Render setup, dashboard is hosted separately."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("""
        <h2>TB Detection API is running ✅</h2>
        <p>POST /predict — upload chest X-ray for prediction</p>
        <p>GET  /health  — check server status</p>
    """)


# ─────────────────── PREPROCESSING ───────────────────────────

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Resize, normalize using ImageNet stats, add batch dim."""
    img = cv2.resize(img_rgb, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.expand_dims(img, axis=0)   # (1, 224, 224, 3)


# ─────────────────── GRAD-CAM (FIXED) ────────────────────────

def make_gradcam(img_array: np.ndarray) -> np.ndarray:
    """
    Generate Grad-CAM heatmap.
    Handles nested DenseNet121 inside outer model correctly.
    """
    try:
        # Get DenseNet submodel
        densenet  = model.get_layer("densenet121")
        last_conv = densenet.get_layer("conv5_block16_2_conv")

        # Inner model: densenet input → [last conv out, densenet out]
        inner_model = tf.keras.Model(
            inputs=densenet.input,
            outputs=[last_conv.output, densenet.output]
        )

        # Get classification head layers
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

            # Pass through classification head manually
            x = gap(dn_out)
            x = drop1(x, training=False)
            x = dense1(x)
            x = bn(x, training=False)
            x = drop2(x, training=False)
            preds = out_layer(x)
            loss  = preds[:, 0]

        # Compute gradients
        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = np.squeeze(
            (conv_out[0] @ pooled[..., tf.newaxis]).numpy()
        )

        # ReLU + normalize
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    except Exception as e:
        print(f"❌ GradCAM Error: {e}")
        return None


def overlay_heatmap(img_rgb: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    hm   = cv2.applyColorMap(
               np.uint8(255 * cv2.resize(heatmap, (w, h))),
               cv2.COLORMAP_JET)
    sup  = cv2.addWeighted(
               cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1-alpha,
               hm, alpha, 0)
    return cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)


def get_bbox(heatmap: np.ndarray, w: int, h: int, thr: float = 0.5):
    binary      = (cv2.resize(heatmap, (w, h)) > thr).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return {
        "left"  : round(x  / w * 100, 1),
        "top"   : round(y  / h * 100, 1),
        "width" : round(bw / w * 100, 1),
        "height": round(bh / h * 100, 1),
    }


def to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────── PREDICT ENDPOINT ────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"\n📥 Received: {file.filename}")

        # ── Load image
        raw     = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(pil_img)
        print(f"   Original shape: {img_rgb.shape}")

        # ── Resize for display
        display = cv2.resize(img_rgb, IMG_SIZE)

        # ── Preprocess for model
        inp = preprocess(display)
        print(f"   Model input shape: {inp.shape}")

        # ── Predict
        pred    = model.predict(inp, verbose=0)
        prob_tb = float(pred[0][0])

        # Handle NaN/Inf
        if not np.isfinite(prob_tb):
            prob_tb = 0.0

        prob_nor = 1.0 - prob_tb
        is_tb    = prob_tb >= THRESHOLD
        conf     = prob_tb if is_tb else prob_nor

        print(f"   TB prob: {prob_tb:.4f} | Normal prob: {prob_nor:.4f}")
        print(f"   Prediction: {'TUBERCULOSIS' if is_tb else 'NORMAL'}")

        # ── Grad-CAM (always run, not just for high prob)
        print("🔥 Running Grad-CAM...")
        heatmap     = make_gradcam(inp)
        gradcam_b64 = None
        bbox        = None

        if heatmap is not None:
            overlaid    = overlay_heatmap(display, heatmap)
            gradcam_b64 = to_b64(overlaid)
            bbox        = get_bbox(heatmap, *IMG_SIZE) if is_tb else None
            print("✅ Grad-CAM generated")
        else:
            print("⚠️  Grad-CAM failed — skipping")

        # ── Finding text
        if is_tb:
            finding = (
                f"Findings highly suggestive of active pulmonary TB. "
                f"TB probability: {prob_tb*100:.1f}%. "
                f"Please consult a specialist immediately."
            )
        else:
            finding = (
                f"No significant pulmonary abnormalities detected. "
                f"Normal probability: {prob_nor*100:.1f}%. "
                f"Routine follow-up recommended."
            )

        return JSONResponse({
            "success"     : True,
            "is_tb"       : is_tb,
            "prob_tb"     : round(prob_tb  * 100, 1),
            "prob_normal" : round(prob_nor * 100, 1),
            "confidence"  : round(conf     * 100, 1),
            "label"       : "HIGH RISK OF TUBERCULOSIS" if is_tb else "NORMAL — No TB Detected",
            "finding"     : finding,
            "gradcam_b64" : gradcam_b64,
            "bbox"        : bbox,
            "filename"    : file.filename,
        })

    except Exception as e:
        import traceback
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


# ─────────────────── HEALTH CHECK ────────────────────────────

@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "model"    : "CheXNet DenseNet-121",
        "threshold": THRESHOLD,
        "img_size" : IMG_SIZE,
    }


# ─────────────────── RUN ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)