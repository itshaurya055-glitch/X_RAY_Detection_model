"""
=============================================================
  TB Detection - Grad-CAM Visualization (Fixed)
  Works with nested DenseNet121 model structure
=============================================================
  Usage: python gradcam.py
         python gradcam.py --image path/to/xray.png
=============================================================
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_PATH    = "models/best_model_phase2.keras"
PROCESSED_DIR = "processed_data"
OUTPUT_DIR    = "gradcam_results"
IMG_SIZE      = (224, 224)
NUM_SAMPLES   = 8
THRESHOLD     = 0.35
MEAN          = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD           = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────── IMAGE LOADING ───────────────────────────

def load_and_preprocess(img_path):
    """Load image, resize, normalize. Returns (model_input, display_img)."""
    img = cv2.imread(img_path)
    if img is None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, IMG_SIZE)
    display    = img_resized.copy()

    # Normalize
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - MEAN) / STD
    return np.expand_dims(img_norm, axis=0), display


# ─────────────────── GRAD-CAM (FIXED) ────────────────────────

def make_gradcam_heatmap(img_array, model):
    """
    Generate Grad-CAM heatmap.
    Handles nested DenseNet121 inside the outer model.
    """
    # Get the DenseNet submodel
    densenet = model.get_layer("densenet121")

    # Build a standalone model: densenet_input → [last_conv_out, densenet_out]
    last_conv_layer = densenet.get_layer("conv5_block16_2_conv")

    # Inner model: densenet input → last conv output
    inner_model = tf.keras.Model(
        inputs=densenet.input,
        outputs=[last_conv_layer.output, densenet.output]
    )

    # Get our classification head layers (after densenet)
    # We'll manually pass through them
    gap      = model.get_layer("global_average_pooling2d")
    drop1    = model.get_layer("dropout")
    dense1   = model.get_layer("dense")
    bn       = model.get_layer("batch_normalization")
    drop2    = model.get_layer("dropout_1")
    out_layer= model.get_layer("tb_prediction")

    with tf.GradientTape() as tape:
        # Step 1: pass through densenet, watch conv output
        img_tensor = tf.cast(img_array, tf.float32)
        conv_output, densenet_output = inner_model(img_tensor, training=False)
        tape.watch(conv_output)

        # Step 2: pass densenet output through classification head
        x = gap(densenet_output)
        x = drop1(x, training=False)
        x = dense1(x)
        x = bn(x, training=False)
        x = drop2(x, training=False)
        predictions = out_layer(x)
        loss = predictions[:, 0]

    # Compute gradients of prediction w.r.t. last conv output
    grads = tape.gradient(loss, conv_output)           # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)) # (C,)

    # Weight conv outputs by gradients
    conv_output = conv_output[0]                                    # (H, W, C)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]           # (H, W, 1)
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU + normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(original, heatmap, alpha=0.45):
    """Superimpose Grad-CAM heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr    = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    superimposed    = cv2.addWeighted(original_bgr, 1-alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


# ─────────────────── SINGLE IMAGE ────────────────────────────

def visualize_single(img_path, model, true_label=None):
    """Grad-CAM for one image — 3 panel plot."""
    img_array, display = load_and_preprocess(img_path)

    prob       = model.predict(img_array, verbose=0)[0][0]
    pred_label = "Tuberculosis" if prob >= THRESHOLD else "Normal"
    confidence = prob if pred_label == "Tuberculosis" else 1 - prob

    heatmap  = make_gradcam_heatmap(img_array, model)
    overlaid = overlay_heatmap(display, heatmap)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * cv2.resize(
        heatmap, IMG_SIZE)), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, img, title in zip(axes,
                               [display, heatmap_rgb, overlaid],
                               ["Original X-ray", "Grad-CAM Heatmap", "Overlay"]):
        ax.imshow(img); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    color = "#F44336" if pred_label == "Tuberculosis" else "#4CAF50"
    title = f"Prediction: {pred_label}  ({confidence*100:.1f}%)"
    if true_label:
        tick = "✓" if true_label == pred_label else "✗"
        title += f"  |  True: {true_label} {tick}"
    fig.suptitle(title, fontsize=13, fontweight="bold", color=color)

    plt.tight_layout()
    fname = os.path.splitext(os.path.basename(img_path))[0]
    path  = os.path.join(OUTPUT_DIR, f"gradcam_{fname}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → {path}")


# ─────────────────── BATCH FROM TEST SET ─────────────────────

def visualize_batch(model, n=NUM_SAMPLES):
    """Grad-CAM grid for N random test images."""
    from pathlib import Path
    import random

    all_imgs = []
    for label in ["Normal", "Tuberculosis"]:
        d = Path(PROCESSED_DIR) / "test" / label
        if d.exists():
            all_imgs += [(str(p), label) for p in d.glob("*.png")]

    if not all_imgs:
        print(f"❌ No test images found in {PROCESSED_DIR}/test/")
        return

    random.seed(42)
    samples = random.sample(all_imgs, min(n, len(all_imgs)))

    cols = 4
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols,
                             figsize=(cols * 3.5, rows * 6))
    fig.suptitle("Grad-CAM — TB Detection (Top: Original | Bottom: Heatmap)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()
    ax_idx = 0

    for img_path, true_label in samples:
        img_array, display = load_and_preprocess(img_path)
        prob       = model.predict(img_array, verbose=0)[0][0]
        pred_label = "Tuberculosis" if prob >= THRESHOLD else "Normal"
        confidence = prob if pred_label == "Tuberculosis" else 1 - prob
        correct    = pred_label == true_label
        color      = "#4CAF50" if correct else "#F44336"
        tick       = "✓" if correct else "✗"

        heatmap  = make_gradcam_heatmap(img_array, model)
        overlaid = overlay_heatmap(display, heatmap)

        # Original
        axes[ax_idx].imshow(display)
        axes[ax_idx].set_title(f"True: {true_label}", fontsize=9, fontweight="bold")
        for sp in axes[ax_idx].spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        axes[ax_idx].axis("off")
        ax_idx += 1

        # Overlay
        axes[ax_idx].imshow(overlaid)
        axes[ax_idx].set_title(
            f"Pred: {pred_label} {tick}\n{confidence*100:.0f}%",
            fontsize=9, color=color)
        for sp in axes[ax_idx].spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        axes[ax_idx].axis("off")
        ax_idx += 1

    while ax_idx < len(axes):
        axes[ax_idx].set_visible(False)
        ax_idx += 1

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradcam_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Grad-CAM grid saved → {path}")


# ─────────────────── MAIN ─────────────────────────────────────

def main(single_image=None):
    print("=" * 60)
    print("  CheXNet TB Detection — Grad-CAM Visualization")
    print("=" * 60)

    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")

    # Verify densenet layer exists
    try:
        model.get_layer("densenet121")
        print("✅ DenseNet121 layer found")
    except:
        print("❌ Could not find densenet121 layer")
        return

    if single_image:
        visualize_single(single_image, model)
    else:
        print(f"\n🎨 Generating Grad-CAM for {NUM_SAMPLES} test samples...")
        visualize_batch(model, n=NUM_SAMPLES)

    print(f"\n🎉 Done! Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()
    main(single_image=args.image)