"""
=============================================================
  TB Detection - MobileNetV2 (Lightweight Model)
  Fast CPU training — ~15-20 mins
  Model size: ~14MB (vs 88MB DenseNet)
  Uses existing processed_data/ folder
=============================================================
  Usage: python train_mobilenet.py
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
PROCESSED_DIR  = "processed_data"
MODEL_SAVE_DIR = "models"
LOG_DIR        = "logs/mobilenet"
IMG_SIZE       = (224, 224)
INPUT_SHAPE    = (224, 224, 3)
BATCH_SIZE     = 8
FREEZE_EPOCHS  = 8
FINETUNE_EPOCHS= 20
LR_HEAD        = 1e-3
LR_FINETUNE    = 1e-4
RANDOM_SEED    = 42
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
# ──────────────────────────────────────────────────────────────

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)

# Limit CPU threads to avoid freezing laptop
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ─────────────────── DATA GENERATORS ──────────────────────────

def preprocess_input(x):
    """float32 fix + ImageNet normalization."""
    x = x.astype("float32") / 255.0
    return (x - MEAN) / STD


def make_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=8,
        zoom_range=0.08,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
    )
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    print(f"\n📂 Class mapping : {train_gen.class_indices}")
    print(f"   Train : {train_gen.n} | Val : {val_gen.n} | Test : {test_gen.n}")
    return train_gen, val_gen, test_gen


# ─────────────────── BUILD MODEL ──────────────────────────────

def build_mobilenet(freeze_base=True):
    """
    MobileNetV2 — lightweight CheXNet alternative:
      - 14MB vs 88MB (DenseNet)
      - 5x faster inference on CPU
      - Similar accuracy on small datasets

    Architecture:
        Input (224×224×3)
            ↓
        MobileNetV2 (pretrained ImageNet)
            ↓
        GlobalAveragePooling2D
            ↓
        Dropout(0.3)
            ↓
        Dense(128, relu)
            ↓
        BatchNorm + Dropout(0.2)
            ↓
        Dense(1, sigmoid) → 0=Normal, 1=TB
    """
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
        alpha=1.0       # full size MobileNetV2
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="tb_prediction")(x)

    model = Model(inputs, output, name="MobileNetV2_TB")
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    print(f"   Compiled — lr={lr}")


# ─────────────────── CALLBACKS ────────────────────────────────

def get_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, f"mobilenet_best_{phase}.keras"),
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc", patience=6,
            mode="max", restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"log_{phase}.csv")
        ),
    ]


# ─────────────────── TRAINING ─────────────────────────────────

def plot_history(hist1, hist2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MobileNetV2 TB Detection — Training History",
                 fontsize=14, fontweight="bold")

    combined = {
        k: hist1.history.get(k, []) + hist2.history.get(k, [])
        for k in ["loss", "accuracy", "auc",
                  "val_loss", "val_accuracy", "val_auc"]
    }
    ep = range(1, len(combined["loss"]) + 1)
    p2 = len(hist1.history["loss"])

    for ax, m in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(ep, combined[m],           label="Train", color="#2196F3", lw=2)
        ax.plot(ep, combined[f"val_{m}"],  label="Val",   color="#F44336",
                lw=2, linestyle="--")
        ax.axvline(x=p2, color="green", linestyle=":", lw=1.5,
                   label="Fine-tune starts")
        ax.set_title(m.upper(), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Curves saved → {path}")


def train():
    print("=" * 60)
    print("  MobileNetV2 TB Detection — CPU Training")
    print("=" * 60)

    # Data
    train_gen, val_gen, test_gen = make_generators()

    # Class weights
    labels  = train_gen.classes
    weights = compute_class_weight("balanced",
                                   classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    print(f"\n⚖️  Class weights: {class_weights}")

    # ── Phase 1: Train head only
    print(f"\n🔒 Phase 1: Head only ({FREEZE_EPOCHS} epochs)")
    model, base = build_mobilenet(freeze_base=True)
    compile_model(model, LR_HEAD)

    total     = sum(tf.size(w).numpy() for w in model.weights)
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"   Total params     : {total:,}")
    print(f"   Trainable params : {trainable:,} (head only)")
    print(f"   Frozen params    : {total-trainable:,} (MobileNetV2 base)\n")

    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FREEZE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
        verbose=1
    )

    # ── Phase 2: Fine-tune all layers
    print(f"\n🔓 Phase 2: Fine-tuning all layers (up to {FINETUNE_EPOCHS} more epochs)")
    base.trainable = True
    compile_model(model, LR_FINETUNE)

    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FREEZE_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=FREEZE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
        verbose=1
    )

    # Save
    final_path = os.path.join(MODEL_SAVE_DIR, "mobilenet_tb_final.keras")
    model.save(final_path)
    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"\n✅ Model saved → {final_path}  ({size_mb:.1f} MB)")

    # Plot
    plot_history(hist1, hist2)

    # Test evaluation
    print("\n📊 Test Set Evaluation:")
    results = model.evaluate(test_gen, verbose=1)
    print()
    for name, val in zip(model.metrics_names, results):
        print(f"   {name:<12} : {val:.4f}")

    print(f"\n🎉 Done! Model size: {size_mb:.1f} MB")
    print(f"   vs DenseNet-121 : ~88 MB")
    print(f"   Speedup on CPU  : ~5x faster inference")
    print(f"\n   Next → update app.py to use mobilenet_best_phase2.keras")
    return model


if __name__ == "__main__":
    train()