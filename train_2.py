"""
=============================================================
  TB Detection - CheXNet Training Script (TensorFlow/Keras)
  Model  : DenseNet-121 (CheXNet-style)
  Task   : Binary Classification — Normal vs Tuberculosis
  Fixed  : float32 memory error + reduced batch size
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR  = "processed_data"   
MODEL_SAVE_DIR = "models"
LOG_DIR        = "logs"
IMG_SIZE       = (224, 224)
INPUT_SHAPE    = (224, 224, 3)
BATCH_SIZE     = 8        # ✅ fixed from 16 → 8
EPOCHS         = 50
LR_INITIAL     = 1e-4
LR_FINE_TUNE   = 1e-5
FREEZE_EPOCHS  = 10
RANDOM_SEED    = 42

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)


def make_generators():
    def preprocess_input(x):
        # ✅ FIXED: float32 prevents memory allocation error
        x = x.astype("float32") / 255.0
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return (x - mean) / std

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest"
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True, seed=RANDOM_SEED
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "val"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False
    )

    print(f"\n🗂  Class indices: {train_gen.class_indices}")
    print(f"   Train: {train_gen.n} | Val: {val_gen.n} | Test: {test_gen.n}")
    return train_gen, val_gen, test_gen


def build_chexnet(freeze_base=True):
    base = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = base(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="tb_prediction")(x)
    return Model(inputs, outputs, name="CheXNet_TB"), base


def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )


def get_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, f"best_model_{phase}.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_auc", patience=8, mode="max",
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        callbacks.CSVLogger(os.path.join(LOG_DIR, f"training_log_{phase}.csv"))
    ]


def plot_history(hist1, hist2):
    metrics = ["loss", "accuracy", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CheXNet Training History", fontsize=14, fontweight="bold")

    for key in ["loss","accuracy","auc","val_loss","val_accuracy","val_auc"]:
        pass

    combined = {k: hist1.history.get(k,[]) + hist2.history.get(k,[])
                for k in ["loss","accuracy","auc","val_loss","val_accuracy","val_auc"]}
    ep = range(1, len(combined["loss"]) + 1)
    p2 = len(hist1.history["loss"])

    for ax, m in zip(axes, metrics):
        ax.plot(ep, combined[m], label="Train", color="#2196F3", lw=2)
        ax.plot(ep, combined[f"val_{m}"], label="Val", color="#F44336", lw=2, linestyle="--")
        ax.axvline(x=p2, color="green", linestyle=":", lw=1.5, label="Fine-tune starts")
        ax.set_title(m.upper(), fontweight="bold")
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Training curves saved → {path}")


def train():
    print("=" * 60)
    print("  CheXNet TB Detection — Training")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {gpus[0].name}")
    else:
        print("⚠️  No GPU — running on CPU")

    train_gen, val_gen, test_gen = make_generators()
    labels  = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    print(f"⚖️  Class weights: {class_weights}")

    # Phase 1
    print(f"\n🔒 Phase 1: Head only ({FREEZE_EPOCHS} epochs)")
    model, base = build_chexnet(freeze_base=True)
    compile_model(model, LR_INITIAL)
    hist1 = model.fit(train_gen, validation_data=val_gen,
                      epochs=FREEZE_EPOCHS, class_weight=class_weights,
                      callbacks=get_callbacks("phase1"), verbose=1)

    # Phase 2
    print(f"\n🔓 Phase 2: Fine-tuning all layers (up to {EPOCHS} epochs)")
    base.trainable = True
    compile_model(model, LR_FINE_TUNE)
    hist2 = model.fit(train_gen, validation_data=val_gen,
                      epochs=EPOCHS, initial_epoch=FREEZE_EPOCHS,
                      class_weight=class_weights,
                      callbacks=get_callbacks("phase2"), verbose=1)

    model.save(os.path.join(MODEL_SAVE_DIR, "chexnet_tb_final.keras"))
    print(f"\n✅ Model saved → {MODEL_SAVE_DIR}/chexnet_tb_final.keras")

    plot_history(hist1, hist2)

    print("\n📊 Test Evaluation:")
    results = model.evaluate(test_gen, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"   {name:12s}: {val:.4f}")

    print("\n🎉 Done! Next → python 3_evaluate.py")
    return model, test_gen


if __name__ == "__main__":
    model, test_gen = train()