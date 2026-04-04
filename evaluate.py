"""
=============================================================
  TB Detection - Model Evaluation & Metrics
  Generates: Confusion Matrix, ROC Curve, PR Curve,
             Classification Report, Per-sample predictions
=============================================================
  Run after: 2_train.py
  Usage    : python 3_evaluate.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    matthews_corrcoef
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
PROCESSED_DIR  = "processed_data"
MODEL_PATH     = "models/best_model_phase2.keras"
OUTPUT_DIR     = "evaluation_results"
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 16
THRESHOLD      = 0.527
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_input(x):
    x = x.astype("float32") / 255.0  # ← float32 fix
    return (x - MEAN) / STD


def load_test_data():
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = gen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return test_gen


def get_predictions(model, test_gen):
    """Run inference and return true labels, probabilities, predicted classes."""
    test_gen.reset()
    probs = model.predict(test_gen, verbose=1).flatten()
    y_true = test_gen.classes
    y_pred = (probs >= THRESHOLD).astype(int)
    return y_true, probs, y_pred


# ─────────────── PLOT FUNCTIONS ──────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")

    # Compute per-class metrics from cm
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ax.set_xlabel(
        f"Predicted Label\n\nSensitivity (Recall): {sensitivity:.3f}  |  "
        f"Specificity: {specificity:.3f}", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Confusion matrix saved: {path}")
    return tn, fp, fn, tp


def plot_roc_curve(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=120,
               color="red", zorder=5,
               label=f"Optimal threshold = {optimal_threshold:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — TB Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ ROC curve saved: {path}  |  AUC = {roc_auc:.4f}")
    return roc_auc, optimal_threshold


def plot_precision_recall_curve(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4CAF50", lw=2.5,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")
    ax.axhline(y=y_true.mean(), color="k", linestyle="--", lw=1.5,
               label=f"Baseline (prevalence = {y_true.mean():.2f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — TB Detection",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ PR curve saved: {path}  |  AP = {ap:.4f}")
    return ap


def plot_prediction_distribution(y_true, probs):
    """Histogram of prediction probabilities separated by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs[y_true == 0], bins=30, alpha=0.7,
            color="#2196F3", label="Normal (True)", edgecolor="white")
    ax.hist(probs[y_true == 1], bins=30, alpha=0.7,
            color="#F44336", label="Tuberculosis (True)", edgecolor="white")
    ax.axvline(THRESHOLD, color="k", linestyle="--", lw=2,
               label=f"Threshold = {THRESHOLD}")
    ax.set_xlabel("Predicted Probability (TB)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Score Distribution by Class",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Score distribution saved: {path}")


def print_metrics_report(y_true, y_pred, probs, tn, fp, fn, tp):
    """Print and save a comprehensive metrics table."""
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)
    f1          = f1_score(y_true, y_pred)
    mcc         = matthews_corrcoef(y_true, y_pred)
    roc_auc     = auc(*roc_curve(y_true, probs)[:2])

    metrics = {
        "Accuracy"        : f"{accuracy:.4f}",
        "AUC-ROC"         : f"{roc_auc:.4f}",
        "Sensitivity"     : f"{sensitivity:.4f}  ← True Positive Rate (TB detection)",
        "Specificity"     : f"{specificity:.4f}  ← True Negative Rate",
        "Precision (PPV)" : f"{ppv:.4f}",
        "NPV"             : f"{npv:.4f}",
        "F1 Score"        : f"{f1:.4f}",
        "MCC"             : f"{mcc:.4f}",
        "TP"              : str(tp),
        "TN"              : str(tn),
        "FP"              : str(fp),
        "FN"              : str(fn),
    }

    print("\n" + "=" * 55)
    print("  📊 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("=" * 55)

    # Detailed sklearn report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Tuberculosis"]))

    # Save to CSV
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    csv_path = os.path.join(OUTPUT_DIR, "metrics_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Metrics saved: {csv_path}")


def evaluate():
    print("=" * 60)
    print("  CheXNet TB Detection — Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load data
    test_gen = load_test_data()
    class_names = list(test_gen.class_indices.keys())
    print(f"   Classes: {class_names}")

    # Predict
    print("\n🔍 Running inference on test set...")
    y_true, probs, y_pred = get_predictions(model, test_gen)

    # Plots
    tn, fp, fn, tp = plot_confusion_matrix(y_true, y_pred, class_names)
    roc_auc, optimal_thresh = plot_roc_curve(y_true, probs)
    plot_precision_recall_curve(y_true, probs)
    plot_prediction_distribution(y_true, probs)

    # Report
    print_metrics_report(y_true, y_pred, probs, tn, fp, fn, tp)

    print(f"\n💡 Optimal Threshold (Youden's J): {optimal_thresh:.4f}")
    print(f"   (Default used: {THRESHOLD})")
    print(f"\n🎉 Evaluation complete! Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    evaluate()