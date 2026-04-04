"""
=============================================================
  TB Detection - Data Preprocessing Pipeline
  Dataset : Chest X-ray Masks and Labels (Kaggle - Shenzhen)
  Naming  : CHNCXR_XXXX_0.png = Normal
            CHNCXR_XXXX_1.png = Tuberculosis
  Masks   : CHNCXR_XXXX_0_mask.png  (same prefix + _mask)
=============================================================

Your Folder Structure (as downloaded):
    data/
    ├── CXR_png/               ← all X-ray images here
    │   ├── CHNCXR_0001_0.png  (0 = Normal)
    │   ├── CHNCXR_0002_0.png
    │   ├── CHNCXR_0003_1.png  (1 = TB)
    │   └── ...
    └── masks/                 ← all mask images here
        ├── CHNCXR_0001_0_mask.png
        ├── CHNCXR_0002_0_mask.png
        └── ...

Output after running this script:
    processed_data/
    ├── train/
    │   ├── Normal/
    │   └── Tuberculosis/
    ├── val/
    │   ├── Normal/
    │   └── Tuberculosis/
    └── test/
        ├── Normal/
        └── Tuberculosis/
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────
# 👇 Change these paths to where YOUR folders are located
CXR_DIR      = "Lung Segmentation\CXR_png"      # folder containing X-ray images
MASK_DIR     = "Lung Segmentation\masks"        # folder containing mask images
OUTPUT_DIR   = "processed_data"    # where cropped output will be saved

IMG_SIZE     = (224, 224)          # required by DenseNet-121
RANDOM_SEED  = 42
TEST_SIZE    = 0.15                # 15% for testing
VAL_SIZE     = 0.15                # 15% for validation
PADDING      = 10                  # extra pixels around cropped lung
# ──────────────────────────────────────────────────────────────


# ─────────────────── STEP 1: BUILD FILE LIST ──────────────────

def build_dataframe(cxr_dir: str, mask_dir: str) -> pd.DataFrame:
    """
    Scan CXR_png folder, read label from filename last digit,
    find matching mask file, return as a DataFrame.

    Filename pattern:  CHNCXR_XXXX_L.png
                                   └── L = 0 (Normal) or 1 (TB)
    Mask pattern:      CHNCXR_XXXX_L_mask.png
    """
    records = []
    cxr_path  = Path(cxr_dir)
    mask_path = Path(mask_dir)

    all_images = sorted(cxr_path.glob("*.png"))
    print(f"📁 Found {len(all_images)} images in {cxr_dir}")

    skipped = 0
    for img_file in all_images:
        stem = img_file.stem          # e.g. "CHNCXR_0001_0"
        parts = stem.split("_")       # ["CHNCXR", "0001", "0"]

        # ── Read label from last part of filename
        last_digit = parts[-1]        # "0" or "1"
        if last_digit == "0":
            label = "Normal"
        elif last_digit == "1":
            label = "Tuberculosis"
        else:
            print(f"  ⚠️  Skipping {img_file.name} — unknown label '{last_digit}'")
            skipped += 1
            continue

        # ── Find matching mask: CHNCXR_XXXX_L_mask.png
        mask_file = mask_path / f"{stem}_mask.png"
        if not mask_file.exists():
            print(f"  ⚠️  No mask found for {img_file.name}, skipping.")
            skipped += 1
            continue

        records.append({
            "filename"  : img_file.name,
            "stem"      : stem,
            "image_path": str(img_file),
            "mask_path" : str(mask_file),
            "label"     : label,
            "label_int" : int(last_digit),
        })

    df = pd.DataFrame(records)

    print(f"\n📊 Dataset Summary:")
    print(f"   ✅ Total usable : {len(df)}")
    print(f"   🟢 Normal       : {(df['label'] == 'Normal').sum()}")
    print(f"   🔴 Tuberculosis : {(df['label'] == 'Tuberculosis').sum()}")
    if skipped:
        print(f"   ⚠️  Skipped      : {skipped}")
    return df


# ─────────────────── STEP 2: CROP LUNG REGION ─────────────────

def crop_lung_with_mask(image: np.ndarray,
                        mask: np.ndarray,
                        padding: int = PADDING) -> np.ndarray:
    """
    Uses the white region in the mask to find the lung bounding box,
    then crops the original image to that region.

    Why? → Removing background helps the model focus only on lungs.
    """
    # Make mask black & white (binary)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find the outline of white regions (lungs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No lung region found → just resize the full image
        return cv2.resize(image, IMG_SIZE)

    # Get one big bounding box that covers ALL contours (both lungs)
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Add padding, but don't go outside the image
    H, W = image.shape
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(W, int(x_max) + padding)
    y_max = min(H, int(y_max) + padding)

    # Crop and resize to 224×224
    cropped = image[y_min:y_max, x_min:x_max]
    return cv2.resize(cropped, IMG_SIZE)


# ─────────────────── STEP 3: PROCESS & SAVE ───────────────────

def process_and_save(split_df: pd.DataFrame,
                     split_name: str,
                     output_root: str) -> None:
    """
    For each image in the split:
      1. Load grayscale X-ray
      2. Load its mask
      3. Crop lung region
      4. Convert to 3-channel RGB (DenseNet-121 needs 3 channels)
      5. Save to processed_data/split_name/label/filename.png
    """
    errors = []

    for _, row in tqdm(split_df.iterrows(),
                       total=len(split_df),
                       desc=f"  Processing [{split_name:5s}]"):

        # Create output folder if needed
        save_dir = os.path.join(output_root, split_name, row["label"])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, row["filename"])

        if os.path.exists(save_path):
            continue  # already done, skip

        try:
            # Load both as grayscale
            image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            mask  = cv2.imread(row["mask_path"],  cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise FileNotFoundError(f"Cannot read image: {row['image_path']}")
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {row['mask_path']}")

            # If mask size doesn't match image, resize mask to match
            if image.shape != mask.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Crop lung region using mask
            cropped = crop_lung_with_mask(image, mask)

            # Grayscale → 3-channel RGB (copy same channel 3 times)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)

            # Save
            cv2.imwrite(save_path, rgb)

        except Exception as e:
            errors.append((row["filename"], str(e)))

    if errors:
        print(f"\n  ⚠️  {len(errors)} errors in [{split_name}]:")
        for fname, err in errors[:5]:
            print(f"     {fname}: {err}")


# ─────────────────── STEP 4: VISUALIZE SAMPLES ────────────────

def visualize_samples(df: pd.DataFrame, n: int = 6) -> None:
    """
    Show side-by-side: Original | Mask | Cropped Lung
    for a few random images so you can verify it's working.
    """
    samples = df.sample(min(n, len(df)), random_state=RANDOM_SEED)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    fig.suptitle("Preprocessing Preview: Original → Mask → Cropped Lung",
                 fontsize=13, fontweight="bold")

    for i, (_, row) in enumerate(samples.iterrows()):
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(row["mask_path"],  cv2.IMREAD_GRAYSCALE)
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Overlay mask on image (semi-transparent)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        overlay = cv2.addWeighted(image, 0.65, binary, 0.35, 0)

        cropped = crop_lung_with_mask(image, mask)

        col_titles = ["Original X-ray", "Mask Overlay", f"Cropped Lung\n({row['label']})"]
        imgs = [image, overlay, cropped]

        for j, (ax, img, title) in enumerate(zip(axes[i], imgs, col_titles)):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10,
                         color="#4CAF50" if row["label"] == "Normal" else "#F44336")
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "preprocessing_preview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Preview image saved → {out_path}")


# ─────────────────── MAIN ─────────────────────────────────────

def main():
    print("=" * 60)
    print("  TB Detection — Data Preprocessing")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Build file list with labels
    print("\n🔍 Scanning files...")
    df = build_dataframe(CXR_DIR, MASK_DIR)

    if df.empty:
        print("\n❌ No data found! Please check:")
        print(f"   CXR_DIR  = '{CXR_DIR}'")
        print(f"   MASK_DIR = '{MASK_DIR}'")
        return

    # ── 2. Train / Val / Test split (stratified = equal ratio in each split)
    train_val, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val["label"],
        random_state=RANDOM_SEED
    )

    print(f"\n📂 Split Results:")
    print(f"   Train : {len(train_df):>4} images  "
          f"(Normal: {(train_df['label']=='Normal').sum()}, "
          f"TB: {(train_df['label']=='Tuberculosis').sum()})")
    print(f"   Val   : {len(val_df):>4} images  "
          f"(Normal: {(val_df['label']=='Normal').sum()}, "
          f"TB: {(val_df['label']=='Tuberculosis').sum()})")
    print(f"   Test  : {len(test_df):>4} images  "
          f"(Normal: {(test_df['label']=='Normal').sum()}, "
          f"TB: {(test_df['label']=='Tuberculosis').sum()})")

    # Save CSVs so you always know which image is in which split
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(OUTPUT_DIR, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, "test.csv"),  index=False)
    print(f"\n✅ Split CSVs saved to '{OUTPUT_DIR}/'")

    # ── 3. Crop & save all images
    print("\n✂️  Cropping lung regions and saving...")
    for name, split in [("train", train_df),
                         ("val",   val_df),
                         ("test",  test_df)]:
        process_and_save(split, name, OUTPUT_DIR)

    # ── 4. Show a visual preview
    print("\n🖼️  Generating preprocessing preview...")
    visualize_samples(df, n=6)

    print("\n" + "=" * 60)
    print("  ✅ Preprocessing COMPLETE!")
    print(f"  Cropped images saved in: '{OUTPUT_DIR}/'")
    print("  Next step → run: python 2_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()