"""
ingest_eyes_dataset.py
-----------------------
Ingests, crops, and structures the Eyes-defy-anemia dataset (India & Italy cohorts)
for HemoLens Model 3 (Conjunctiva Hb Estimator).
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

CACHE_DATASET_DIR = r"C:\Users\tanay\.cache\kagglehub\datasets\harshwardhanfartale\eyes-defy-anemia\versions\1\dataset anemia"
DEST_CROPS_DIR = os.path.join("data", "crops", "conjunctiva")
DEST_RAW_DIR = os.path.join("data", "eyes_dataset")
MANIFEST_OUT = os.path.join("data", "conjunctiva_manifest.csv")

def extract_tight_crop(rgba_img: Image.Image, target_size=(224, 224)) -> Image.Image:
    arr = np.array(rgba_img)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3]
        pos = np.argwhere(alpha > 0)
        if len(pos) > 0:
            ymin, xmin = int(pos[:, 0].min()), int(pos[:, 1].min())
            ymax, xmax = int(pos[:, 0].max()), int(pos[:, 1].max())
            pad_y = max(1, int((ymax - ymin) * 0.05))
            pad_x = max(1, int((xmax - xmin) * 0.05))
            ymin = max(0, ymin - pad_y)
            ymax = min(arr.shape[0], ymax + pad_y)
            xmin = max(0, xmin - pad_x)
            xmax = min(arr.shape[1], xmax + pad_x)
            crop_arr = arr[ymin:ymax, xmin:xmax, :3]
            return Image.fromarray(crop_arr).resize(target_size, Image.Resampling.BILINEAR)
    return rgba_img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)

def process_cohort(cohort_name: str, cache_dir: str):
    cohort_dir = os.path.join(cache_dir, cohort_name)
    excel_path = os.path.join(cohort_dir, f"{cohort_name}.xlsx")
    if not os.path.exists(excel_path):
        print(f"Excel not found: {excel_path}")
        return []

    df = pd.read_excel(excel_path)
    df["Hgb_clean"] = pd.to_numeric(df["Hgb"], errors="coerce")
    df = df.dropna(subset=["Hgb_clean"])

    records = []
    for _, row in df.iterrows():
        p_num = int(row["Number"])
        hb = float(row["Hgb_clean"])
        gender = str(row.get("Gender", "Unknown"))
        age = row.get("Age", np.nan)

        p_folder = os.path.join(cohort_dir, str(p_num))
        if not os.path.isdir(p_folder):
            continue

        # Look for palpebral image
        palp_candidates = glob.glob(os.path.join(p_folder, "*palpebral*.png"))
        # prefer pure palpebral over forniceal_palpebral if available
        pure_palp = [p for p in palp_candidates if "forniceal" not in os.path.basename(p).lower()]
        palp_file = pure_palp[0] if pure_palp else (palp_candidates[0] if palp_candidates else None)

        # Look for raw photo (jpg)
        jpgs = glob.glob(os.path.join(p_folder, "*.jpg"))
        raw_file = jpgs[0] if jpgs else None

        if not palp_file:
            continue

        patient_id = f"{cohort_name.lower()}_{p_num}"
        crop_filename = f"{patient_id}_palpebral.jpg"
        crop_dest = os.path.join(DEST_CROPS_DIR, crop_filename)

        raw_dest = None
        if raw_file and os.path.exists(raw_file):
            raw_filename = f"{patient_id}_raw.jpg"
            raw_dest = os.path.join(DEST_RAW_DIR, raw_filename)
            shutil.copyfile(raw_file, raw_dest)

        # Crop and save palpebral tissue
        try:
            with Image.open(palp_file) as im:
                crop = extract_tight_crop(im)
                crop.save(crop_dest, quality=95)
        except Exception as e:
            print(f"Error processing {palp_file}: {e}")
            continue

        records.append({
            "patient_id": patient_id,
            "cohort": cohort_name,
            "patient_num": p_num,
            "hemoglobin": round(hb, 2),
            "gender": gender,
            "age": age,
            "crop_path": os.path.relpath(crop_dest).replace("\\", "/"),
            "raw_image_path": os.path.relpath(raw_dest).replace("\\", "/") if raw_dest else "",
        })

    print(f"[{cohort_name}] Processed {len(records)} valid patients.")
    return records

def main():
    os.makedirs(DEST_CROPS_DIR, exist_ok=True)
    os.makedirs(DEST_RAW_DIR, exist_ok=True)

    records = []
    records.extend(process_cohort("India", CACHE_DATASET_DIR))
    records.extend(process_cohort("Italy", CACHE_DATASET_DIR))

    manifest_df = pd.DataFrame(records)
    print(f"\nTotal conjunctival records: {len(manifest_df)}")
    print(manifest_df["hemoglobin"].describe())

    # Train / Val / Test Split (70% / 15% / 15% stratified by Hb binned)
    bins = [0, 10.0, 12.0, 25.0]
    labels = ["severe_mod", "mild", "normal"]
    manifest_df["hb_bin"] = pd.cut(manifest_df["hemoglobin"], bins=bins, labels=labels, right=False)

    train_df, temp_df = train_test_split(manifest_df, test_size=0.30, random_state=42, stratify=manifest_df["hb_bin"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["hb_bin"])

    manifest_df.loc[train_df.index, "split"] = "train"
    manifest_df.loc[val_df.index, "split"] = "val"
    manifest_df.loc[test_df.index, "split"] = "test"
    manifest_df.drop(columns=["hb_bin"], inplace=True)

    manifest_df.to_csv(MANIFEST_OUT, index=False)
    print(f"\nSaved manifest to {MANIFEST_OUT}")
    print(manifest_df["split"].value_counts())

if __name__ == "__main__":
    main()
