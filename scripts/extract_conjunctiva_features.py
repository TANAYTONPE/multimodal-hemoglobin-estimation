"""
extract_conjunctiva_features.py
--------------------------------
Extracts 1296-dimensional features (1280 EfficientNet-B0 + 16 color biomarkers)
for all processed palpebral conjunctiva images in conjunctiva_manifest.csv,
and caches them into data/cached_features/features_conjunctiva.npz.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from feature_extractor import get_feature_extractor

def main():
    manifest_path = os.path.join("data", "conjunctiva_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    print(f"[FeatureExtractor] Processing {len(df)} conjunctival cases...")

    extractor = get_feature_extractor()
    features_list = []
    patient_ids = []
    hb_values = []
    splits = []

    for idx, row in df.iterrows():
        crop_path = row["crop_path"]
        if not os.path.isfile(crop_path):
            print(f"Warning: {crop_path} missing, skipping.")
            continue

        im = Image.open(crop_path).convert("RGB")
        feats = extractor.extract_single_patch_features(im)

        features_list.append(feats)
        patient_ids.append(row["patient_id"])
        hb_values.append(float(row["hemoglobin"]))
        splits.append(str(row["split"]))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} images")

    features_arr = np.array(features_list, dtype=np.float32)
    pids_arr = np.array(patient_ids)
    hb_arr = np.array(hb_values, dtype=np.float32)
    splits_arr = np.array(splits)

    out_path = os.path.join("data", "cached_features", "features_conjunctiva.npz")
    np.savez_compressed(
        out_path,
        features=features_arr,
        patient_ids=pids_arr,
        hemoglobin=hb_arr,
        splits=splits_arr,
    )
    print(f"[FeatureExtractor] Saved {features_arr.shape} features to {out_path}")

if __name__ == "__main__":
    main()
