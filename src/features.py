"""
features.py
-----------
Pretrained CNN (EfficientNet-B0) feature extraction and handcrafted color
feature extraction (RGB, HSV, LAB statistics and physiological color ratios).
Batched for high performance on CPU / GPU.
"""

import os
import time
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── CNN Feature Extractor ──────────────────────────────────────────────────────
class CNNFeatureExtractor:
    """
    Extracts 1280-dimensional feature representations using a frozen pretrained
    EfficientNet-B0 backbone (Global Average Pooled).
    """
    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[CNNFeatureExtractor] Loading pretrained EfficientNet-B0 on {device}...")
        t0 = time.time()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        self.model = backbone.to(self.device)
        print(f"[CNNFeatureExtractor] Model ready in {time.time() - t0:.2f}s")
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract_single(self, pil_image: Image.Image) -> np.ndarray:
        """Extracts 1280-dim feature vector from a single PIL Image."""
        img_rgb = pil_image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, pil_images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extracts 1280-dim features for a list of PIL Images using batching."""
        all_features = []
        for i in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[i:i + batch_size]
            tensors = [self.transform(img.convert("RGB")) for img in batch_imgs]
            batch_tensor = torch.stack(tensors).to(self.device)
            feats = self.model(batch_tensor)
            all_features.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_features, axis=0)


# ── Color Feature Extractor ────────────────────────────────────────────────────
def rgb_to_lab(img_np: np.ndarray) -> np.ndarray:
    """
    Convert RGB image [0, 1] to CIE L*a*b* color space.
    Input: img_np shape (H, W, 3) in [0, 1]
    Output: LAB shape (H, W, 3)
    """
    mask = img_np > 0.04045
    srgb = np.zeros_like(img_np)
    srgb[mask] = ((img_np[mask] + 0.055) / 1.055) ** 2.4
    srgb[~mask] = img_np[~mask] / 12.92

    X = srgb[:, :, 0] * 0.4124564 + srgb[:, :, 1] * 0.3575761 + srgb[:, :, 2] * 0.1804375
    Y = srgb[:, :, 0] * 0.2126729 + srgb[:, :, 1] * 0.7151522 + srgb[:, :, 2] * 0.0721750
    Z = srgb[:, :, 0] * 0.0193339 + srgb[:, :, 1] * 0.1191920 + srgb[:, :, 2] * 0.9503041

    X = X / 0.95047
    Y = Y / 1.00000
    Z = Z / 1.08883

    eps = 0.008856
    kappa = 903.3
    fx = np.where(X > eps, np.cbrt(X), (kappa * X + 16.0) / 116.0)
    fy = np.where(Y > eps, np.cbrt(Y), (kappa * Y + 16.0) / 116.0)
    fz = np.where(Z > eps, np.cbrt(Z), (kappa * Z + 16.0) / 116.0)

    L = np.maximum(0.0, 116.0 * fy - 16.0)
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_hsv(img_np: np.ndarray) -> np.ndarray:
    """Convert RGB image [0, 1] to HSV color space."""
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc

    s = np.where(maxc != 0, deltac / (maxc + 1e-7), 0.0)

    rc = (maxc - r) / (deltac + 1e-7)
    gc = (maxc - g) / (deltac + 1e-7)
    bc = (maxc - b) / (deltac + 1e-7)

    h = np.zeros_like(r)
    mask_r = (r == maxc) & (deltac != 0)
    mask_g = (g == maxc) & (deltac != 0)
    mask_b = (b == maxc) & (deltac != 0)

    h[mask_r] = bc[mask_r] - gc[mask_r]
    h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
    h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
    h = (h / 6.0) % 1.0 * 360.0

    return np.stack([h, s, v], axis=-1)


class ColorFeatureExtractor:
    """
    Extracts 24 medically grounded color features from a cropped ROI:
    - RGB means and standard deviations (6)
    - HSV means and standard deviations (6)
    - CIE LAB means and standard deviations (6)
    - Physiological color ratios & erythema indices (6)
    """
    @staticmethod
    def extract(pil_image: Image.Image) -> np.ndarray:
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb).astype(np.float32) / 255.0
        
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        r_mean, r_std = float(r.mean()), float(r.std())
        g_mean, g_std = float(g.mean()), float(g.std())
        b_mean, b_std = float(b.mean()), float(b.std())

        hsv = rgb_to_hsv(arr)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        h_mean, h_std = float(h.mean()), float(h.std())
        s_mean, s_std = float(s.mean()), float(s.std())
        v_mean, v_std = float(v.mean()), float(v.std())

        lab = rgb_to_lab(arr)
        L, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        L_mean, L_std = float(L.mean()), float(L.std())
        a_mean, a_std = float(a_ch.mean()), float(a_ch.std())
        b_mean, b_std = float(b_ch.mean()), float(b_ch.std())

        total_rgb = r_mean + g_mean + b_mean + 1e-6
        norm_r = r_mean / total_rgb
        norm_g = g_mean / total_rgb
        norm_b = b_mean / total_rgb
        rg_ratio = r_mean / (g_mean + 1e-6)
        erythema_idx = (r_mean - g_mean) / (r_mean + g_mean + 1e-6)
        lab_erythema = a_mean / (L_mean + 1e-6)

        features = [
            r_mean, r_std, g_mean, g_std, b_mean, b_std,
            h_mean, h_std, s_mean, s_std, v_mean, v_std,
            L_mean, L_std, a_mean, a_std, b_mean, b_std,
            norm_r, norm_g, norm_b, rg_ratio, erythema_idx, lab_erythema
        ]
        return np.array(features, dtype=np.float32)


# ── Feature Cache Builder ──────────────────────────────────────────────────────
def extract_and_cache_dataset(manifest_csv="data/dataset_manifest.csv", cache_dir="features"):
    """
    Extracts CNN and Color features for all patients and modalities in batches,
    and caches them as .npz files for fast, reproducible training.
    """
    import pandas as pd
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    print(f"Loading {len(df)} samples from {manifest_csv} for batched feature caching...")

    cnn_extractor = CNNFeatureExtractor()
    color_extractor = ColorFeatureExtractor()

    nail_images = []
    skin_images = []
    patient_ids = []
    splits = []
    labels = []
    nail_color_feats = []
    skin_color_feats = []

    t0 = time.time()
    for idx, row in df.iterrows():
        pid = row["patient_id"]
        split = row["split"]
        hb = float(row["hemoglobin"])
        nail_path = row["nail_crop_primary"]
        skin_path = row["skin_crop_primary"]

        if not os.path.isfile(nail_path) or not os.path.isfile(skin_path):
            continue

        nail_img = Image.open(nail_path).convert("RGB")
        skin_img = Image.open(skin_path).convert("RGB")

        nail_images.append(nail_img)
        skin_images.append(skin_img)
        nail_color_feats.append(color_extractor.extract(nail_img))
        skin_color_feats.append(color_extractor.extract(skin_img))

        patient_ids.append(pid)
        splits.append(split)
        labels.append(hb)

    print(f"Loaded {len(nail_images)} images and extracted color features in {time.time() - t0:.2f}s")
    
    print("Extracting batched CNN features for Fingernails...")
    t1 = time.time()
    nail_cnn = cnn_extractor.extract_batch(nail_images, batch_size=32)
    print(f"Nail CNN features extracted in {time.time() - t1:.2f}s, shape: {nail_cnn.shape}")

    print("Extracting batched CNN features for Skin...")
    t2 = time.time()
    skin_cnn = cnn_extractor.extract_batch(skin_images, batch_size=32)
    print(f"Skin CNN features extracted in {time.time() - t2:.2f}s, shape: {skin_cnn.shape}")

    data_dict = {
        "patient_ids": np.array(patient_ids),
        "splits": np.array(splits),
        "labels": np.array(labels, dtype=np.float32),
        "nail_cnn": nail_cnn,
        "nail_color": np.array(nail_color_feats, dtype=np.float32),
        "skin_cnn": skin_cnn,
        "skin_color": np.array(skin_color_feats, dtype=np.float32),
    }

    cache_file = os.path.join(cache_dir, "features_cache.npz")
    np.savez_compressed(cache_file, **data_dict)
    print(f"\n[DONE] Successfully cached features to {cache_file} in {time.time() - t0:.2f}s total!")
    print(f"  Nail CNN shape:   {data_dict['nail_cnn'].shape}")
    print(f"  Nail Color shape: {data_dict['nail_color'].shape}")
    print(f"  Skin CNN shape:   {data_dict['skin_cnn'].shape}")
    print(f"  Skin Color shape: {data_dict['skin_color'].shape}")
    print(f"  Labels shape:     {data_dict['labels'].shape}")


if __name__ == "__main__":
    extract_and_cache_dataset()
