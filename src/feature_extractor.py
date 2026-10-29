"""
feature_extractor.py
--------------------
Feature extraction module for non-invasive hemoglobin estimation.
Combines:
1. Deep latent representations from pretrained EfficientNet-B0 (1280-dim).
2. Clinically grounded color space descriptors (RGB, HSV, CIELAB, erythema ratios; 16-dim).
3. Patient-level multi-ROI pooling and feature caching.
"""

import os
import ast
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

from roi_extractor import parse_bounding_boxes, crop_roi_by_box, auto_detect_roi

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CNN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_color_features(pil_image: Image.Image, filter_glare: bool = True) -> np.ndarray:
    """
    Extract 16-dimensional color descriptor vector from an ROI patch:
    - RGB: mean (R, G, B), std (R, G, B) [6]
    - Ratios: Norm R, Norm G, Norm B, R/G ratio, Erythema Index [5]
    - HSV: mean (H, S, V) [3]
    - LAB: mean a* (green-red chromaticity biomarker), mean L* [2]
    Optional glare filtering ensures specular reflections do not corrupt optical biomarkers.
    """
    img_rgb = pil_image.convert("RGB")
    arr = np.array(img_rgb).astype(np.float64) / 255.0  # normalize to [0, 1]

    # Optional specular glare rejection mask (overexposed white hot-spots)
    valid_mask = np.ones(arr.shape[:2], dtype=bool)
    if filter_glare:
        glare = (arr[..., 0] > 0.94) & (arr[..., 1] > 0.94) & (arr[..., 2] > 0.94)
        if np.sum(~glare) >= 100:
            valid_mask = ~glare

    # RGB statistics
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    mean_r, std_r = float(r[valid_mask].mean()), float(r[valid_mask].std())
    mean_g, std_g = float(g[valid_mask].mean()), float(g[valid_mask].std())
    mean_b, std_b = float(b[valid_mask].mean()), float(b[valid_mask].std())

    # Normalized color ratios
    total_intensity = mean_r + mean_g + mean_b + 1e-6
    norm_r = mean_r / total_intensity
    norm_g = mean_g / total_intensity
    norm_b = mean_b / total_intensity
    rg_ratio = mean_r / (mean_g + 1e-6)
    # Erythema index proxy based on hemoglobin absorption difference
    erythema_index = float(np.log(mean_r + 1e-4) - np.log(mean_g + 1e-4))

    # HSV space
    img_hsv = pil_image.convert("HSV")
    arr_hsv = np.array(img_hsv).astype(np.float64) / 255.0
    mean_h = float(arr_hsv[..., 0][valid_mask].mean())
    mean_s = float(arr_hsv[..., 1][valid_mask].mean())
    mean_v = float(arr_hsv[..., 2][valid_mask].mean())

    # Approximate CIELAB L* and a* (a* strongly tracks vascular redness/erythema)
    # Using standard linear sRGB -> XYZ -> LAB approximation
    X = 0.412453 * mean_r + 0.357580 * mean_g + 0.180423 * mean_b
    Y = 0.212671 * mean_r + 0.715160 * mean_g + 0.072169 * mean_b
    Z = 0.019334 * mean_r + 0.119193 * mean_g + 0.950227 * mean_b

    # Reference white D65: Xn=0.950456, Yn=1.0, Zn=1.088754
    def f_lab(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

    fx = f_lab(X / 0.950456)
    fy = f_lab(Y / 1.0)
    fz = f_lab(Z / 1.088754)

    L_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)  # positive = red, negative = green

    color_vec = np.array([
        mean_r, mean_g, mean_b, std_r, std_g, std_b,
        norm_r, norm_g, norm_b, rg_ratio, erythema_index,
        mean_h, mean_s, mean_v,
        a_star, L_star
    ], dtype=np.float32)

    return color_vec


def generate_erythema_heatmap(
    pil_image: Image.Image,
    alpha: float = 0.50,
    colormap_name: str = "plasma"
) -> Image.Image:
    """
    Computes a dense, pixel-resolution microvascular Erythema Index map:
        EI(x, y) = log(R(x, y) + eps) - log(G(x, y) + eps)
    Visualizes dense hemoglobin absorption and capillary pooling across the tissue.
    """
    import matplotlib.pyplot as plt
    img_rgb = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]

    ei_map = np.log(r + 1e-4) - np.log(g + 1e-4)
    p2 = float(np.percentile(ei_map, 2))
    p98 = float(np.percentile(ei_map, 98))
    ei_norm = np.clip((ei_map - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)

    cmap = plt.get_cmap(colormap_name)
    heat_rgb = (cmap(ei_norm)[..., :3] * 255).astype(np.uint8)

    orig_uint8 = np.array(img_rgb, dtype=np.uint8)
    blended = (alpha * orig_uint8 + (1.0 - alpha) * heat_rgb).astype(np.uint8)
    return Image.fromarray(blended)


class DeepFeatureExtractor:
    """
    Singleton-style feature extractor leveraging pretrained EfficientNet-B0.
    Frozen backbone in eval mode to prevent gradient drift and produce
    consistent 1280-dim representations.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()  # 1280-d output
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.to(self.device)

    @torch.no_grad()
    def extract_deep_features(self, pil_image: Image.Image) -> np.ndarray:
        """Extract 1280-dim deep feature embedding for a single 224x224 PIL image."""
        tensor = CNN_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        feats = self.backbone(tensor)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def generate_cam_heatmap(
        self,
        pil_image: Image.Image,
        weights: Optional[np.ndarray] = None,
        alpha: float = 0.55,
        colormap_name: str = "jet"
    ) -> Image.Image:
        """
        Computes a deep Regression Activation Map (CAM) / Visual Saliency Heatmap
        from EfficientNet-B0's final convolutional layer (features[8]).
        If weights are provided (e.g., reg.coef_[:1280]), computes the exact
        spatial contribution to the hemoglobin prediction.
        """
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        img_crop = pil_image.convert("RGB").resize((224, 224))
        tensor = CNN_TRANSFORM(img_crop).unsqueeze(0).to(self.device)
        feat_maps = self.backbone.features(tensor)  # (1, 1280, 7, 7)

        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float32)[:1280]
            w_tensor = torch.from_numpy(w_arr).to(self.device).view(1, 1280, 1, 1)
            cam = (feat_maps * w_tensor).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
        else:
            cam = feat_maps.mean(dim=1, keepdim=True)

        cam_up = F.interpolate(cam, size=(224, 224), mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        min_v, max_v = float(cam_up.min()), float(cam_up.max())
        cam_norm = (cam_up - min_v) / (max_v - min_v + 1e-8)

        cmap = plt.get_cmap(colormap_name)
        heat_rgb = (cmap(cam_norm)[..., :3] * 255).astype(np.uint8)

        orig_uint8 = np.array(img_crop, dtype=np.uint8)
        blended = (alpha * orig_uint8 + (1.0 - alpha) * heat_rgb).astype(np.uint8)
        return Image.fromarray(blended)

    def generate_explainability_maps(
        self,
        pil_image: Image.Image,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, Image.Image]:
        """
        Generates both Deep Convolutional Saliency Map and Biophysical Erythema Map.
        """
        cam = self.generate_cam_heatmap(pil_image, weights=weights, alpha=0.55)
        erythema = generate_erythema_heatmap(pil_image, alpha=0.50)
        return {
            "cam_heatmap": cam,
            "erythema_heatmap": erythema,
        }

    def extract_single_patch_features(self, pil_image: Image.Image) -> np.ndarray:
        """Combine deep features (1280) and color features (16) -> 1296-dim vector."""
        deep = self.extract_deep_features(pil_image)
        color = extract_color_features(pil_image)
        return np.concatenate([deep, color], axis=0)

    def extract_multibox_features(
        self,
        image: Image.Image,
        boxes: List[List[int]],
        modality: str = "fingernail"
    ) -> Tuple[np.ndarray, List[Image.Image]]:
        """
        Extract features from multiple bounding boxes (e.g. 3 fingers)
        and average them into a pooled subject feature vector.
        If no boxes are provided, runs auto_detect_roi.
        """
        crops = []
        if boxes and len(boxes) > 0:
            for b in boxes:
                crops.append(crop_roi_by_box(image, b, target_size=(224, 224)))
        else:
            crop_auto, _ = auto_detect_roi(image, modality=modality)
            crops.append(crop_auto)

        patch_vectors = [self.extract_single_patch_features(c) for c in crops]
        # Multi-ROI average pooling
        pooled_vector = np.mean(patch_vectors, axis=0)
        return pooled_vector, crops


# Cached singleton instance
_EXTRACTOR_INSTANCE: Optional[DeepFeatureExtractor] = None

def get_feature_extractor() -> DeepFeatureExtractor:
    global _EXTRACTOR_INSTANCE
    if _EXTRACTOR_INSTANCE is None:
        _EXTRACTOR_INSTANCE = DeepFeatureExtractor()
    return _EXTRACTOR_INSTANCE


def extract_and_cache_dataset_features(
    csv_path: str = "data/metadata.csv",
    root_dir: str = ".",
    cache_dir: str = "data/cached_features",
    force_recompute: bool = False
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extracts and caches features for all patients in metadata.csv for both
    'fingernail' and 'skin' modalities.
    """
    os.makedirs(cache_dir, exist_ok=True)
    nail_cache_path = os.path.join(cache_dir, "features_fingernail.npz")
    skin_cache_path = os.path.join(cache_dir, "features_skin.npz")

    if not force_recompute and os.path.isfile(nail_cache_path) and os.path.isfile(skin_cache_path):
        print(f"[FeatureExtractor] Loading cached features from {cache_dir}...")
        data_nail = np.load(nail_cache_path, allow_pickle=True)
        data_skin = np.load(skin_cache_path, allow_pickle=True)
        return {
            "fingernail": {k: data_nail[k] for k in data_nail.files},
            "skin": {k: data_skin[k] for k in data_skin.files},
        }

    import pandas as pd
    df = pd.read_csv(csv_path)
    extractor = get_feature_extractor()

    print(f"[FeatureExtractor] Extracting features for {len(df)} subjects...")

    nail_feats_list, skin_feats_list = [], []
    patient_ids, hb_values, splits = [], [], []

    for idx, row in df.iterrows():
        img_rel = row["image_path"]
        img_full = os.path.join(root_dir, img_rel)
        if not os.path.isfile(img_full):
            continue

        im = Image.open(img_full).convert("RGB")
        nail_boxes = parse_bounding_boxes(str(row.get("NAIL_BOUNDING_BOXES", "")))
        skin_boxes = parse_bounding_boxes(str(row.get("SKIN_BOUNDING_BOXES", "")))

        # Extract multi-finger pooled features
        f_nail, _ = extractor.extract_multibox_features(im, nail_boxes, modality="fingernail")
        f_skin, _ = extractor.extract_multibox_features(im, skin_boxes, modality="skin")

        nail_feats_list.append(f_nail)
        skin_feats_list.append(f_skin)
        patient_ids.append(row["PATIENT_ID"])
        hb_values.append(float(row["hemoglobin"]))
        splits.append(str(row["split"]))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} subjects")

    nail_arr = np.array(nail_feats_list, dtype=np.float32)
    skin_arr = np.array(skin_feats_list, dtype=np.float32)
    pids_arr = np.array(patient_ids)
    hb_arr = np.array(hb_values, dtype=np.float32)
    splits_arr = np.array(splits)

    np.savez_compressed(
        nail_cache_path,
        features=nail_arr,
        patient_ids=pids_arr,
        hemoglobin=hb_arr,
        splits=splits_arr,
    )
    np.savez_compressed(
        skin_cache_path,
        features=skin_arr,
        patient_ids=pids_arr,
        hemoglobin=hb_arr,
        splits=splits_arr,
    )
    print(f"[FeatureExtractor] Successfully cached {len(nail_arr)} patient feature vectors to {cache_dir}!")

    return {
        "fingernail": {
            "features": nail_arr,
            "patient_ids": pids_arr,
            "hemoglobin": hb_arr,
            "splits": splits_arr,
        },
        "skin": {
            "features": skin_arr,
            "patient_ids": pids_arr,
            "hemoglobin": hb_arr,
            "splits": splits_arr,
        },
    }
