"""
inference.py
------------
Reusable, production-grade inference pipeline for non-invasive hemoglobin estimation.
Handles:
- Image file validation and quality checks (brightness, contrast, resolution, color channels)
- ROI extraction and preprocessing for Fingernail and Skin
- Feature extraction using frozen pretrained EfficientNet-B0
- Model loading (cached) and Hb estimation in g/dL
- Clear WHO anemia classification and medical disclaimers
- Graceful handling of missing models / pending modalities
"""

import os
import pickle
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List

from features import CNNFeatureExtractor, ColorFeatureExtractor

# Global cache for loaded model pipelines
_MODEL_CACHE = {}
_CNN_EXTRACTOR = None


def get_cnn_extractor():
    global _CNN_EXTRACTOR
    if _CNN_EXTRACTOR is None:
        _CNN_EXTRACTOR = CNNFeatureExtractor()
    return _CNN_EXTRACTOR


def get_cached_model(model_filename: str):
    global _MODEL_CACHE
    path = os.path.join("saved_models", model_filename)
    if path not in _MODEL_CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        with open(path, "rb") as f:
            _MODEL_CACHE[path] = pickle.load(f)
    return _MODEL_CACHE[path]


# ── Quality Validation ─────────────────────────────────────────────────────────
def validate_image_quality(img: Image.Image, modality_name: str = "Image") -> List[str]:
    """
    Checks resolution, illumination, contrast, and color balance.
    Returns a list of warning strings (empty if image passes all checks).
    """
    warnings = []
    w, h = img.size
    
    # Resolution check
    if w < 40 or h < 40:
        warnings.append(f"⚠ Low resolution ({w}x{h}): ROI should ideally be at least 60x60 px.")

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())

    if brightness < 0.20:
        warnings.append(f"⚠ Low lighting detected (brightness {brightness:.2f}): Image appears underexposed.")
    elif brightness > 0.88:
        warnings.append(f"⚠ High glare detected (brightness {brightness:.2f}): Image appears overexposed.")

    if contrast < 0.04:
        warnings.append(f"⚠ Low contrast detected ({contrast:.2f}): Features may be indistinct or blurred.")

    # Check for near-monochrome / grayscale input
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg_diff = np.abs(r - g).mean()
    rb_diff = np.abs(r - b).mean()
    if rg_diff < 0.015 and rb_diff < 0.015:
        warnings.append("⚠ Image appears monochromatic/grayscale. Erythema evaluation requires color channels.")

    return warnings


# ── ROI Extraction ─────────────────────────────────────────────────────────────
def extract_roi(img: Image.Image, modality: str = "nail") -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extracts or normalizes the Region of Interest (ROI).
    If the image is already a tight crop (< 250px on either side), uses the full image.
    If it is a full photo (e.g. 800x600), extracts the canonical center/upper region.
    Returns:
        (cropped_image, (xmin, ymin, xmax, ymax))
    """
    w, h = img.size
    # Already a tight crop
    if w <= 250 and h <= 250:
        return img, (0, 0, w, h)

    # Full photo ROI extraction
    if modality.lower() == "nail":
        # Fingernail typically in middle third horizontal, top third vertical
        xmin = int(w * 0.40)
        xmax = int(w * 0.65)
        ymin = int(h * 0.35)
        ymax = int(h * 0.60)
    elif modality.lower() == "skin":
        xmin = int(w * 0.50)
        xmax = int(w * 0.75)
        ymin = int(h * 0.60)
        ymax = int(h * 0.85)
    else:  # conjunctiva / eye
        xmin = int(w * 0.25)
        xmax = int(w * 0.75)
        ymin = int(h * 0.30)
        ymax = int(h * 0.70)

    crop = img.crop((xmin, ymin, xmax, ymax))
    return crop, (xmin, ymin, xmax, ymax)


# ── Anemia Classification ──────────────────────────────────────────────────────
def classify_hb(hb_g_dl: float) -> Tuple[str, str, str]:
    """
    World Health Organization (WHO) Hemoglobin Assessment:
    Returns (Category, CSS Chip Class, Explanation).
    """
    if hb_g_dl >= 12.0:
        return "Normal", "chip-green", "Hemoglobin level is within standard clinical reference range (≥ 12.0 g/dL)."
    elif hb_g_dl >= 10.0:
        return "Mild Anemia", "chip-yellow", "Hemoglobin is moderately reduced (10.0 – 11.9 g/dL)."
    elif hb_g_dl >= 7.0:
        return "Moderate Anemia", "chip-orange", "Significant reduction in hemoglobin (7.0 – 9.9 g/dL)."
    else:
        return "Severe Anemia", "chip-red", "Critical low hemoglobin (< 7.0 g/dL). Requires immediate medical attention."


# ── Main Inference Results Dataclass ───────────────────────────────────────────
@dataclass
class InferenceResult:
    predicted_hb: Optional[float]
    classification: str
    chip_class: str
    clinical_note: str
    model_name: str
    modality: str
    feature_dim: int
    quality_warnings: List[str]
    roi_images: dict
    is_available: bool
    status_message: str


# ── Single-Modality Inference ──────────────────────────────────────────────────
def predict_single_modality(
    image: Image.Image,
    modality: str = "Fingernail"
) -> InferenceResult:
    """Runs inference for a single modality image (Fingernail or Skin)."""
    modality_key = modality.strip().lower()
    
    if "conjunctiv" in modality_key:
        return InferenceResult(
            predicted_hb=None,
            classification="Model Pending",
            chip_class="chip-dark",
            clinical_note="No paired clinical conjunctival dataset is available for training in this iteration.",
            model_name="Model 3: Conjunctival Hb Estimator",
            modality="Conjunctiva",
            feature_dim=1280,
            quality_warnings=[],
            roi_images={},
            is_available=False,
            status_message="Model 3 is an architectural branch whose clinical training is pending paired eye dataset availability."
        )

    if "nail" in modality_key or "finger" in modality_key:
        model_file = "model1_nail.pkl"
        model_name = "Model 1: Fingernail Hb Estimator"
        mod_tag = "nail"
    elif "skin" in modality_key:
        model_file = "model2_skin.pkl"
        model_name = "Model 2: Skin Hb Estimator"
        mod_tag = "skin"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    # Validate image
    warnings = validate_image_quality(image, modality)
    roi_crop, roi_box = extract_roi(image, mod_tag)

    # Load model pipeline
    try:
        model = get_cached_model(model_file)
    except FileNotFoundError:
        return InferenceResult(
            predicted_hb=None,
            classification="Model Unavailable",
            chip_class="chip-dark",
            clinical_note="Model checkpoint not found. Please run training first.",
            model_name=model_name,
            modality=modality,
            feature_dim=1280,
            quality_warnings=warnings,
            roi_images={mod_tag: roi_crop},
            is_available=False,
            status_message="Model weights missing from saved_models directory."
        )

    # Extract features
    extractor = get_cnn_extractor()
    features = extractor.extract_single(roi_crop).reshape(1, -1)  # (1, 1280)

    # Predict
    pred_val = float(model.predict(features)[0])
    label, chip_cls, note = classify_hb(pred_val)

    return InferenceResult(
        predicted_hb=round(pred_val, 2),
        classification=label,
        chip_class=chip_cls,
        clinical_note=note,
        model_name=model_name,
        modality=modality,
        feature_dim=1280,
        quality_warnings=warnings,
        roi_images={mod_tag: roi_crop},
        is_available=True,
        status_message="Successfully computed inference via frozen EfficientNet-B0 + Ridge head."
    )


# ── Multimodal Inference ───────────────────────────────────────────────────────
def predict_multimodal(
    nail_image: Optional[Image.Image] = None,
    skin_image: Optional[Image.Image] = None,
    conjunctiva_image: Optional[Image.Image] = None,
    fusion_type: str = "Nail + Skin"
) -> InferenceResult:
    """Runs multimodal fusion inference."""
    # Check for 3-modality Model 6 request
    if fusion_type == "Three-Modality (Nail + Skin + Conjunctiva)" or conjunctiva_image is not None:
        return InferenceResult(
            predicted_hb=None,
            classification="Model Pending",
            chip_class="chip-dark",
            clinical_note="The three-branch Model 6 forward architecture is fully validated via smoke testing, but full tripartite training requires a paired dataset with conjunctival clinical Hb ground-truth.",
            model_name="Model 6: Main Proposed 3-Branch Model",
            modality="Nail + Skin + Conjunctiva",
            feature_dim=3840,
            quality_warnings=[],
            roi_images={},
            is_available=False,
            status_message="Three-modality model training is pending paired conjunctival clinical labels. Use Nail + Skin dual fusion for validated predictions."
        )

    # Dual Modality: Nail + Skin (Model 4)
    if nail_image is None or skin_image is None:
        return InferenceResult(
            predicted_hb=None,
            classification="Missing Input",
            chip_class="chip-dark",
            clinical_note="Multimodal Model 4 requires both Fingernail and Skin images to be provided.",
            model_name="Model 4: Nail + Skin Fusion",
            modality="Nail + Skin",
            feature_dim=2560,
            quality_warnings=[],
            roi_images={},
            is_available=False,
            status_message="Please upload both fingernail and skin images to perform multimodal estimation."
        )

    warnings = []
    warnings.extend(validate_image_quality(nail_image, "Fingernail"))
    warnings.extend(validate_image_quality(skin_image, "Skin"))

    nail_roi, _ = extract_roi(nail_image, "nail")
    skin_roi, _ = extract_roi(skin_image, "skin")

    try:
        model = get_cached_model("model4_nail_skin_fusion.pkl")
    except FileNotFoundError:
        return InferenceResult(
            predicted_hb=None,
            classification="Model Unavailable",
            chip_class="chip-dark",
            clinical_note="Model 4 weights not found in saved_models directory.",
            model_name="Model 4: Nail + Skin Fusion",
            modality="Nail + Skin",
            feature_dim=2560,
            quality_warnings=warnings,
            roi_images={"nail": nail_roi, "skin": skin_roi},
            is_available=False,
            status_message="Dual-modality checkpoint missing."
        )

    extractor = get_cnn_extractor()
    n_feat = extractor.extract_single(nail_roi).reshape(1, -1)
    s_feat = extractor.extract_single(skin_roi).reshape(1, -1)
    fused_feat = np.concatenate([n_feat, s_feat], axis=-1)  # (1, 2560)

    pred_val = float(model.predict(fused_feat)[0])
    label, chip_cls, note = classify_hb(pred_val)

    return InferenceResult(
        predicted_hb=round(pred_val, 2),
        classification=label,
        chip_class=chip_cls,
        clinical_note=note,
        model_name="Model 4: Multimodal Fingernail + Skin Fusion",
        modality="Nail + Skin",
        feature_dim=2560,
        quality_warnings=warnings,
        roi_images={"nail": nail_roi, "skin": skin_roi},
        is_available=True,
        status_message="Multimodal inference computed from concatenated 2560-d nail and skin embeddings."
    )
