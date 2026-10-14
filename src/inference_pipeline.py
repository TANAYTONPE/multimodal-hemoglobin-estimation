"""
inference_pipeline.py
---------------------
Reusable inference engine for single-modal and multimodal hemoglobin estimation.
Connects ROI localization, pretrained EfficientNet-B0 + color feature extraction,
and trained regression heads with WHO clinical classification.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import joblib
import torch

from roi_extractor import auto_detect_roi, draw_roi_overlay, crop_roi_by_box
from feature_extractor import get_feature_extractor, extract_color_features
from models_multimodal import TriModalHemoglobinNet

CHECKPOINTS_DIR = "checkpoints"


def classify_who_anemia(hb_val: float) -> Tuple[str, str, str]:
    """
    WHO diagnostic hemoglobin classification.
    Returns (label, chip_class, description).
    """
    if hb_val >= 12.0:
        return "Normal", "chip-green", "Hemoglobin concentration within normal diagnostic range (≥ 12.0 g/dL)."
    elif hb_val >= 10.0:
        return "Mild Anemia", "chip-yellow", "Mild anemia detected (10.0 – 11.9 g/dL). Recommend dietary and medical review."
    elif hb_val >= 7.0:
        return "Moderate Anemia", "chip-yellow", "Moderate anemia detected (7.0 – 9.9 g/dL). Clinical evaluation recommended."
    else:
        return "Severe Anemia", "chip-red", "Severe anemia alert (< 7.0 g/dL). Immediate medical attention required."


class HemoglobinInferenceEngine:
    """
    Unified inference manager loading and caching model artifacts.
    """
    def __init__(self, ckpt_dir: str = CHECKPOINTS_DIR):
        self.ckpt_dir = ckpt_dir
        self.extractor = get_feature_extractor()
        self._models: Dict[str, Any] = {}
        self._load_available_models()

    def _load_available_models(self):
        m1_path = os.path.join(self.ckpt_dir, "model1_fingernail.joblib")
        m2_path = os.path.join(self.ckpt_dir, "model2_skin.joblib")
        m3_path = os.path.join(self.ckpt_dir, "model3_conjunctiva.joblib")
        m4_path = os.path.join(self.ckpt_dir, "model4_nail_skin_fusion.joblib")
        m6_path = os.path.join(self.ckpt_dir, "model6_tri_modal.pth")

        if os.path.isfile(m1_path):
            self._models["model1_fingernail"] = joblib.load(m1_path)
        if os.path.isfile(m2_path):
            self._models["model2_skin"] = joblib.load(m2_path)
        if os.path.isfile(m3_path):
            self._models["model3_conjunctiva"] = joblib.load(m3_path)
        if os.path.isfile(m4_path):
            self._models["model4_nail_skin_fusion"] = joblib.load(m4_path)
        if os.path.isfile(m6_path):
            net6 = TriModalHemoglobinNet(embed_dim=1296, fusion_dim=256)
            ckpt = torch.load(m6_path, map_location="cpu", weights_only=False)
            net6.load_state_dict(ckpt["state_dict"])
            net6.eval()
            self._models["model6_tri_modal"] = net6

    def predict_single(
        self,
        image: Image.Image,
        modality: str = "fingernail"
    ) -> Dict[str, Any]:
        """
        Run end-to-end inference on a single uploaded image.
        """
        modality_clean = modality.lower().strip()
        roi_crop, roi_box = auto_detect_roi(image, modality=modality_clean)
        overlay = draw_roi_overlay(image, [roi_box], [f"{modality.capitalize()} ROI"])

        # Extract features (1280 deep + 16 color = 1296)
        features = self.extractor.extract_single_patch_features(roi_crop)
        color_vec = features[1280:]

        color_stats = {
            "mean_r": float(color_vec[0]),
            "mean_g": float(color_vec[1]),
            "mean_b": float(color_vec[2]),
            "norm_r": float(color_vec[6]),
            "norm_g": float(color_vec[7]),
            "norm_b": float(color_vec[8]),
            "rg_ratio": float(color_vec[9]),
            "erythema_index": float(color_vec[10]),
            "a_star": float(color_vec[14]),
            "L_star": float(color_vec[15]),
        }

        # Select model based on modality
        if "nail" in modality_clean or "finger" in modality_clean:
            model_key = "model1_fingernail"
            model_name = "Model 1: Fingernail CNN + Color Regressor"
            status = "Trained & Clinically Evaluated (MAE = 1.58 g/dL, R² = 0.36)"
            is_pending = False
        elif "skin" in modality_clean:
            model_key = "model2_skin"
            model_name = "Model 2: Skin CNN + Color Regressor"
            status = "Trained & Clinically Evaluated (MAE = 1.80 g/dL, R² = 0.06)"
            is_pending = False
        elif "conjunctiv" in modality_clean or "eye" in modality_clean:
            model_key = "model3_conjunctiva"
            model_name = "Model 3: Conjunctiva CNN + Color Regressor"
            status = "Trained & Clinically Evaluated (MAE = 1.39 g/dL, R² = 0.49)"
            is_pending = False
        else:
            model_key = "model1_fingernail"
            model_name = "Default Regressor (Nail)"
            status = "Fallback Regressor"
            is_pending = False

        model = self._models.get(model_key)
        if model is None:
            raise RuntimeError(f"Checkpoint for {model_key} not found. Please run train_evaluate_all.py first.")

        pred_val = float(model.predict(features.reshape(1, -1))[0])
        pred_val = round(pred_val, 2)
        who_label, chip_cls, who_desc = classify_who_anemia(pred_val)

        return {
            "predicted_hb": pred_val,
            "who_label": who_label,
            "chip_class": chip_cls,
            "who_description": who_desc,
            "roi_crop": roi_crop,
            "roi_overlay": overlay,
            "roi_box": roi_box,
            "color_stats": color_stats,
            "model_name": model_name,
            "status": status,
            "is_pending": is_pending,
            "modality": modality,
        }

    def predict_multimodal_fusion(
        self,
        nail_image: Image.Image,
        skin_image: Image.Image
    ) -> Dict[str, Any]:
        """
        Model 4: Paired Fingernail + Skin Feature Fusion inference.
        """
        # Nail ROI & Features
        nail_crop, nail_box = auto_detect_roi(nail_image, modality="fingernail")
        nail_overlay = draw_roi_overlay(nail_image, [nail_box], ["Fingernail Bed"])
        f_nail = self.extractor.extract_single_patch_features(nail_crop)

        # Skin ROI & Features
        skin_crop, skin_box = auto_detect_roi(skin_image, modality="skin")
        skin_overlay = draw_roi_overlay(skin_image, [skin_box], ["Finger Skin Tissue"])
        f_skin = self.extractor.extract_single_patch_features(skin_crop)

        # Fused 2592-dim vector
        f_fused = np.concatenate([f_nail, f_skin], axis=0).reshape(1, -1)

        model_fusion = self._models.get("model4_nail_skin_fusion")
        if model_fusion is None:
            raise RuntimeError("Model 4 fusion checkpoint not loaded. Run train_evaluate_all.py.")

        pred_fusion = float(model_fusion.predict(f_fused)[0])
        pred_fusion = round(pred_fusion, 2)
        who_label, chip_cls, who_desc = classify_who_anemia(pred_fusion)

        # Also compute single modality predictions for direct comparison
        m1 = self._models.get("model1_fingernail")
        m2 = self._models.get("model2_skin")
        pred_m1 = round(float(m1.predict(f_nail.reshape(1, -1))[0]), 2) if m1 else None
        pred_m2 = round(float(m2.predict(f_skin.reshape(1, -1))[0]), 2) if m2 else None

        return {
            "predicted_hb": pred_fusion,
            "who_label": who_label,
            "chip_class": chip_cls,
            "who_description": who_desc,
            "pred_nail_alone": pred_m1,
            "pred_skin_alone": pred_m2,
            "nail_crop": nail_crop,
            "nail_overlay": nail_overlay,
            "skin_crop": skin_crop,
            "skin_overlay": skin_overlay,
            "model_name": "Model 4: Fingernail + Skin Feature Fusion",
            "status": "Trained & Clinically Evaluated (MAE = 1.62 g/dL, R² = 0.29)",
        }

    def run_trimodal_forward_pass(
        self,
        nail_image: Image.Image,
        skin_image: Image.Image,
        conj_image: Image.Image
    ) -> Dict[str, Any]:
        """
        Model 6: Three-Branch Multimodal Forward Pass.
        """
        nail_crop, _ = auto_detect_roi(nail_image, modality="fingernail")
        skin_crop, _ = auto_detect_roi(skin_image, modality="skin")
        conj_crop, _ = auto_detect_roi(conj_image, modality="conjunctiva")

        f_nail = torch.from_numpy(self.extractor.extract_single_patch_features(nail_crop)).unsqueeze(0)
        f_skin = torch.from_numpy(self.extractor.extract_single_patch_features(skin_crop)).unsqueeze(0)
        f_conj = torch.from_numpy(self.extractor.extract_single_patch_features(conj_crop)).unsqueeze(0)

        net6 = self._models.get("model6_tri_modal")
        if net6 is None:
            raise RuntimeError("Model 6 architecture not found.")

        with torch.no_grad():
            pred_tensor = net6(f_nail, f_skin, f_conj)
            pred_hb = float(pred_tensor.squeeze().item())

        return {
            "branch1_dim": f_nail.shape[-1],
            "branch2_dim": f_skin.shape[-1],
            "branch3_dim": f_conj.shape[-1],
            "fused_dim": f_nail.shape[-1] * 3,
            "forward_output_hb": round(pred_hb, 2),
            "status": "Architecture Verified (Awaiting Tri-Modal Paired Dataset with Lab Hb)",
        }


# Singleton engine instance
_INFERENCE_ENGINE: Optional[HemoglobinInferenceEngine] = None

def get_inference_engine() -> HemoglobinInferenceEngine:
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        _INFERENCE_ENGINE = HemoglobinInferenceEngine()
    return _INFERENCE_ENGINE
