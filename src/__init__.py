"""
Multimodal Non-Invasive Hemoglobin Estimation (HemoLens Multimodal)
------------------------------------------------------------------
Deep Learning and Optical Colorimetry for Hemoglobin Estimation from
Fingernail Bed, Finger Skin, and Palpebral Conjunctiva Images.
"""

import os
import sys

# Ensure src is on sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

__version__ = "2.0.0"
__author__ = "Tanay Tonpe"

from .preprocessing import (
    get_train_transforms,
    get_val_transforms,
    HemoglobinDataset,
    apply_gray_world_white_balance,
)
from .roi_extractor import (
    auto_detect_roi,
    draw_roi_overlay,
    crop_roi_by_box,
    assess_image_quality,
    detect_specular_glare_mask,
    detect_hand_nail_roi,
    detect_eye_conjunctiva_roi,
)
from .feature_extractor import (
    get_feature_extractor,
    extract_color_features,
    generate_erythema_heatmap,
)
from .models import (
    FingernailModel,
    SkinModel,
    ConjunctivalModel,
    NailSkinFusionModel,
    MainProposedMultimodalModel,
)
from .models_multimodal import TriModalHemoglobinNet, SklearnHbRegressor, evaluate_predictions
from .inference_pipeline import (
    get_inference_engine,
    HemoglobinInferenceEngine,
    MultimodalInferenceEngine,
    classify_who_anemia,
)

__all__ = [
    "get_train_transforms",
    "get_val_transforms",
    "HemoglobinDataset",
    "apply_gray_world_white_balance",
    "auto_detect_roi",
    "draw_roi_overlay",
    "crop_roi_by_box",
    "assess_image_quality",
    "detect_specular_glare_mask",
    "detect_hand_nail_roi",
    "detect_eye_conjunctiva_roi",
    "get_feature_extractor",
    "extract_color_features",
    "generate_erythema_heatmap",
    "FingernailModel",
    "SkinModel",
    "ConjunctivalModel",
    "NailSkinFusionModel",
    "MainProposedMultimodalModel",
    "TriModalHemoglobinNet",
    "SklearnHbRegressor",
    "evaluate_predictions",
    "get_inference_engine",
    "HemoglobinInferenceEngine",
    "MultimodalInferenceEngine",
    "classify_who_anemia",
]
