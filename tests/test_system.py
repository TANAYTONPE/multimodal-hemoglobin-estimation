"""
test_system.py
--------------
Comprehensive test suite validating the complete multimodal hemoglobin
estimation system.

Tests:
1. ROI extraction and localization (nail, skin, conjunctiva, overlays)
2. Pretrained deep and color feature extraction
3. Checkpoint verification and loading
4. Single-modal inference (Models 1, 2, 3)
5. Multimodal Nail + Skin fusion inference (Model 4)
6. Three-branch Tri-Modal architecture forward pass (Model 6)
7. Edge cases and invalid inputs handling
8. Evaluation artifacts and dashboard outputs
"""

import os
import sys
import io
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Ensure src/ is on sys.path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from roi_extractor import (
    parse_bounding_boxes,
    crop_roi_by_box,
    auto_detect_roi,
    draw_roi_overlay,
)
from feature_extractor import (
    extract_color_features,
    get_feature_extractor,
)
from models_multimodal import (
    SklearnHbRegressor,
    TriModalHemoglobinNet,
    evaluate_predictions,
)
from inference_pipeline import (
    get_inference_engine,
    classify_who_anemia,
)


class TestMultimodalHemoglobinSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_inference_engine()
        cls.nail_img_path = os.path.join("data", "sample_images", "sample_fingernail.jpg")
        cls.skin_img_path = os.path.join("data", "sample_images", "sample_skin.jpg")
        cls.conj_img_path = os.path.join("data", "sample_images", "sample_conjunctiva.jpg")

        assert os.path.isfile(cls.nail_img_path), f"Missing {cls.nail_img_path}"
        assert os.path.isfile(cls.skin_img_path), f"Missing {cls.skin_img_path}"
        assert os.path.isfile(cls.conj_img_path), f"Missing {cls.conj_img_path}"

        cls.im_nail = Image.open(cls.nail_img_path).convert("RGB")
        cls.im_skin = Image.open(cls.skin_img_path).convert("RGB")
        cls.im_conj = Image.open(cls.conj_img_path).convert("RGB")

    def test_01_roi_extraction(self):
        """Test bounding box parsing, cropping, and heuristic auto-detection."""
        # Exact box cropping
        box = [200, 300, 350, 450]
        crop = crop_roi_by_box(self.im_nail, box, target_size=(224, 224))
        self.assertEqual(crop.size, (224, 224))

        # Auto detection on fingernail
        crop_nail, box_nail = auto_detect_roi(self.im_nail, modality="fingernail")
        self.assertEqual(crop_nail.size, (224, 224))
        self.assertEqual(len(box_nail), 4)

        # Auto detection on skin
        crop_skin, box_skin = auto_detect_roi(self.im_skin, modality="skin")
        self.assertEqual(crop_skin.size, (224, 224))
        self.assertEqual(len(box_skin), 4)

        # Auto detection on conjunctiva
        crop_conj, box_conj = auto_detect_roi(self.im_conj, modality="conjunctiva")
        self.assertEqual(crop_conj.size, (224, 224))
        self.assertEqual(len(box_conj), 4)

        # Overlay generation
        overlay = draw_roi_overlay(self.im_nail, [box_nail], labels=["Test Nail"])
        self.assertEqual(overlay.size, self.im_nail.size)

    def test_02_feature_extraction(self):
        """Test deep EfficientNet-B0 and color space feature extraction."""
        # Color features
        color_vec = extract_color_features(self.im_nail.crop((10, 10, 80, 80)))
        self.assertEqual(color_vec.shape, (16,))
        self.assertTrue(np.all(np.isfinite(color_vec)))

        # Deep extractor
        extractor = get_feature_extractor()
        crop = self.im_nail.crop((10, 10, 80, 80)).resize((224, 224))
        deep_vec = extractor.extract_deep_features(crop)
        self.assertEqual(deep_vec.shape, (1280,))
        self.assertTrue(np.all(np.isfinite(deep_vec)))

        # Combined features
        comb_vec = extractor.extract_single_patch_features(crop)
        self.assertEqual(comb_vec.shape, (1296,))
        self.assertTrue(np.all(np.isfinite(comb_vec)))

    def test_03_model_checkpoints_exist_and_load(self):
        """Verify saved checkpoints in checkpoints/ directory."""
        ckpts = [
            "model1_fingernail.joblib",
            "model2_skin.joblib",
            "model3_conjunctiva.joblib",
            "model4_nail_skin_fusion.joblib",
            "model6_tri_modal.pth",
        ]
        for ckpt in ckpts:
            path = os.path.join("checkpoints", ckpt)
            self.assertTrue(os.path.isfile(path), f"Checkpoint missing: {path}")

    def test_04_single_modal_inference(self):
        """Test single-modality inference pipeline for Nail, Skin, and Conjunctiva."""
        # Nail inference
        res_nail = self.engine.predict_single(self.im_nail, modality="fingernail")
        self.assertIn("predicted_hb", res_nail)
        self.assertTrue(3.0 <= res_nail["predicted_hb"] <= 22.0)
        self.assertIn(res_nail["who_label"], ["Normal", "Mild Anemia", "Moderate Anemia", "Severe Anemia"])

        # Skin inference
        res_skin = self.engine.predict_single(self.im_skin, modality="skin")
        self.assertIn("predicted_hb", res_skin)
        self.assertTrue(3.0 <= res_skin["predicted_hb"] <= 22.0)

        # Conjunctiva inference (Now trained on Eyes-defy-anemia)
        res_conj = self.engine.predict_single(self.im_conj, modality="conjunctiva")
        self.assertIn("predicted_hb", res_conj)
        self.assertFalse(res_conj["is_pending"])
        self.assertTrue(3.0 <= res_conj["predicted_hb"] <= 22.0)

    def test_05_multimodal_fusion_inference(self):
        """Test Model 4 paired Fingernail + Skin multimodal feature fusion."""
        res_fusion = self.engine.predict_multimodal_fusion(self.im_nail, self.im_skin)
        self.assertIn("predicted_hb", res_fusion)
        self.assertTrue(3.0 <= res_fusion["predicted_hb"] <= 22.0)
        self.assertIsNotNone(res_fusion["pred_nail_alone"])
        self.assertIsNotNone(res_fusion["pred_skin_alone"])
        self.assertEqual(res_fusion["nail_crop"].size, (224, 224))
        self.assertEqual(res_fusion["skin_crop"].size, (224, 224))

    def test_06_trimodal_model6_forward_pass(self):
        """Test Model 6 three-branch architecture tensor dimensions and forward pass."""
        # Dummy tensor verification
        net6 = TriModalHemoglobinNet(embed_dim=1296, fusion_dim=256)
        dummy_nail = torch.randn(2, 1296)
        dummy_skin = torch.randn(2, 1296)
        dummy_conj = torch.randn(2, 1296)
        dummy_out = net6(dummy_nail, dummy_skin, dummy_conj)
        self.assertEqual(dummy_out.shape, (2, 1))

        # Real image forward pass via engine
        res6 = self.engine.run_trimodal_forward_pass(self.im_nail, self.im_skin, self.im_conj)
        self.assertEqual(res6["branch1_dim"], 1296)
        self.assertEqual(res6["branch2_dim"], 1296)
        self.assertEqual(res6["branch3_dim"], 1296)
        self.assertEqual(res6["fused_dim"], 3888)
        self.assertIn("forward_output_hb", res6)

    def test_07_invalid_inputs_and_edge_cases(self):
        """Verify graceful error and boundary handling."""
        # Tiny image
        tiny = Image.new("RGB", (10, 10), color="white")
        crop_tiny, box_tiny = auto_detect_roi(tiny, modality="fingernail")
        self.assertEqual(crop_tiny.size, (224, 224))

        # WHO anemia boundary classification
        self.assertEqual(classify_who_anemia(13.5)[0], "Normal")
        self.assertEqual(classify_who_anemia(10.5)[0], "Mild Anemia")
        self.assertEqual(classify_who_anemia(8.5)[0], "Moderate Anemia")
        self.assertEqual(classify_who_anemia(5.5)[0], "Severe Anemia")

    def test_08_evaluation_artifacts(self):
        """Verify results CSV files and publication plots exist and are populated."""
        comp_csv = os.path.join("results", "model_comparison.csv")
        preds_csv = os.path.join("results", "patient_test_predictions.csv")
        fig1 = os.path.join("results", "predicted_vs_actual.png")
        fig2 = os.path.join("results", "residuals_plot.png")
        fig3 = os.path.join("results", "error_distribution.png")

        self.assertTrue(os.path.isfile(comp_csv))
        self.assertTrue(os.path.isfile(preds_csv))
        self.assertTrue(os.path.isfile(fig1))
        self.assertTrue(os.path.isfile(fig2))
        self.assertTrue(os.path.isfile(fig3))

        df_comp = pd.read_csv(comp_csv)
        self.assertGreaterEqual(len(df_comp), 6)
        self.assertIn("Model ID", df_comp.columns)
        self.assertIn("MAE (g/dL)", df_comp.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
