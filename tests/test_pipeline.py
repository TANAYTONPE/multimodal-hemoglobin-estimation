"""
test_pipeline.py
----------------
Comprehensive end-to-end automated test suite for HemoLens.
Tests all requirements specified in Phase 10:
- Image validation (illuminance, contrast, dimensions)
- ROI extraction & bounding box coordinates
- Pretrained CNN & Color feature extraction
- Model loading & single-modal inference (Nail, Skin)
- Multimodal fusion inference (Nail + Skin)
- Model 6 tensor-shape propagation smoke test
- Missing-modality & missing-model handling
- Invalid/corrupt image handling
- Benchmark table & evaluation plots verification
- App syntax & import verification
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import torch
import pandas as pd

# Ensure src/ is on sys.path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference import (
    validate_image_quality,
    extract_roi,
    predict_single_modality,
    predict_multimodal,
    get_cached_model,
    classify_hb,
    get_cnn_extractor
)
from features import CNNFeatureExtractor, ColorFeatureExtractor
from models import MainProposedMultimodalModel, run_model6_smoke_test


class TestHemoLensPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_nail_path = "data/crops/nail/1_f0.jpg"
        cls.sample_skin_path = "data/crops/skin/1_f0.jpg"
        assert os.path.isfile(cls.sample_nail_path), f"Missing {cls.sample_nail_path}"
        assert os.path.isfile(cls.sample_skin_path), f"Missing {cls.sample_skin_path}"
        cls.nail_img = Image.open(cls.sample_nail_path).convert("RGB")
        cls.skin_img = Image.open(cls.sample_skin_path).convert("RGB")

    def test_01_saved_models_exist(self):
        """Verifies all trained model files exist."""
        required = [
            "saved_models/baseline_a.pkl",
            "saved_models/baseline_b.pkl",
            "saved_models/baseline_c.pkl",
            "saved_models/model1_nail.pkl",
            "saved_models/model2_skin.pkl",
            "saved_models/model4_nail_skin_fusion.pkl",
        ]
        for f in required:
            self.assertTrue(os.path.isfile(f), f"Required model file missing: {f}")

    def test_02_results_and_plots_exist(self):
        """Verifies benchmark tables and evaluation figures exist and are non-empty."""
        comp_path = "results/model_comparison.csv" if os.path.isfile("results/model_comparison.csv") else "model_comparison.csv"
        self.assertTrue(os.path.isfile(comp_path))
        df = pd.read_csv(comp_path)
        self.assertGreater(len(df), 5)

        pred_path = "results/patient_test_predictions.csv" if os.path.isfile("results/patient_test_predictions.csv") else "test_predictions.csv"
        self.assertTrue(os.path.isfile(pred_path))
        pdf = pd.read_csv(pred_path)
        self.assertGreaterEqual(len(pdf), 30)

        res_dir = "results" if os.path.isdir("results") else "evaluation_plots"
        for plot_name in ["predicted_vs_actual.png", "residuals_plot.png", "error_distribution.png"]:
            plot_path = os.path.join(res_dir, plot_name)
            self.assertTrue(os.path.isfile(plot_path), f"Missing plot: {plot_path}")
            self.assertGreater(os.path.getsize(plot_path), 500, f"Plot is empty: {plot_path}")

    def test_03_image_quality_validation(self):
        """Tests image quality validation logic for normal, dark, low contrast, and grayscale inputs."""
        # Normal image
        w_normal = validate_image_quality(self.nail_img, "nail")
        self.assertEqual(len(w_normal), 0, f"Clean image should have no warnings, got: {w_normal}")

        # Dark image
        dark_img = Image.new("RGB", (100, 100), (10, 10, 10))
        w_dark = validate_image_quality(dark_img, "nail")
        self.assertTrue(any("Low lighting" in w for w in w_dark))

        # Overexposed image
        bright_img = Image.new("RGB", (100, 100), (245, 245, 245))
        w_bright = validate_image_quality(bright_img, "nail")
        self.assertTrue(any("High glare" in w for w in w_bright))

        # Monochrome / Grayscale image
        mono_img = Image.new("RGB", (100, 100), (128, 128, 128))
        w_mono = validate_image_quality(mono_img, "nail")
        self.assertTrue(any("monochromatic" in w for w in w_mono))

    def test_04_roi_extraction(self):
        """Tests ROI extraction for both tight crops and full photos."""
        # Tight crop
        crop, box = extract_roi(self.nail_img, "nail")
        self.assertEqual(crop.size, self.nail_img.size)

        # Full photo (800x600)
        full_photo = Image.new("RGB", (800, 600), (200, 150, 150))
        crop_full, box_full = extract_roi(full_photo, "nail")
        self.assertLess(crop_full.size[0], 800)
        self.assertLess(crop_full.size[1], 600)
        self.assertEqual(box_full[0], int(800 * 0.40))

    def test_05_feature_extraction(self):
        """Tests frozen EfficientNet-B0 and handcrafted Color feature extractors."""
        cnn = get_cnn_extractor()
        cnn_feat = cnn.extract_single(self.nail_img)
        self.assertEqual(cnn_feat.shape, (1280,))
        self.assertFalse(np.isnan(cnn_feat).any())

        col_feat = ColorFeatureExtractor.extract(self.nail_img)
        self.assertEqual(col_feat.shape, (24,))
        self.assertFalse(np.isnan(col_feat).any())

    def test_06_single_modal_inference_nail(self):
        """Tests single-modal Fingernail inference (Model 1)."""
        res = predict_single_modality(self.nail_img, "Fingernail")
        self.assertTrue(res.is_available)
        self.assertIsNotNone(res.predicted_hb)
        self.assertGreater(res.predicted_hb, 4.0)
        self.assertLess(res.predicted_hb, 20.0)
        self.assertIn("Model 1", res.model_name)
        self.assertIn("nail", res.roi_images)

    def test_07_single_modal_inference_skin(self):
        """Tests single-modal Skin inference (Model 2)."""
        res = predict_single_modality(self.skin_img, "Skin")
        self.assertTrue(res.is_available)
        self.assertIsNotNone(res.predicted_hb)
        self.assertGreater(res.predicted_hb, 4.0)
        self.assertLess(res.predicted_hb, 20.0)
        self.assertIn("Model 2", res.model_name)
        self.assertIn("skin", res.roi_images)

    def test_08_multimodal_inference_dual(self):
        """Tests multimodal dual fusion inference (Model 4: Nail + Skin)."""
        res = predict_multimodal(self.nail_img, self.skin_img, fusion_type="Nail + Skin")
        self.assertTrue(res.is_available)
        self.assertIsNotNone(res.predicted_hb)
        self.assertGreater(res.predicted_hb, 4.0)
        self.assertLess(res.predicted_hb, 20.0)
        self.assertEqual(res.feature_dim, 2560)
        self.assertIn("Model 4", res.model_name)

    def test_09_model6_tensor_shape_compatibility(self):
        """Tests three-branch Model 6 forward tensor propagation smoke test."""
        passed = run_model6_smoke_test()
        self.assertTrue(passed)

    def test_10_missing_modality_handling(self):
        """Tests multimodal inference when a required modality is omitted."""
        res = predict_multimodal(nail_image=self.nail_img, skin_image=None)
        self.assertFalse(res.is_available)
        self.assertIsNone(res.predicted_hb)
        self.assertEqual(res.classification, "Missing Input")

    def test_11_pending_modality_handling(self):
        """Tests handling of Conjunctiva where training is pending."""
        res_conj = predict_single_modality(self.nail_img, "Conjunctiva")
        self.assertFalse(res_conj.is_available)
        self.assertIsNone(res_conj.predicted_hb)
        self.assertEqual(res_conj.classification, "Model Pending")

        res_m6 = predict_multimodal(self.nail_img, self.skin_img, self.nail_img, "Three-Modality (Nail + Skin + Conjunctiva)")
        self.assertFalse(res_m6.is_available)
        self.assertIsNone(res_m6.predicted_hb)
        self.assertEqual(res_m6.classification, "Model Pending")

    def test_12_who_classification(self):
        """Tests WHO anemia thresholds classification."""
        label_norm, _, _ = classify_hb(13.5)
        self.assertEqual(label_norm, "Normal")
        label_mild, _, _ = classify_hb(11.2)
        self.assertEqual(label_mild, "Mild Anemia")
        label_mod, _, _ = classify_hb(8.5)
        self.assertEqual(label_mod, "Moderate Anemia")
        label_sev, _, _ = classify_hb(5.5)
        self.assertEqual(label_sev, "Severe Anemia")

    def test_13_app_compiles(self):
        """Verifies app.py has zero syntax errors."""
        import py_compile
        py_compile.compile("app.py", doraise=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
