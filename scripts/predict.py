"""
scripts/predict.py
------------------
Command-line interface (CLI) for non-invasive hemoglobin estimation
using HemoLens Multimodal.

Usage Examples
--------------
# Single-modality inference:
python scripts/predict.py --image data/sample_images/sample_fingernail.jpg --modality fingernail
python scripts/predict.py --image data/sample_images/sample_conjunctiva.jpg --modality conjunctiva

# Multimodal consensus inference (2 or 3 modalities):
python scripts/predict.py --nail data/sample_images/sample_fingernail.jpg \
                          --skin data/sample_images/sample_skin.jpg \
                          --conjunctiva data/sample_images/sample_conjunctiva.jpg

# Save annotated overlay & ROI crop:
python scripts/predict.py --image data/sample_images/sample_conjunctiva.jpg \
                          --modality conjunctiva \
                          --save-overlay results/sample_eye_overlay.jpg \
                          --save-crop results/sample_eye_crop.jpg
"""

import os
import sys
import argparse
import json
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure src/ is on sys.path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_pipeline import get_inference_engine


def main():
    parser = argparse.ArgumentParser(
        description="HemoLens Multimodal: Non-Invasive Hemoglobin Estimation CLI"
    )

    # Single-modality options
    parser.add_argument("--image", type=str, default=None, help="Path to clinical photograph for single-modality inference.")
    parser.add_argument("--modality", type=str, default="fingernail",
                        choices=["fingernail", "skin", "conjunctiva"],
                        help="Anatomical tissue site of the input image.")

    # Multimodal options
    parser.add_argument("--nail", type=str, default=None, help="Path to fingernail image for multimodal consensus.")
    parser.add_argument("--skin", type=str, default=None, help="Path to skin tissue image for multimodal consensus.")
    parser.add_argument("--conjunctiva", type=str, default=None, help="Path to conjunctiva image for multimodal consensus.")

    # Additional options
    parser.add_argument("--white-balance", action="store_true", help="Apply Gray-World chromatic constancy calibration.")
    parser.add_argument("--save-overlay", type=str, default=None, help="Output path to save annotated ROI overlay image.")
    parser.add_argument("--save-crop", type=str, default=None, help="Output path to save 224x224 cropped tissue ROI.")
    parser.add_argument("--save-cam", type=str, default=None, help="Output path to save deep CAM visual saliency heatmap.")
    parser.add_argument("--save-erythema", type=str, default=None, help="Output path to save microvascular erythema index heatmap.")
    parser.add_argument("--json", action="store_true", help="Output result as JSON string.")

    args = parser.parse_args()

    engine = get_inference_engine()

    # Determine if running single-modality or multimodal consensus
    is_multimodal = (args.nail is not None) or (args.skin is not None) or (args.conjunctiva is not None)

    if not is_multimodal and args.image is None:
        parser.print_help()
        sys.exit(1)

    if is_multimodal:
        im_nail = Image.open(args.nail).convert("RGB") if args.nail else None
        im_skin = Image.open(args.skin).convert("RGB") if args.skin else None
        im_conj = Image.open(args.conjunctiva).convert("RGB") if args.conjunctiva else None

        result = engine.predict_multimodal_consensus(
            nail_image=im_nail,
            skin_image=im_skin,
            conj_image=im_conj,
            apply_white_balance=args.white_balance,
        )

        if args.json:
            clean_res = {
                "predicted_hb_g_dL": result["predicted_hb"],
                "who_classification": result["who_label"],
                "model_name": result["model_name"],
                "individual_predictions": result["individual_predictions"],
                "normalized_weights": result["normalized_weights"],
                "status": result["status"]
            }
            print(json.dumps(clean_res, indent=2))
        else:
            print("\n" + "=" * 60)
            print("  HEMOLENS MULTIMODAL CONSENSUS ESTIMATION")
            print("=" * 60)
            print(f"  Model Engine         : {result['model_name']}")
            print(f"  Estimated Hb         : {result['predicted_hb']} g/dL")
            print(f"  WHO Diagnosis        : {result['who_label']}")
            print(f"  Clinical Assessment  : {result['who_description']}")
            print("  --- Individual Modality Breakdown ---")
            for mod, val in result["individual_predictions"].items():
                w = result["normalized_weights"].get(mod, 0.0)
                print(f"    - {mod.capitalize():12}: {val:.2f} g/dL (Weight: {w*100:.1f}%)")
            if result.get("paired_m4_pred"):
                print(f"    - Model 4 Fusion: {result['paired_m4_pred']:.2f} g/dL")
            print("=" * 60 + "\n")

    else:
        if not os.path.isfile(args.image):
            print(f"Error: Image not found at {args.image}", file=sys.stderr)
            sys.exit(1)

        img = Image.open(args.image).convert("RGB")
        result = engine.predict_single(
            img,
            modality=args.modality,
            apply_white_balance=args.white_balance
        )

        if args.save_overlay:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_overlay)), exist_ok=True)
            result["roi_overlay"].save(args.save_overlay)
            print(f"[Saved] ROI Overlay -> {args.save_overlay}")

        if args.save_crop:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_crop)), exist_ok=True)
            result["roi_crop"].save(args.save_crop)
            print(f"[Saved] Tissue Crop -> {args.save_crop}")

        if args.save_cam and "cam_heatmap" in result:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_cam)), exist_ok=True)
            result["cam_heatmap"].save(args.save_cam)
            print(f"[Saved] CAM Saliency Heatmap -> {args.save_cam}")

        if args.save_erythema and "erythema_heatmap" in result:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_erythema)), exist_ok=True)
            result["erythema_heatmap"].save(args.save_erythema)
            print(f"[Saved] Erythema Perfusion Heatmap -> {args.save_erythema}")

        if args.json:
            clean_res = {
                "modality": args.modality,
                "predicted_hb_g_dL": result["predicted_hb"],
                "who_classification": result["who_label"],
                "model_name": result["model_name"],
                "color_biomarkers": result["color_stats"],
                "roi_bounding_box": result["roi_box"],
                "status": result["status"]
            }
            print(json.dumps(clean_res, indent=2))
        else:
            print("\n" + "=" * 60)
            print("  HEMOLENS CLINICAL ESTIMATION OUTPUT")
            print("=" * 60)
            print(f"  Modality             : {args.modality.capitalize()}")
            print(f"  Model Architecture   : {result['model_name']}")
            print(f"  Estimated Hb         : {result['predicted_hb']} g/dL")
            print(f"  WHO Diagnosis        : {result['who_label']}")
            print(f"  Clinical Assessment  : {result['who_description']}")
            print("  --- Extracted Optical Biomarkers ---")
            c = result["color_stats"]
            print(f"    - Norm Red (R)     : {c['norm_r']:.3f}")
            print(f"    - Norm Green (G)   : {c['norm_g']:.3f}")
            print(f"    - R / G Ratio      : {c['rg_ratio']:.2f}")
            print(f"    - Erythema Index   : {c['erythema_index']:.3f}")
            print(f"    - CIELAB a* (Red)  : {c['a_star']:.2f}")
            print(f"    - CIELAB L* (Lum)  : {c['L_star']:.2f}")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
