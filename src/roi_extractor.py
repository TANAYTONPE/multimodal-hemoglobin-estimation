"""
roi_extractor.py
----------------
Region of Interest (ROI) extraction and visualization for non-invasive
hemoglobin estimation from fingernail, skin, and conjunctival images.

Supports:
1. Ground-truth bounding box cropping from dataset metadata ([ymin, xmin, ymax, xmax]).
2. Saliency- and color-guided automated ROI localization for arbitrary uploads.
3. High-contrast visual overlay for UI display and quality inspection.
"""

import ast
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_bounding_boxes(boxes_str: str) -> List[List[int]]:
    """Parse bounding boxes string from metadata.csv (format: [[ymin, xmin, ymax, xmax], ...])."""
    if not boxes_str or str(boxes_str).strip() == "" or str(boxes_str).lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(boxes_str)
        if isinstance(parsed, list):
            return [[int(c) for c in box] for box in parsed if len(box) == 4]
    except Exception:
        pass
    return []


def crop_roi_by_box(image: Image.Image, box: List[int], target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Crop an ROI from an image given [ymin, xmin, ymax, xmax] coordinates.
    Resizes the cropped patch to target_size (default 224x224).
    """
    w, h = image.size
    ymin, xmin, ymax, xmax = box

    # Clamp coordinates to image boundaries
    xmin = max(0, min(xmin, w - 1))
    xmax = max(xmin + 1, min(xmax, w))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(ymin + 1, min(ymax, h))

    # PIL crop expects (left, top, right, bottom) = (xmin, ymin, xmax, ymax)
    crop = image.crop((xmin, ymin, xmax, ymax))
    if target_size:
        crop = crop.resize(target_size, Image.Resampling.BILINEAR)
    return crop


def auto_detect_roi(
    image: Image.Image,
    modality: str = "fingernail",
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[Image.Image, List[int]]:
    """
    Automated heuristic ROI localization for user-uploaded photos
    when exact dataset annotations are not available.

    Returns:
        (cropped_roi_image, [ymin, xmin, ymax, xmax])
    """
    w, h = image.size
    img_rgb = image.convert("RGB")
    np_img = np.array(img_rgb).astype(float)

    modality = modality.lower().strip()

    if "nail" in modality or "finger" in modality:
        # Fingernail bed detection:
        # Nail beds exhibit high red-to-green optical contrast and central/upper orientation
        # Search central vertical strip (20% to 80% width, 15% to 75% height)
        y0, y1 = int(0.15 * h), int(0.75 * h)
        x0, x1 = int(0.20 * w), int(0.80 * w)
        roi_sub = np_img[y0:y1, x0:x1]

        r = roi_sub[..., 0]
        g = roi_sub[..., 1]
        b = roi_sub[..., 2]

        # Erythema saliency: high red, moderate green, distinct luminance
        saliency = (r - g) + 0.5 * (r - b)

        # Standard window size: 25% of dimensions
        win_h = max(20, int(0.25 * (y1 - y0)))
        win_w = max(20, int(0.25 * (x1 - x0)))

        # 2D max search via integral image or grid sampling
        best_score = -1e9
        best_r = 0
        best_c = 0
        step_y = max(4, win_h // 4)
        step_x = max(4, win_w // 4)

        for cy in range(0, roi_sub.shape[0] - win_h, step_y):
            for cx in range(0, roi_sub.shape[1] - win_w, step_x):
                score = np.mean(saliency[cy:cy + win_h, cx:cx + win_w])
                if score > best_score:
                    best_score = score
                    best_r = cy
                    best_c = cx

        ymin = y0 + best_r
        xmin = x0 + best_c
        ymax = min(h, ymin + win_h)
        xmax = min(w, xmin + win_w)

    elif "skin" in modality:
        # Finger skin region detection:
        # Uniform cutaneous region, slightly below nail bed or mid-finger pad
        y0 = int(0.40 * h)
        y1 = int(0.85 * h)
        x0 = int(0.25 * w)
        x1 = int(0.75 * w)
        ymin = max(0, y0)
        ymax = min(h, y1)
        xmin = max(0, x0)
        xmax = min(w, x1)

    elif "conjunctiv" in modality or "eye" in modality:
        # Palpebral conjunctiva detection:
        # In lower eyelid eversion photos, the vascular conjunctival tissue is
        # prominently located in the lower-central region below the sclera
        ymin = int(0.45 * h)
        ymax = min(h, int(0.85 * h))
        xmin = int(0.20 * w)
        xmax = min(w, int(0.80 * w))

    else:
        # Default centered square crop
        side = int(min(w, h) * 0.6)
        ymin = (h - side) // 2
        ymax = ymin + side
        xmin = (w - side) // 2
        xmax = xmin + side

    box = [ymin, xmin, ymax, xmax]
    crop = crop_roi_by_box(image, box, target_size=target_size)
    return crop, box


def draw_roi_overlay(
    image: Image.Image,
    boxes: List[List[int]],
    labels: Optional[List[str]] = None,
    color: str = "#FFD200"
) -> Image.Image:
    """
    Draw clean, high-contrast bounding boxes with labels over the original image.
    """
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size

    for idx, box in enumerate(boxes):
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        label_text = labels[idx] if labels and idx < len(labels) else f"ROI #{idx + 1}"

        # Draw thick rectangle
        for offset in range(3):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=color
            )

        # Label tag above box
        tag_w = max(50, len(label_text) * 7 + 10)
        tag_h = 16
        tag_y = max(0, ymin - tag_h)
        draw.rectangle([xmin, tag_y, xmin + tag_w, tag_y + tag_h], fill=color)
        draw.text((xmin + 4, tag_y + 2), label_text, fill="#0F0C29")

    return overlay
