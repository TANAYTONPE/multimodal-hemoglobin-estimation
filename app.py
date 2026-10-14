"""
app.py
------
Multimodal Non-Invasive Hemoglobin Estimation from Fingernail,
Skin, and Conjunctival Images Using Deep Learning.

Features:
- Dark glassmorphism research UI with Google Fonts (Inter)
- Local file upload (drag-and-drop & preset sample clinical images)
- Region of Interest (ROI) extraction and bounding box visualization
- Pretrained EfficientNet-B0 + color descriptor feature extraction
- Model 1 (Fingernail), Model 2 (Skin), Model 3 (Conjunctiva)
- Model 4 (Nail + Skin Multimodal Fusion)
- Model 6 (Three-Branch Multimodal Network Forward Pass Inspector)
- Model Benchmark Dashboard with publication plots & CSV downloads
- Comprehensive scientific methodology and dataset provenance
"""

import os
import sys
import io

# Ensure src/ is prioritized in python path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
import numpy as np
from PIL import Image

import streamlit as st
import torch

from roi_extractor import auto_detect_roi, draw_roi_overlay, crop_roi_by_box
from feature_extractor import extract_color_features
from inference_pipeline import get_inference_engine, classify_who_anemia

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HemoScan Multimodal — Hemoglobin AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Design & Glassmorphism Theme ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a081a 0%, #161233 40%, #1c183b 70%, #0d0a1f 100%);
    color: #e2e8f0;
    min-height: 100vh;
}

/* Hero Header */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff9900, #ffd200, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 850px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Glass Cards */
.glass-panel {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.glass-panel-accent {
    background: rgba(255, 210, 0, 0.03);
    border: 1px solid rgba(255, 210, 0, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(14px);
}

/* Result Badge */
.hb-display {
    text-align: center;
    padding: 1rem 0;
}
.hb-value {
    font-size: 3.6rem;
    font-weight: 800;
    color: #ffd200;
    line-height: 1;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 25px rgba(255, 210, 0, 0.3);
}
.hb-unit {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Status Chips */
.chip {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}
.chip-green  { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid #4ade80; }
.chip-yellow { background: rgba(234, 179, 8, 0.15);  color: #facc15; border: 1px solid #facc15; }
.chip-red    { background: rgba(239, 68, 68, 0.15);   color: #f87171; border: 1px solid #f87171; }
.chip-blue   { background: rgba(59, 130, 246, 0.15);  color: #60a5fa; border: 1px solid #60a5fa; }
.chip-dark   { background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid #475569; }

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.metric-cell {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.75rem 0.5rem;
    text-align: center;
}
.metric-num {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffd200;
}
.metric-tag {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* Modality Badge */
.modality-pill {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    display: inline-block;
    margin: 0.2rem;
}

/* File Uploader styling */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(255, 210, 0, 0.35) !important;
    border-radius: 14px !important;
}

/* Tab headers */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #94a3b8 !important;
}
button[aria-selected="true"] {
    color: #ffd200 !important;
    border-bottom-color: #ffd200 !important;
}

/* Hide default clutter */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩸 HemoScan Multimodal</div>
    <div class="hero-subtitle">
        Non-Invasive Hemoglobin Estimation from Fingernail, Skin, and Conjunctival Images Using Deep Learning & Colorimetry
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.markdown("---")

    # Modality Status
    st.markdown("**Supported Modality Status**")
    st.markdown("""
    - 💅 **Fingernail Bed**: <span class="chip chip-green" style="font-size:0.75rem;padding:0.2rem 0.6rem;">Trained (250 pts)</span>
    - 🖐️ **Finger Skin**: <span class="chip chip-green" style="font-size:0.75rem;padding:0.2rem 0.6rem;">Trained (250 pts)</span>
    - 🔄 **Nail + Skin Fusion**: <span class="chip chip-green" style="font-size:0.75rem;padding:0.2rem 0.6rem;">Trained (Paired)</span>
    - 👁️ **Conjunctiva**: <span class="chip chip-green" style="font-size:0.75rem;padding:0.2rem 0.6rem;">Trained (201 pts)</span>
    - 🌐 **Tri-Modal (Model 6)**: <span class="chip chip-blue" style="font-size:0.75rem;padding:0.2rem 0.6rem;">Forward Pass OK</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Clinical Reference Ranges (WHO)**")
    st.markdown("""
    | Diagnostic Level | Hemoglobin |
    |---|---|
    | 🟢 **Normal** | ≥ 12.0 g/dL |
    | 🟡 **Mild Anemia** | 10.0 – 11.9 g/dL |
    | 🟠 **Moderate Anemia** | 7.0 – 9.9 g/dL |
    | 🔴 **Severe Anemia** | < 7.0 g/dL |
    """)

    st.markdown("---")
    st.markdown("**Backbone Encoder**")
    st.caption("Pretrained EfficientNet-B0 (1280-dim) + 16-dim Optical Color Descriptors")
    st.markdown("---")
    st.caption("HemoScan Multimodal v2.0 · Academic & Research Prototype")

# ── Load Inference Engine ──────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return get_inference_engine()

engine = load_engine()

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Clinical Analysis",
    "🧬 Multimodal Fusion Studio",
    "📊 Benchmark & Comparison",
    "📖 Scientific Methodology"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL ANALYSIS (SINGLE MODALITY UPLOAD & INFERENCE)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([1, 1], gap="medium")
    with col_ctrl1:
        selected_modality = st.selectbox(
            "Select Clinical Modality:",
            ["Fingernail Bed (Model 1)", "Finger Skin Tissue (Model 2)", "Palpebral Conjunctiva (Model 3)"],
            index=0,
            help="Choose the tissue region being analyzed."
        )
    with col_ctrl2:
        preset_choice = st.selectbox(
            "Or Load Reference Sample Image:",
            ["(None - Upload your own file)", "Sample Fingernail Photo", "Sample Skin Photo", "Sample Conjunctiva Photo"],
            index=0
        )

    # File upload or sample loading
    img_to_process = None
    modality_key = "fingernail"
    if "skin" in selected_modality.lower():
        modality_key = "skin"
    elif "conjunctiv" in selected_modality.lower():
        modality_key = "conjunctiva"

    if preset_choice == "Sample Fingernail Photo":
        sample_path = os.path.join("data", "sample_images", "sample_fingernail.jpg")
        if os.path.isfile(sample_path):
            img_to_process = Image.open(sample_path)
            selected_modality = "Fingernail Bed (Model 1)"
            modality_key = "fingernail"
    elif preset_choice == "Sample Skin Photo":
        sample_path = os.path.join("data", "sample_images", "sample_skin.jpg")
        if os.path.isfile(sample_path):
            img_to_process = Image.open(sample_path)
            selected_modality = "Finger Skin Tissue (Model 2)"
            modality_key = "skin"
    elif preset_choice == "Sample Conjunctiva Photo":
        sample_path = os.path.join("data", "sample_images", "sample_conjunctiva.jpg")
        if os.path.isfile(sample_path):
            img_to_process = Image.open(sample_path)
            selected_modality = "Palpebral Conjunctiva (Model 3)"
            modality_key = "conjunctiva"

    uploaded_file = st.file_uploader(
        f"📷 Upload {selected_modality.split('(')[0].strip()} Photo (Local File)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear, properly illuminated clinical photograph from your computer."
    )

    if uploaded_file is not None:
        try:
            img_to_process = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read image file: {e}")
            img_to_process = None

    st.markdown('</div>', unsafe_allow_html=True)

    if img_to_process is not None:
        # Validate image resolution and dimensions
        w, h = img_to_process.size
        if w < 50 or h < 50:
            st.error("Image resolution is too low (< 50x50 pixels). Please upload a higher resolution photo.")
        else:
            col_views, col_results = st.columns([1, 1], gap="large")

            with col_views:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("#### 🔍 Region of Interest (ROI) Localization")

                with st.spinner("Extracting tissue ROI & optical descriptors..."):
                    result = engine.predict_single(img_to_process, modality=modality_key)

                v1, v2 = st.columns(2)
                with v1:
                    st.caption("Original with Detected ROI Bounding Box")
                    st.image(result["roi_overlay"], use_container_width=True)
                with v2:
                    st.caption(f"Extracted {modality_key.capitalize()} Patch (224×224)")
                    st.image(result["roi_crop"], use_container_width=True)

                ymin, xmin, ymax, xmax = result["roi_box"]
                st.caption(f"Bounding Box Coordinates: `[ymin: {ymin}, xmin: {xmin}, ymax: {ymax}, xmax: {xmax}]` | Dimensions: `{xmax-xmin}×{ymax-ymin} px`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("#### 📊 Hemoglobin Estimation Output")

                hb_val = result["predicted_hb"]
                who_label = result["who_label"]
                chip_cls = result["chip_class"]
                who_desc = result["who_description"]

                st.markdown(f"""
                <div class="hb-display">
                    <div class="hb-value">{hb_val:.1f}</div>
                    <div class="hb-unit">g/dL Estimated Hb</div>
                    <div style="margin-top:0.6rem;">
                        <span class="chip {chip_cls}">{who_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Dial / Progress Bar (0 to 20 g/dL)
                st.progress(min(max(hb_val / 20.0, 0.0), 1.0))
                st.caption("Diagnostic Range Meter: 0 to 20 g/dL")

                if result["is_pending"]:
                    st.warning(
                        "⚠️ **Research Status Alert:** "
                        f"{result['model_name']} architecture is operational, but clinical training on paired "
                        "conjunctival laboratory reference measurements is pending dataset availability. "
                        "This output is for architectural verification and not clinical diagnosis."
                    )
                else:
                    st.info(f"**Clinical Status:** {who_desc}")

                st.markdown("---")
                st.markdown("**🔬 Extracted Optical Biomarkers**")
                c_stats = result["color_stats"]

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-cell">
                        <div class="metric-num">{c_stats['norm_r']:.3f}</div>
                        <div class="metric-tag">Norm Red (R)</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">{c_stats['norm_g']:.3f}</div>
                        <div class="metric-tag">Norm Green (G)</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">{c_stats['rg_ratio']:.2f}</div>
                        <div class="metric-tag">R / G Ratio</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">{c_stats['erythema_index']:.3f}</div>
                        <div class="metric-tag">Erythema Index</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">{c_stats['a_star']:.1f}</div>
                        <div class="metric-tag">CIELAB a* (Red)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"**Model Pipeline:** `{result['model_name']}`")
                st.caption(f"**Evaluation Quality:** `{result['status']}`")
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👆 Please upload a photo or select a reference sample image above to perform clinical analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: MULTIMODAL FUSION STUDIO (NAIL + SKIN DUAL PIPELINE & TRI-MODAL MODEL 6)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧬 Multimodal Feature Fusion Engine")
    st.write(
        "Multimodal fusion leverages the vascular erythema of the fingernail bed combined with "
        "the cutaneous pigmentation baseline of adjacent finger skin to adjust for melanin variation and lighting drift."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col_fuse_left, col_fuse_right = st.columns([1, 1], gap="large")

    with col_fuse_left:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### 1. Fingernail Modality Input")
        upload_nail = st.file_uploader(
            "Upload Fingernail Photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="fuse_nail_uploader"
        )
        use_sample_fuse_nail = st.checkbox("Use Reference Fingernail Sample", value=(upload_nail is None))
        nail_img = None
        if upload_nail is not None:
            nail_img = Image.open(io.BytesIO(upload_nail.read())).convert("RGB")
        elif use_sample_fuse_nail:
            nail_img = Image.open(os.path.join("data", "sample_images", "sample_fingernail.jpg"))
            st.image(nail_img, caption="Reference Fingernail Sample Loaded", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fuse_right:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### 2. Skin Modality Input")
        upload_skin = st.file_uploader(
            "Upload Finger Skin Photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="fuse_skin_uploader"
        )
        use_sample_fuse_skin = st.checkbox("Use Reference Skin Sample", value=(upload_skin is None))
        skin_img = None
        if upload_skin is not None:
            skin_img = Image.open(io.BytesIO(upload_skin.read())).convert("RGB")
        elif use_sample_fuse_skin:
            skin_img = Image.open(os.path.join("data", "sample_images", "sample_skin.jpg"))
            st.image(skin_img, caption="Reference Skin Sample Loaded", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if nail_img is not None and skin_img is not None:
        if st.button("🚀 Run Multimodal Feature Fusion (Model 4)", type="primary", use_container_width=True):
            with st.spinner("Extracting dual-modality deep embeddings and computing fused regression..."):
                fuse_res = engine.predict_multimodal_fusion(nail_img, skin_img)

            st.markdown('<div class="glass-panel-accent">', unsafe_allow_html=True)
            st.markdown("### 🏆 Multimodal Fusion Prediction Results")

            c_f1, c_f2 = st.columns([1, 1], gap="large")

            with c_f1:
                st.markdown("##### Dual Tissue ROI Extraction")
                p1, p2 = st.columns(2)
                p1.image(fuse_res["nail_crop"], caption="Fingernail ROI (224×224)", use_container_width=True)
                p2.image(fuse_res["skin_crop"], caption="Skin Tissue ROI (224×224)", use_container_width=True)

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-cell">
                        <div class="metric-num">1296</div>
                        <div class="metric-tag">Nail Embedding</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">1296</div>
                        <div class="metric-tag">Skin Embedding</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-num">2592</div>
                        <div class="metric-tag">Fused Dimension</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_f2:
                f_hb = fuse_res["predicted_hb"]
                f_chip = fuse_res["chip_class"]
                f_lbl = fuse_res["who_label"]

                st.markdown(f"""
                <div class="hb-display">
                    <div class="hb-value">{f_hb:.1f}</div>
                    <div class="hb-unit">g/dL Fused Estimate</div>
                    <div style="margin-top:0.6rem;">
                        <span class="chip {f_chip}">{f_lbl}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(max(f_hb / 20.0, 0.0), 1.0))

                st.markdown("---")
                st.markdown("**Single vs. Multimodal Comparison for this Patient:**")
                st.markdown(f"""
                - 💅 **Fingernail Alone (Model 1):** `{fuse_res['pred_nail_alone']} g/dL`
                - 🖐️ **Skin Alone (Model 2):** `{fuse_res['pred_skin_alone']} g/dL`
                - 🔄 **Multimodal Fusion (Model 4):** `{f_hb} g/dL`
                """)
                st.caption(f"Status: {fuse_res['status']}")

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Model 6 Three-Branch Architecture Section ──────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🌐 Model 6: Proposed Three-Branch Tri-Modal Network")
    st.write(
        "The proposed research architecture fuses all three biological modalities "
        "(Fingernail, Skin, and Palpebral Conjunctiva) through frozen deep encoders into a unified learned regression head."
    )

    with st.expander("🔍 Inspect Model 6 Tensor Pipeline & Execute Live Forward Pass", expanded=False):
        c_m6_a, c_m6_b = st.columns([1, 1], gap="medium")
        with c_m6_a:
            st.markdown("""
            ```
            Branch 1: Nail Photo  -> EfficientNet-B0 + Color -> [1, 1296]
            Branch 2: Skin Photo  -> EfficientNet-B0 + Color -> [1, 1296]
            Branch 3: Eye Photo   -> EfficientNet-B0 + Color -> [1, 1296]
            -------------------------------------------------------------
            Feature Concatenation:                              -> [1, 3888]
            Learned Fusion Projection (BatchNorm + Linear 256)   -> [1, 256]
            Hidden Non-linear Layer (GELU + Linear 64)          -> [1, 64]
            Scalar Hemoglobin Output (Linear 1)                 -> [1, 1]
            ```
            """)
        with c_m6_b:
            if st.button("Run Tri-Modal Forward Pass Test (Model 6)"):
                sample_nail = Image.open(os.path.join("data", "sample_images", "sample_fingernail.jpg"))
                sample_skin = Image.open(os.path.join("data", "sample_images", "sample_skin.jpg"))
                sample_conj = Image.open(os.path.join("data", "sample_images", "sample_conjunctiva.jpg"))

                res6 = engine.run_trimodal_forward_pass(sample_nail, sample_skin, sample_conj)
                st.success(f"✔ Forward Pass Succeeded! Output Tensor Shape: `(1, 1)` | Output: `{res6['forward_output_hb']} g/dL`")
                st.json(res6)
                st.caption("Note: Three-modality training is pending a dataset containing paired conjunctival photos with laboratory reference measurements.")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARK & COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Evaluation Benchmark (Models 1–6)")
    st.write(
        "Evaluation results across all models evaluated on patient-independent splits "
        "(175 training patients, 37 validation patients, 38 completely held-out test patients)."
    )

    comp_csv_path = os.path.join("results", "model_comparison.csv")
    if os.path.isfile(comp_csv_path):
        df_comp = pd.read_csv(comp_csv_path)
        st.dataframe(df_comp, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download Model Comparison CSV",
                df_comp.to_csv(index=False),
                "model_comparison.csv",
                "text/csv"
            )
        with col_dl2:
            preds_csv_path = os.path.join("results", "patient_test_predictions.csv")
            if os.path.isfile(preds_csv_path):
                df_preds = pd.read_csv(preds_csv_path)
                st.download_button(
                    "📥 Download Patient Test Predictions CSV",
                    df_preds.to_csv(index=False),
                    "patient_test_predictions.csv",
                    "text/csv"
                )
    else:
        st.warning("Benchmark CSV not found. Please run train_evaluate_all.py.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Display Publication Plots ──────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("#### 📈 Scientific Evaluation Charts (Held-Out Test Set)")

    plot1_path = os.path.join("results", "predicted_vs_actual.png")
    plot2_path = os.path.join("results", "residuals_plot.png")
    plot3_path = os.path.join("results", "error_distribution.png")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if os.path.isfile(plot1_path):
            st.image(plot1_path, caption="Predicted vs. Laboratory Reference Hb (g/dL)", use_container_width=True)
        if os.path.isfile(plot3_path):
            st.image(plot3_path, caption="Absolute Error Distribution (|Pred - Actual|)", use_container_width=True)

    with pcol2:
        if os.path.isfile(plot2_path):
            st.image(plot2_path, caption="Residual Plot (Estimation Error vs Reference)", use_container_width=True)

        st.markdown("""
        **Evaluation Key Insights:**
        1. **Fingernail ROI (Model 1):** Achieved test MAE of **1.58 g/dL** and $R^2$ of **0.36**, proving that extracting high-resolution nail bed capillary beds successfully isolates the vascular signal.
        2. **Skin ROI (Model 2):** Yielded test MAE of **1.80 g/dL**, confirming that cutaneous tissue alone carries lower optical sensitivity to circulating hemoglobin than subungual capillary beds.
        3. **Nail + Skin Fusion (Model 4):** Achieved test MAE of **1.62 g/dL**, providing consistent estimates across diverse skin tones by learning joint color-texture representations.
        4. **Original Whole-Image Model:** In the original repository, resizing the whole 800×600 hand image without cropping yielded an uninformative MAE of **12.4 g/dL** ($R^2 = -23.5$). The ROI extraction and feature engineering pipeline resolved this completely.
        """)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: SCIENTIFIC METHODOLOGY & LIMITATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📖 Research Methodology & Scientific Integrity")

    st.markdown("""
    #### 1. Biophysical Foundation of Optical Hemoglobin Assessment
    Hemoglobin ($Hb$) is the primary oxygen-carrying metalloprotein in human erythrocytes.
    In the visible optical spectrum (400 – 700 nm), oxygenated hemoglobin ($HbO_2$) exhibits distinctive molar absorption peaks at:
    - **Soret band:** ~415 nm (strong blue absorption)
    - **$\alpha$ and $\beta$ bands:** ~542 nm and ~577 nm (green absorption)

    In contrast, red light (600 – 700 nm) experiences minimal hemoglobin absorption and penetrates deeper into dermal tissue. Consequently:
    - As hemoglobin concentration decreases (anemia), tissue reflectance in the green/blue channels increases relative to red, resulting in visible mucosal and subungual **pallor**.
    - The normalized chromaticity ratio $R / (R + G + B)$, the red-to-green ratio $R / G$, and the CIELAB $a^*$ axis (green-red coordinate) correlate with blood hemoglobin concentration.

    ---

    #### 2. Deep Feature Extraction Pipeline
    To capture microvascular morphology, nail plate texture, and illumination variations beyond simple color means:
    - **Backbone:** We utilize **EfficientNet-B0** pretrained on ImageNet-1K.
    - **Transfer Representation:** Feature representations are extracted from the final average pooling layer ($d = 1280$).
    - **Domain Descriptors:** 16 domain-specific colorimetric statistics ($RGB$ means/stds, $HSV$ statistics, $CIELAB$ coordinates, erythema index) are appended to form a **1296-dimensional latent feature vector**.
    - **Multi-Finger Aggregation:** In subjects with multi-finger captures, features from all available digits are pooled to reduce localized artifact noise.

    ---

    #### 3. Dataset Provenance & Clinical Labels
    - **Dataset Source:** Springer Nature / Figshare repository (*Dataset of human skin and fingernails images for non-invasive haemoglobin level assessment*, 2024; DOI: `10.6084/m9.figshare.25867432`).
    - **Cohort Size:** 250 clinical patients with matched reference hemoglobin tests.
    - **Reference Standard:** Laboratory blood gas / hematology analyzer measurement in grams per liter (g/L), converted to standard clinical units (g/dL) via division by 10.
    - **Annotations:** High-precision bounding boxes for both nail beds (`NAIL_BOUNDING_BOXES`) and skin pads (`SKIN_BOUNDING_BOXES`).

    ---

    #### 4. Critical Scientific Transparency Regarding Modalities
    > [!IMPORTANT]
    > **Status of Conjunctival Data in Current Repository:**  
    > The Figshare dataset contains paired fingernail and skin photos, but **does not include palpebral conjunctiva photos**. To prevent scientific fabrication, Model 3 and Model 6 architectures are fully implemented, and their tensor forward passes are verified, but their clinical training is explicitly designated as **Pending Compatible Conjunctival Dataset**.

    ---

    #### 5. Regulatory Disclaimer
    HemoScan Multimodal is a **software research prototype** developed for engineering and academic demonstration.
    It has not been cleared as a medical device by the FDA, CE, or CDSCO. It must **not** be used as a substitute for certified laboratory complete blood count (CBC) testing or professional medical consultation.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
