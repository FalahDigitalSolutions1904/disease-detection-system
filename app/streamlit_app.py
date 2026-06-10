"""
streamlit_app.py
================
Clinical-grade UI for doctors. Integrates clinical metrics, Grad-CAM++ overlays, 
epistemic uncertainty (MC-dropout), patient metadata, and ReportLab PDF clinical 
report generation in a unified diagnostic interface.
"""

import io
import sys
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path
from datetime import datetime

# Allow src imports when running from app/ directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
    IMAGE_SIZE, ANALYSIS_MODES, DEFAULT_MODE, CLINICAL_RISK_LEVELS, MODELS_DIR, PROCESSED_DATA_DIR,
    mc_dropout_predict, uncertainty_label, determine_clinical_risk
)
from src.model import build_model, make_gradcam_heatmap, overlay_heatmap
from src.data_loader import apply_clahe


# ─── Page Settings ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Decision Support System (CDSS)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Premium Clinical Theme CSS ────────────────────────────────────────────────

def render_header():
    """Apply CSS styles and render clinical dashboard header."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@300;500&display=swap');
        
        /* Main page adjustments */
        .stApp {
            background-color: #0f111a;
            color: #e2e8f0;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Main Title styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -0.05em;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #94a3b8;
            font-weight: 400;
            margin-top: -10px;
            margin-bottom: 15px;
        }
        
        /* Glassmorphic Cards */
        .clinical-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .clinical-card:hover {
            border-color: rgba(96, 165, 250, 0.3);
            transform: translateY(-2px);
        }
        
        /* Metric Badges */
        .clinical-metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            font-size: 1.8rem;
            color: #f8fafc;
        }
        
        /* Regulatory Disclaimer styling */
        .regulatory-disclaimer {
            font-size: 0.8rem;
            color: #64748b;
            border-left: 3px solid #e11d48;
            padding-left: 15px;
            margin-top: 30px;
            line-height: 1.4;
        }
    </style>
    <div class='main-title'>🔬 Clinical Decision Support System</div>
    <div class='sub-title'>AI-Assisted Diagnostics · EfficientNetB4 · Grad-CAM++ Explainability · MC-Dropout Uncertainty</div>
    <hr style='border: 0; height: 1px; background: linear-gradient(to right, rgba(96, 165, 250, 0.8), rgba(15, 17, 26, 0)); margin: 15px 0 25px;'>
    """, unsafe_allow_html=True)


def render_prediction_card(pred_class: str, confidence: float, risk_level: str, risk_color: str, risk_icon: str):
    """Render a premium prediction summary card with color coding."""
    st.markdown(f"""
    <div class='clinical-card'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 0.95rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Diagnostic Analysis Result</span>
            <span style='background: {risk_color}22; color: {risk_color}; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; border: 1px solid {risk_color}44;'>
                {risk_icon} {risk_level.upper()}
            </span>
        </div>
        <div style='font-size: 2.2rem; font-weight: 800; color: #ffffff; line-height: 1.1; margin-bottom: 8px;'>{pred_class}</div>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='font-size: 1.1rem; color: #94a3b8;'>Model Confidence:</span>
            <span style='font-size: 1.25rem; font-weight: 800; color: #60a5fa;'>{confidence:.1%}</span>
        </div>
        <div style='margin-top: 15px; height: 6px; background: rgba(255, 255, 255, 0.05); border-radius: 999px; overflow: hidden;'>
            <div style='width: {confidence*100}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 999px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_chart(class_names: list, probabilities: np.ndarray):
    """Render a clean, custom-styled Plotly horizontal bar chart."""
    indices = np.argsort(probabilities)
    sorted_probs = [probabilities[i] for i in indices]
    sorted_classes = [class_names[i] for i in indices]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation="h",
        marker=dict(
            color=sorted_probs,
            colorscale=[[0, "#1d4ed8"], [0.5, "#3b82f6"], [1.0, "#60a5fa"]],
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        hovertemplate="Class: %{y}<br>Probability: %{x:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(size=14, color="#f8fafc", family="Outfit")
        ),
        xaxis=dict(
            title="Probability",
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            range=[0, 1.05],
            tickformat=".0%",
            font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            font=dict(color="#f8fafc")
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Outfit"),
        height=240,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_heatmap_section(original_img: np.ndarray, overlay_bgr: np.ndarray):
    """Render side-by-side view of original scan and explainability map."""
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.subheader("🗺️ Explainability & Localization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;'>Original Clinical Image</div>", unsafe_allow_html=True)
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, use_column_width=True)
        
    with col2:
        st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;'>Grad-CAM++ Attention Heatmap</div>", unsafe_allow_html=True)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, use_column_width=True)
        
    st.markdown("""
        <div style='font-size: 0.85rem; color: #94a3b8; margin-top: 15px; text-align: center;'>
            <i>The heatmap highlights regions the model focused on (warm colors like red/yellow represent higher diagnostic relevance).</i>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_uncertainty_section(results: dict, label: str):
    """Render the MC-Dropout epistemic uncertainty and calibration parameters."""
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.subheader("📊 Epistemic Uncertainty Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style='text-align: center;'>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Predictive Entropy</div>
                <div class='clinical-metric-value'>{results['predictive_entropy']:.4f}</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Shannon Entropy score</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div style='text-align: center;'>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Inference Reliability</div>
                <div class='clinical-metric-value'>{results['reliability']:.1%}</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Bayesian consistency</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div style='text-align: center;'>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Uncertainty Level</div>
                <div class='clinical-metric-value' style='font-size: 1.1rem; padding-top: 8px;'>{label}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 View Stochastic Variance breakdown (30 MC runs)"):
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>Standard Deviation (epistemic uncertainty) per class. High values indicate model disagreement across stochastic forward passes.</p>", unsafe_allow_html=True)
        st.bar_chart(results["std"])
        
    st.markdown("</div>", unsafe_allow_html=True)


def render_regulatory_notices():
    """Render clinical disclaimers and regulatory notifications."""
    st.markdown("""
    <div class='regulatory-disclaimer'>
        <strong>⚠️ DEMO/REGULATORY DISCLAIMER</strong><br>
        This software is a prototype research application representing a Clinical Decision Support System (CDSS) for diagnostic image analysis. 
        It has NOT been cleared or approved by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory body 
        for clinical diagnosis. The model weights are for demonstration/educational use only. Predictions generated by this system must always 
        be verified by a licensed healthcare professional. Never rely solely on this software to make treatment or diagnostic decisions.
    </div>
    """, unsafe_allow_html=True)


# ─── PDF Report Exporter ────────────────────────────────────────────────────────

def export_pdf_report(
    image: Image.Image,
    gradcam_img: np.ndarray,
    pred_class: str,
    confidence: float,
    uncertainty: str,
    class_names: list,
    probabilities: np.ndarray,
    patient_meta: dict,
) -> bytes:
    """
    Generate a highly formatted PDF report with ReportLab.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import cm

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm
        )
        
        styles = getSampleStyleSheet()
        
        primary_color = HexColor("#1e3a8a")  # Navy Blue
        accent_color = HexColor("#3b82f6")   # Light Blue
        text_color = HexColor("#1f2937")     # Dark Charcoal
        
        styles['Normal'].textColor = text_color
        styles['Normal'].fontSize = 10
        styles['Normal'].leading = 14
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=primary_color,
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            'ReportSection',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=primary_color,
            spaceBefore=15,
            spaceAfter=10,
            keepWithNext=True
        )

        meta_label_style = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            textColor=HexColor("#4b5563")
        )

        story = []

        # 1. Header
        story.append(Paragraph("CLINICAL DIAGNOSTIC REPORT", title_style))
        story.append(Paragraph("AI-Assisted Diagnostics Support System", ParagraphStyle('Sub', parent=styles['Normal'], textColor=accent_color, fontSize=11, fontName='Helvetica-Oblique')))
        story.append(Spacer(1, 0.4 * cm))
        
        # Separator line
        line_table = Table([[""]], colWidths=[18 * cm])
        line_table.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1.5, primary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0)
        ]))
        story.append(line_table)
        story.append(Spacer(1, 0.4 * cm))

        # 2. Metadata Table
        meta_data = [
            [
                Paragraph("Patient Name:", meta_label_style), Paragraph(patient_meta.get("name", "N/A"), styles['Normal']),
                Paragraph("Scan Modality:", meta_label_style), Paragraph(patient_meta.get("modality", "N/A"), styles['Normal'])
            ],
            [
                Paragraph("Age / Gender:", meta_label_style), Paragraph(f"{patient_meta.get('age', 'N/A')} / {patient_meta.get('gender', 'N/A')}", styles['Normal']),
                Paragraph("Analysis Date:", meta_label_style), Paragraph(patient_meta.get("date", "N/A"), styles['Normal'])
            ],
            [
                Paragraph("Patient ID:", meta_label_style), Paragraph(patient_meta.get("id", "N/A"), styles['Normal']),
                Paragraph("System Version:", meta_label_style), Paragraph("v1.0.0 (Demo Mode)", styles['Normal'])
            ]
        ]
        
        meta_table = Table(meta_data, colWidths=[3.2 * cm, 5.8 * cm, 3.2 * cm, 5.8 * cm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), HexColor("#f3f4f6")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.6 * cm))

        # 3. Diagnostic Findings
        story.append(Paragraph("1. Diagnostic Summary Findings", section_style))
        
        findings_data = [
            [Paragraph("<b>Predicted Pathological Finding:</b>", styles['Normal']), Paragraph(f"<b>{pred_class}</b>", styles['Normal'])],
            [Paragraph("<b>Model Confidence:</b>", styles['Normal']), Paragraph(f"{confidence:.2%}", styles['Normal'])],
            [Paragraph("<b>Bayesian Uncertainty Level:</b>", styles['Normal']), Paragraph(uncertainty, styles['Normal'])],
        ]
        findings_table = Table(findings_data, colWidths=[6.5 * cm, 11.5 * cm])
        findings_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 0.5 * cm))

        # 4. Probabilities Breakdown
        story.append(Paragraph("2. Differential Diagnosis Probabilities Breakdown", section_style))
        prob_rows = [[Paragraph("<b>Target Diagnosis Class</b>", meta_label_style), Paragraph("<b>AI Confidence Score</b>", meta_label_style)]]
        
        for name, prob in zip(class_names, probabilities):
            prob_rows.append([Paragraph(name, styles['Normal']), Paragraph(f"{prob:.2%}", styles['Normal'])])
            
        prob_table = Table(prob_rows, colWidths=[9 * cm, 9 * cm])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), HexColor("#e5e7eb")),
            ('PADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, HexColor("#d1d5db")),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.6 * cm))

        # 5. Visual Localization Heatmap
        story.append(Paragraph("3. Localized Pathology Attention Map (Grad-CAM++)", section_style))
        
        overlay_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)
        
        orig_buffer = io.BytesIO()
        image.resize((220, 220)).save(orig_buffer, format="PNG")
        orig_buffer.seek(0)
        
        heat_buffer = io.BytesIO()
        overlay_pil.resize((220, 220)).save(heat_buffer, format="PNG")
        heat_buffer.seek(0)
        
        img_table_data = [
            [
                RLImage(orig_buffer, width=7.5*cm, height=7.5*cm),
                RLImage(heat_buffer, width=7.5*cm, height=7.5*cm)
            ],
            [
                Paragraph("<font color='#6b7280' size='8'>Original Input Scan</font>", ParagraphStyle('C1', parent=styles['Normal'], alignment=1)),
                Paragraph("<font color='#6b7280' size='8'>Grad-CAM++ Attention Overlay</font>", ParagraphStyle('C2', parent=styles['Normal'], alignment=1))
            ]
        ]
        
        img_table = Table(img_table_data, colWidths=[9 * cm, 9 * cm])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
            ('TOPPADDING', (0,1), (-1,1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.6 * cm))
        
        # 6. Disclaimer
        story.append(Paragraph("<b>REGULATORY & CLINICAL USE DISCLAIMER:</b> This report is generated automatically by a clinical decision support research prototype. It has not been approved for diagnostic use by the FDA, EMA, or other regulatory bodies. The contents must be reviewed and signed off by a qualified medical professional prior to establishing any diagnosis or treatment plan.", ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=HexColor("#dc2626"))))
        
        story.append(Spacer(1, 1.0 * cm))
        
        # Signatures
        sig_data = [
            [Paragraph("_____________________________<br>Reporting Radiologist / Doctor Signature", styles['Normal']), 
             Paragraph("_____________________________<br>Reviewing AI Scientist / Tech Signature", styles['Normal'])]
        ]
        sig_table = Table(sig_data, colWidths=[9 * cm, 9 * cm])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(sig_table)

        doc.build(story)
        return buffer.getvalue()

    except Exception as e:
        print(f"Error generating PDF: {e}")
        st.error(f"Error generating PDF: {e}")
        return b""


# ─── Load Model (cached per analysis mode) ──────────────────────────────────────

@st.cache_resource
def load_model(mode_key: str):
    """Build and load weights for the specified clinical mode."""
    class_names = ANALYSIS_MODES[mode_key]["classes"]
    num_classes = len(class_names)

    # Check if this mode has training directories
    data_dir = PROCESSED_DATA_DIR / mode_key / "train"
    if data_dir.exists():
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        if class_dirs:
            num_classes = len(class_dirs)

    # Build model using ImageNet weights if available
    model = build_model(num_classes=num_classes, weights="imagenet")

    # Load local training weights if present
    weights_candidates = [
        MODELS_DIR / f"efficientnet_b4_weights_{mode_key}.h5",
        MODELS_DIR / "checkpoint.weights.h5",
    ]

    for weights_path in weights_candidates:
        if weights_path.exists():
            try:
                model.load_weights(str(weights_path))
                return model
            except Exception:
                continue

    return model


def check_data_availability(mode_key: str) -> dict:
    """Check if training data is available and return counts."""
    base = PROCESSED_DATA_DIR / mode_key
    # Also check raw data as validation
    raw_base = RAW_DATA_DIR / "isic" if mode_key == "skin_lesion" else RAW_DATA_DIR / mode_key
    
    result = {"available": False, "train": 0, "val": 0, "test": 0, "total": 0}
    
    for split in ["train", "val", "test"]:
        split_dir = base / split
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*.jpg"))
            result[split] = count
            result["total"] += count
            
    # If no processed data but raw data exists, count raw files
    if result["total"] == 0 and raw_base.exists():
        raw_count = sum(1 for _ in raw_base.rglob("*.jpg")) + sum(1 for _ in raw_base.rglob("*.jpeg"))
        result["total"] = raw_count
        result["available"] = raw_count > 0
    else:
        result["available"] = result["total"] > 0
        
    return result


# ─── Main Application Logic ───────────────────────────────────────────────────

def main():
    render_header()

    # ─── Sidebar Panel ───
    with st.sidebar:
        st.header("⚙️ Diagnostics Panel")

        selected_mode = st.selectbox(
            "Select Modality / Analysis Mode",
            options=list(ANALYSIS_MODES.keys()),
            index=2, # Default to skin_lesion
            format_func=lambda x: f"{ANALYSIS_MODES[x]['icon']} {ANALYSIS_MODES[x]['name']}"
        )

        mode_info = ANALYSIS_MODES[selected_mode]
        st.info(f"**Modality:** {mode_info['description']}")

        # Dataset status
        data_status = check_data_availability(selected_mode)
        if data_status["available"]:
            st.success(f"✅ Training data: {data_status['total']:,} images")
        else:
            st.warning("⚠️ No training data found")
            with st.expander("📥 How to download data"):
                st.code(
                    f"python -m src.train --mode {selected_mode}",
                    language="bash"
                )
                for ds_info in mode_info.get("datasets", []):
                    st.caption(ds_info)

        st.markdown("<hr style='border: 0.5px solid #333; margin: 15px 0;'>", unsafe_allow_html=True)

        st.subheader("👤 Patient Metadata")
        patient_name = st.text_input("Full Name", value="Jane Doe")
        patient_id = st.text_input("Patient ID / Medical Record #", value="MRN-8472-A")

        col_age, col_gen = st.columns(2)
        patient_age = col_age.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = col_gen.selectbox("Gender", options=["Female", "Male", "Other", "Unknown"])

        st.markdown("<hr style='border: 0.5px solid #333; margin: 15px 0;'>", unsafe_allow_html=True)

        st.subheader("🛠️ Settings")
        mc_passes = st.slider("MC-Dropout passes", min_value=10, max_value=100, value=30, step=10)
        apply_clahe_flag = st.checkbox("Apply CLAHE Contrast Enhancement", value=True)
        explain_variant = st.selectbox("Explainability Variant", options=["Grad-CAM++", "Grad-CAM"])

    # Load model
    with st.spinner(f"Loading {mode_info['name']} model..."):
        model = load_model(selected_mode)
    class_names = mode_info["classes"]

    # Show warning if running in demo mode
    if not (MODELS_DIR / "checkpoint.weights.h5").exists() and not (MODELS_DIR / f"efficientnet_b4_weights_{selected_mode}.h5").exists():
        st.info(
            f"ℹ️ **Demo mode** — Running with ImageNet pre-trained weights only (no fine-tuning).\n\n"
            f"For accurate clinical predictions, run the training loop:\n"
            f"```\npython -m src.train --mode {selected_mode}\n```"
        )

    # ─── Image Upload ───
    uploaded_file = st.file_uploader(
        f"Upload a front-facing scan for {mode_info['name']}",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if apply_clahe_flag:
            img_bgr_processed = apply_clahe(img_bgr)
        else:
            img_bgr_processed = img_bgr.copy()

        img_resized = cv2.resize(img_bgr_processed, IMAGE_SIZE)
        img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            results = mc_dropout_predict(model, img_array, n_passes=mc_passes)
            pred_class_idx = np.argmax(results["mean"])
            
            # Make sure we don't index out of bounds
            if pred_class_idx < len(class_names):
                pred_class = class_names[pred_class_idx]
            else:
                pred_class = f"Class {pred_class_idx}"
                
            confidence = float(results["mean"][pred_class_idx])

            variant_key = "gradcam++" if explain_variant == "Grad-CAM++" else "gradcam"
            try:
                heatmap = make_gradcam_heatmap(
                    img_array, model,
                    last_conv_layer_name=None,
                    pred_index=pred_class_idx,
                    variant=variant_key,
                )
                overlay = overlay_heatmap(heatmap, img_resized)
            except Exception as e:
                st.warning(f"⚠️ Explainability map unavailable: {e}")
                overlay = img_resized

            render_heatmap_section(img_resized, overlay)

        with col_right:
            uncertainty_lbl = uncertainty_label(results["predictive_entropy"])
            risk_key = determine_clinical_risk(pred_class, confidence, uncertainty_lbl)
            risk_data = CLINICAL_RISK_LEVELS[risk_key]

            render_prediction_card(
                pred_class=pred_class,
                confidence=confidence,
                risk_level=risk_data["label"],
                risk_color=risk_data["color"],
                risk_icon=risk_data["icon"],
            )

            st.markdown(f"""
            <div class='clinical-card' style='border-left: 5px solid {risk_data["color"]};'>
                <div style='font-size: 1.1rem; font-weight: 600; color: {risk_data["color"]}; margin-bottom: 8px;'>📋 Recommended Clinical Action</div>
                <div style='color: #f8fafc; font-size: 0.95rem; line-height: 1.4;'>{risk_data["action"]}</div>
            </div>
            """, unsafe_allow_html=True)

            render_confidence_chart(class_names, results["mean"])
            render_uncertainty_section(results, uncertainty_lbl)

            st.subheader("📄 Clinical Documentation")
            patient_meta = {
                "name": patient_name,
                "id": patient_id,
                "age": str(patient_age),
                "gender": patient_gender,
                "modality": mode_info["name"],
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            if st.button("Generate Formal Diagnostic PDF Report", use_container_width=True):
                pdf_data = export_pdf_report(
                    image=pil_img,
                    gradcam_img=overlay,
                    pred_class=pred_class,
                    confidence=confidence,
                    uncertainty=uncertainty_lbl,
                    class_names=class_names,
                    probabilities=results["mean"],
                    patient_meta=patient_meta,
                )
                if pdf_data:
                    st.download_button(
                        label="📥 Download Diagnostic PDF Report",
                        data=pdf_data,
                        file_name=f"clinical_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("Report compiled successfully!")
                else:
                    st.error("Report generation failed. Check reportlab installation.")

    render_regulatory_notices()


if __name__ == "__main__":
    main()
