# app.py  ── Soybean SDS Leaf Classifier (dropdown + PDF report, patched)
# ---------------------------------------------------------------------
import streamlit as st
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib, tempfile, os, io, datetime, base64
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import io

# ─────────────── paths & constants ────────────────────────────────────
REPO_ROOT      = Path(__file__).parent
MODELS_DIR     = REPO_ROOT / "models"
CNN_PATH       = MODELS_DIR / "cnn_fold_5.keras"      # Adjust if different
GA_BANDS       = [14, 25, 53, 72, 90]
RGB_IDX        = [91, 82, 53]
EXPECTED_SHAPE = (125, 100, 101)

# ─────────────── helper functions ────────────────────────────────────
def encode_image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource(show_spinner=False)
def load_cnn_extractor():
    cnn = load_model(CNN_PATH)
    return Sequential(cnn.layers[:-1])

@st.cache_resource(show_spinner=False)
def load_classifier(model_name: str):
    return joblib.load(MODELS_DIR / model_name)


def generate_pdf_report(results: list, filename="SDS_Classification_Report.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=TA_LEFT, fontSize=10)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey)
    ])

    content = []

    # Address + Date Header
    address = "Ag & Biosystems Engineering\nRaven Precision Ag Building 104, Box 2100\nBrookings, SD 57007"
    now = datetime.now().strftime("%B %d, %Y | %I:%M %p")
    content.append(Paragraph(address, subtitle_style))
    content.append(Paragraph(now, subtitle_style))
    content.append(Spacer(1, 12))

    # Title
    content.append(Paragraph("Soybean SDS Leaf Classification Report", title_style))
    content.append(Spacer(1, 24))

    # Table of Results
    table_data = [['Filename', 'Predicted Class']]
    table_data += [[r['filename'], r['prediction']] for r in results]
    table = Table(table_data, colWidths=[3 * inch, 3 * inch])
    table.setStyle(table_style)
    content.append(table)
    content.append(Spacer(1, 24))

    # Spectral Plot (optional - from last file)
    if results and 'spectrum' in results[-1]:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.plot(results[-1]['spectrum'], linewidth=1.5)
        ax.set_title("Spectral Signature (Central Pixel)")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        img_buffer = io.BytesIO()
        plt.tight_layout()
        fig.savefig(img_buffer, format='PNG')
        plt.close(fig)
        img_buffer.seek(0)
        img = RLImage(img_buffer, width=5 * inch, height=2.2 * inch)
        content.append(img)

    # Footer via canvas
    def add_footer(canvas, doc):
        canvas.saveState()
        footer_height = 40
        canvas.setFillColor(colors.HexColor("#00289c"))  # SDSU blue
        canvas.rect(0, 0, doc.pagesize[0], footer_height, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(doc.pagesize[0] / 2, 12, "Ag & Biosystems Engineering · SDSU")
        canvas.restoreState()

    doc.build(content, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer

# ─────────────── Streamlit layout ─────────────────────────────────────
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")

logo_b64 = encode_image_to_base64("sdsu_logo.png")
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{logo_b64}" width="650" style="margin-bottom:65px"/>
        <h1 style="font-size:40px;font-weight:800">🌿Soybean SDS Leaf Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dropdown of available classifiers
joblib_files = sorted(p.name for p in MODELS_DIR.glob("*.joblib"))
if not joblib_files:
    st.error("No .joblib models found in the models directory.")
    st.stop()

chosen_model = st.selectbox("Select a trained classifier model:", joblib_files)

# Upload .mat cubes
uploaded_mats = st.file_uploader(
    "Upload one or more soybean leaf hyperspectral cubes (.mat)",
    type=["mat"],
    accept_multiple_files=True
)

# ─────────────── classify button ─────────────────────────────────────
if st.button("Classify Leaf(s)"):

    if not uploaded_mats:
        st.error("Please upload at least one .mat cube.")
        st.stop()

    if not CNN_PATH.exists():
        st.error(f"CNN model not found at {CNN_PATH}")
        st.stop()

    # cache-loaded models
    try:
        cnn_extractor = load_cnn_extractor()
        classifier    = load_classifier(chosen_model)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    pdf_lines = []

    for up in uploaded_mats:
        st.subheader(f"📁 File: {up.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name

        try:
            mat  = scipy.io.loadmat(tmp_path)
            cube = mat["R_Leaf"][:, :, 6:107]
        except Exception as e:
            st.error(f"Could not read {up.name}: {e}")
            os.remove(tmp_path)
            continue

        if cube.shape != EXPECTED_SHAPE:
            st.warning(f"Skipped (expected {EXPECTED_SHAPE}, got {cube.shape})")
            os.remove(tmp_path)
            continue

        # RGB preview (patched normalization)
        rgb = cube[:, :, RGB_IDX]
        rgb = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-6) * 255).astype(np.uint8)
        st.image(rgb, caption="RGB Visualization")

        # Spectral profile
        cpix = cube[cube.shape[0] // 2, cube.shape[1] // 2, :]
        fig, ax = plt.subplots()
        ax.plot(range(6, 107), cpix)
        ax.set_title("Central Pixel Spectral Profile")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Reflectance")
        st.pyplot(fig)

        # Classification
        sel   = cube[:, :, GA_BANDS][None, ...]
        feats = cnn_extractor.predict(sel, verbose=0)
        label = "Healthy" if classifier.predict(feats)[0] == 0 else "Infected (SDS)"
        st.success(f"✅ Prediction: **{label}**")

        pdf_lines.append(f"{up.name}: {label}")
        os.remove(tmp_path)

    pdf_bytes = generate_pdf_report(classification_results)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="SDS_Classification_Report.pdf",
        mime="application/pdf"
    )


# Footer
st.markdown(
    """
    <hr style="margin-top:50px"/>
    <div style="text-align:center;font-size:14px;color:gray">
        © Machine Vision and Optical Sensor (MVOS) Lab, 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
