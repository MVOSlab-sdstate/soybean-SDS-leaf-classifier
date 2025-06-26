# app.py ── Soybean SDS Leaf Classifier (+ model-name & time-zone in PDF)
# ----------------------------------------------------------------------
import streamlit as st
import numpy as np, scipy.io, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib, tempfile, os, io, base64
from pathlib import Path
from datetime import datetime
import pytz                                              # ← NEW
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Table, TableStyle

# ─────────────── paths & constants ────────────────────────────────────
REPO_ROOT      = Path(__file__).parent
MODELS_DIR     = REPO_ROOT / "models"
CNN_PATH       = MODELS_DIR / "cnn_fold_5.keras"
GA_BANDS       = [14, 25, 53, 72, 90]
RGB_IDX        = [91, 82, 53]
EXPECTED_SHAPE = (125, 100, 101)
LOGO_PATH      = REPO_ROOT / "images" / "sdsu_abe_logo.png"   # make sure it exists
WEB_LOGO_PATH  = "sdsu_logo.png"                              # for Streamlit header

# ─────────────── helper functions ─────────────────────────────────────
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

def generate_pdf_report(results: list, model_name: str, tz_name: str) -> bytes:
    """Build a multi-page PDF report.  results =list of dicts."""
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=LETTER)

    styles         = getSampleStyleSheet()
    file_style     = ParagraphStyle('File', parent=styles['Heading2'])
    pred_style     = ParagraphStyle('Pred', parent=styles['Heading3'])
    subtitle_style = ParagraphStyle('Sub',  parent=styles['Normal'],
                                    alignment=TA_LEFT, fontSize=10)

    story = []

    # ---------- header: address, date-time, logo ----------
    address_html = ("Ag & Biosystems Engineering<br/>"
                    "Raven Precision Ag Building&nbsp;104, Box&nbsp;2100<br/>"
                    "Brookings, SD 57007")
    timestamp = datetime.now(pytz.timezone(tz_name))     # ← TZ aware
    now_html  = timestamp.strftime("%B&nbsp;%d,&nbsp;%Y&nbsp;|&nbsp;%I:%M&nbsp;%p")

    logo_flow = None
    if LOGO_PATH.exists():
        logo_flow = RLImage(str(LOGO_PATH), width=2.0*inch, height=0.7*inch)  # keep aspect

    header_table = Table(
        [[Paragraph(address_html + "<br/>" + now_html, subtitle_style),
          logo_flow]],
        colWidths=[4.2*inch, 2.3*inch]
    )
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.extend([header_table, Spacer(1, 14)])

    # Title and model used
    story.append(Paragraph("Soybean SDS Leaf Classification Report",
                           styles['Title']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Model Used:</b> {model_name}",
                           subtitle_style))
    story.append(Spacer(1, 18))

    # ---------- iterate uploaded cubes ----------
    for idx, rec in enumerate(results, start=1):
        story.append(Paragraph(f"{idx}. File: {rec['filename']}", file_style))
        story.append(Paragraph(f"Prediction: <b>{rec['prediction']}</b>",
                               pred_style))
        story.append(Spacer(1, 10))

        # spectral plot image
        if rec.get("spectrum"):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(rec['spectrum'], lw=1.3)
            ax.set_xlabel("Band"); ax.set_ylabel("Reflectance")
            ax.set_title("Central-Pixel Spectral Signature")
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="PNG", dpi=160)
            plt.close(fig)
            img_buf.seek(0)
            story.append(RLImage(img_buf, width=5*inch, height=2.1*inch))
            story.append(Spacer(1, 18))

        if idx < len(results):
            story.append(PageBreak())

    # ---------- footer ----------
    def footer(canvas, doc):
        canvas.saveState()
        h = 40
        canvas.setFillColor(colors.HexColor("#00289c"))
        canvas.rect(0, 0, doc.pagesize[0], h, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(doc.pagesize[0]/2, 12,
                                 "Ag & Biosystems Engineering · SDSU")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer.getvalue()

# ─────────────── Streamlit UI ─────────────────────────────────────────
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")

web_logo_b64 = encode_image_to_base64(WEB_LOGO_PATH)
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{web_logo_b64}"
             width="650" style="margin-bottom:65px"/>
        <h1 style="font-size:40px;font-weight:800">
            🌿Soybean SDS Leaf Classifier
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Select model
joblib_files = sorted(p.name for p in MODELS_DIR.glob("*.joblib"))
if not joblib_files:
    st.error("No .joblib models in `/models`.")
    st.stop()
chosen_model = st.selectbox("Select a trained classifier model:", joblib_files)

# Time-zone selector
tz_options = ["UTC", "US/Eastern", "US/Central", "US/Mountain",
              "US/Pacific", "Canada/Saskatchewan", "US/Alaska", "US/Hawaii"]
tz_name = st.selectbox("Select your time-zone for the report timestamp:",
                       tz_options, index=tz_options.index("US/Central"))

# Upload cubes
uploaded_mats = st.file_uploader(
    "Upload one or more soybean leaf hyperspectral cubes (.mat)",
    type=["mat"], accept_multiple_files=True
)

# ─────────────── Classify button ─────────────────────────────────────
if st.button("Classify Leaf(s)"):
    if not uploaded_mats:
        st.error("Please upload at least one .mat cube."); st.stop()
    if not CNN_PATH.exists():
        st.error(f"CNN model not found at {CNN_PATH}"); st.stop()

    try:
        cnn_extractor = load_cnn_extractor()
        classifier    = load_classifier(chosen_model)
    except Exception as e:
        st.error(f"Model load error: {e}"); st.stop()

    results = []

    for up in uploaded_mats:
        st.subheader(f"📁 {up.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name

        try:
            cube = scipy.io.loadmat(tmp_path)["R_Leaf"][:, :, 6:107]
        except Exception as e:
            st.error(f"Could not read {up.name}: {e}")
            os.remove(tmp_path); continue

        if cube.shape != EXPECTED_SHAPE:
            st.warning(f"{up.name}: expected {EXPECTED_SHAPE}, got {cube.shape}")
            os.remove(tmp_path); continue

        # RGB preview
        rgb = cube[:, :, RGB_IDX]
        rgb = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-6) * 255).astype(np.uint8)
        st.image(rgb, caption="RGB Visualization")

        # spectral profile
        cpix = cube[cube.shape[0]//2, cube.shape[1]//2, :]
        fig, ax = plt.subplots()
        ax.plot(range(6, 107), cpix); ax.set_xlabel("Band"); ax.set_ylabel("Reflect.")
        ax.set_title("Central Pixel Spectral Profile"); st.pyplot(fig)

        # Classification
        sel   = cube[:, :, GA_BANDS][None, ...]
        feats = cnn_extractor.predict(sel, verbose=0)
        label = "Healthy" if classifier.predict(feats)[0] == 0 else "Infected (SDS)"
        st.success(f"✅ Prediction: **{label}**")

        results.append({"filename": up.name,
                        "prediction": label,
                        "spectrum"  : cpix.tolist()})
        os.remove(tmp_path)

    if results:
        pdf_bytes = generate_pdf_report(results, chosen_model, tz_name)
        st.download_button(
            "📄 Download PDF Report",
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
