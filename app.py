# app.py  – Soybean SDS Leaf Classifier (dropdown + PDF report)
import streamlit as st
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib, tempfile, os, io, datetime
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import base64

# ─────────────── constants ──────────────────────────────────────────────────
MODELS_DIR      = Path("models")
CNN_PATH        = MODELS_DIR / "cnn_feature_extractor.keras"   # change if needed
GA_BANDS        = [14, 25, 53, 72, 90]     # 5 GA-selected bands
RGB_IDX         = [91, 82, 53]             # pseudo-RGB for preview
EXPECTED_SHAPE  = (125, 100, 101)

# ─────────────── helpers ────────────────────────────────────────────────────
def encode_image_to_base64(img_path:str)->str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource(show_spinner=False)
def load_cnn_extractor():
    cnn = load_model(CNN_PATH)
    return Sequential(cnn.layers[:-1])

@st.cache_resource(show_spinner=False)
def load_classifier(model_name:str):
    return joblib.load(MODELS_DIR / model_name)

def generate_pdf(report:list, model_name:str)->bytes:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=LETTER)
    w,h = LETTER
    y   = h - inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Soybean SDS Classification Report")
    y -= 0.3*inch
    c.setFont("Helvetica", 10)
    c.drawString(inch, y, f"Model: {model_name}")
    y -= 0.2*inch
    c.drawString(inch, y, f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M}")
    y -= 0.4*inch
    c.setFont("Helvetica", 11)
    for line in report:
        if y < inch:            # new page if needed
            c.showPage(); y = h - inch
            c.setFont("Helvetica", 11)
        c.drawString(inch, y, line)
        y -= 0.25*inch
    c.save()
    buf.seek(0)
    return buf.read()

# ─────────────── UI - header ────────────────────────────────────────────────
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")
logo_b64 = encode_image_to_base64("sdsu_logo.png")   # adjust path if needed
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{logo_b64}" width="650" style="margin-bottom:65px"/>
        <h1 style="font-size:40px;font-weight:800">🌿Soybean SDS Leaf Classifier</h1>
    </div>""",
    unsafe_allow_html=True,
)

# ─────────────── model dropdown ─────────────────────────────────────────────
joblib_files = sorted(p.name for p in MODELS_DIR.glob("*.joblib"))
if not joblib_files:
    st.error("No .joblib files found in models directory!")
    st.stop()

model_choice = st.selectbox("Select a trained classifier model:", joblib_files)
classifier   = load_classifier(model_choice)
cnn_extractor = load_cnn_extractor()

# ─────────────── uploader ───────────────────────────────────────────────────
uploaded_mats = st.file_uploader(
    "Upload one or more soybean leaf hyperspectral cubes (.mat)",
    type=["mat"], accept_multiple_files=True
)

# ─────────────── classify button ────────────────────────────────────────────
if st.button("Classify Leaf(s)"):
    if not uploaded_mats:
        st.error("Please upload at least one .mat file.")
        st.stop()

    pdf_lines = []
    for up in uploaded_mats:
        st.subheader(f"📁 File: {up.name}")

        # save & load
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(up.read()); tmp_path = tmp.name
        mat  = scipy.io.loadmat(tmp_path)
        cube = mat["R_Leaf"][:, :, 6:107]

        if cube.shape != EXPECTED_SHAPE:
            st.warning(f"Skipped (shape {cube.shape})")
            os.remove(tmp_path); continue

        # RGB preview
        rgb = cube[:, :, RGB_IDX]
        rgb = ((rgb - rgb.min()) / rgb.ptp() * 255).astype(np.uint8)
        st.image(rgb, caption="RGB Visualization")

        # spectral profile (central pixel)
        cpix = cube[cube.shape[0]//2, cube.shape[1]//2, :]
        fig, ax = plt.subplots()
        ax.plot(range(6, 107), cpix)
        ax.set_title("Central Pixel Spectral Profile")
        ax.set_xlabel("Band Index"); ax.set_ylabel("Reflectance")
        st.pyplot(fig)

        # predict
        sel   = cube[:, :, GA_BANDS][None, ...]
        feats = cnn_extractor.predict(sel, verbose=0)
        label = "Healthy" if classifier.predict(feats)[0]==0 else "Infected (SDS)"
        st.success(f"✅ Prediction: **{label}**")

        pdf_lines.append(f"{up.name}: {label}")
        os.remove(tmp_path)   # cleanup

    # ───────── generate & offer PDF ────────
    pdf_bytes = generate_pdf(pdf_lines, model_choice)
    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name="SDS_leaf_report.pdf",
        mime="application/pdf"
    )

# ─────────────── footer ─────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:50px"/>
    <div style="text-align:center;font-size:14px;color:gray">
        © Machine Vision and Optical Sensor (MVOS) Lab, 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
