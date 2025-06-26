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

def build_pdf(lines: list, model_name: str) -> bytes:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER
    y = h - inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Soybean SDS Classification Report")
    y -= 0.3 * inch
    c.setFont("Helvetica", 10)
    c.drawString(inch, y, f"Model: {model_name}")
    y -= 0.2 * inch
    c.drawString(inch, y, f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M}")
    y -= 0.4 * inch
    c.setFont("Helvetica", 11)
    for line in lines:
        if y < inch:
            c.showPage()
            y = h - inch
            c.setFont("Helvetica", 11)
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    c.save()
    buf.seek(0)
    return buf.read()

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

    # Build & offer PDF
    pdf_bytes = build_pdf(pdf_lines, chosen_model)
    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name="SDS_leaf_report.pdf",
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
