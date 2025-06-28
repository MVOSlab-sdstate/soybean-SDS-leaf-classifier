# app.py ── Soybean SDS Leaf Classifier 
# ----------------------------------------------------------------
import streamlit as st
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib, gdown, tempfile, os, io, base64
from pathlib import Path
from datetime import datetime
import pytz

# PDF building
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ─────────────── Google-Drive FILE-ID MAPS ────────────────────────
# Replace the placeholder IDs with the real ones from your Drive links.
CNN_FILE_ID = "1u2YgYoOfxH34glArx4pLJm2yOUZjWk2u"           # cnn_fold_5.keras

MODEL_ID_MAP = {                                    # joblib classifiers
    "AdaBoost_fold_5.joblib"        : "1VSL_q8CKZGaX4_f5Hvt4u6aJAQWMm3OG",
    "Decision Tree_fold_5.joblib"   : "1PQCi_QasKDavdCEDU8ySwOISrGo0aaIi",
    "Gaussian Process_fold_5.joblib": "1PfMq7iqgpWV0p42dBXCDghhtKmVlCiTo",
    "Linear SVM_fold_5.joblib"      : "1x4_Q872CNiFZTUIDWnLI65j2-_2umPyB",
    "Naive Bayes_fold_5.joblib"     : "1RU9wfiWkZJMiz0TUI0UfePiBX1wm3Ljx",
    "Nearest Neighbors_fold_5.joblib": "1id8X_q4s296HxYKEZAkqM4ZRJ37Apd4G",
    "Neural Net_fold_5.joblib"      : "1bhgm-10_1oJxI1juyWRTM1u_RD4mID_w",
    "QDA_fold_5.joblib"             : "13WyKc8vFBLpY0kf4og9KlsHYlpCRQZB3",
    "RBF SVM_fold_5.joblib"         : "1KYo9tQaXmTrKqqfUSSt6Ne26bj2OQAwy",
    "Random Forest_fold_5.joblib"   : "1qbfjSaCKSNHcKrJCuqtSjtK4KQ_HCBmj",
}

# ─────────────── constants & paths ────────────────────────────────
REPO_ROOT      = Path(__file__).parent
GA_BANDS       = [14, 25, 53, 72, 90]
RGB_IDX        = [91, 82, 53]
EXPECTED_SHAPE = (125, 100, 101)

LOGO_PATH      = REPO_ROOT / "images" / "sdsu_abe_logo.png"
WEB_LOGO_PATH  = "sdsu_logo.png"          # header logo for Streamlit page

TMP_DIR = Path(tempfile.gettempdir())     # cache downloads here

# ─────────────── helper utils ─────────────────────────────────────
def encode_b64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def download_from_drive(file_id: str, out_path: Path):
    """Download file from Google Drive only if not cached."""
    if out_path.exists():      # already cached
        return out_path
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out_path), quiet=False)
    return out_path

@st.cache_resource(show_spinner=False)
def load_cnn_extractor():
    cnn_path = download_from_drive(CNN_FILE_ID, TMP_DIR / "cnn_fold_5.keras")
    cnn      = load_model(str(cnn_path))
    return Sequential(cnn.layers[:-1])    # drop final softmax

@st.cache_resource(show_spinner=False)
def load_classifier(model_name: str):
    fid   = MODEL_ID_MAP.get(model_name)
    if fid is None:
        raise ValueError(f"No Drive file-ID for model {model_name}")
    local = download_from_drive(fid, TMP_DIR / model_name)
    return joblib.load(local)

# ─────────────── PDF generation ───────────────────────────────────
def build_pdf(results: list, model_name: str, tz_name: str) -> bytes:
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=LETTER)

    styles      = getSampleStyleSheet()
    file_style  = ParagraphStyle("file",  parent=styles["Heading2"])
    pred_style  = ParagraphStyle("pred",  parent=styles["Heading3"])
    small_style = ParagraphStyle("small", parent=styles["Normal"],
                                 alignment=TA_LEFT, fontSize=10)

    story = []

    # header (address + timestamp + logo)
    addr = ("Ag & Biosystems Engineering<br/>"
            "Raven Precision Ag Building&nbsp;104, Box&nbsp;2100<br/>"
            "Brookings, SD&nbsp;57007")

    ts   = datetime.now(pytz.timezone(tz_name)) \
            .strftime("%B&nbsp;%d,&nbsp;%Y&nbsp;|&nbsp;%I:%M&nbsp;%p")
    logo = RLImage(str(LOGO_PATH), width=2*inch, height=0.7*inch) \
           if LOGO_PATH.exists() else ""

    head_tbl = Table([[Paragraph(addr + "<br/>" + ts, small_style), logo]],
                     colWidths=[4.25*inch, 2*inch])
    head_tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story += [head_tbl, Spacer(1, 14),
              Paragraph("Soybean SDS Leaf Classification Report",
                        styles["Title"]),
              Spacer(1, 6),
              Paragraph(f"<b>Model Used:</b> {model_name}", small_style),
              Spacer(1, 18)]

    # per-file pages
    for i, rec in enumerate(results, 1):
        story += [Paragraph(f"{i}. File: {rec['filename']}", file_style),
                  Paragraph(f"Prediction: <b>{rec['prediction']}</b>",
                            pred_style),
                  Spacer(1, 10)]

        if rec.get("spectrum"):           # add spectral plot
            fig, ax = plt.subplots(figsize=(5,2))
            ax.plot(rec["spectrum"], lw=1.3)
            ax.set_xlabel("Band"); ax.set_ylabel("Reflectance")
            ax.set_title("Central-Pixel Spectral Signature")
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="PNG", dpi=160)
            plt.close(fig)
            img_buf.seek(0)
            story += [RLImage(img_buf, width=5*inch, height=2.1*inch),
                      Spacer(1, 16)]

        if i < len(results):
            story.append(PageBreak())

    # footer drawing fn
    def footer(canvas, doc):
        canvas.saveState()
        h = 40
        canvas.setFillColor(colors.HexColor("#00289c"))
        canvas.rect(0, 0, doc.pagesize[0], h, stroke=0, fill=1)
        canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(doc.pagesize[0]/2, 12,
                                 "Ag & Biosystems Engineering · SDSU")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf.getvalue()

# ─────────────── Streamlit UI ──────────────────────────────────────
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")

BACKGROUND_IMAGE_PATH = REPO_ROOT / "images" / "soybeanfield_homepage.jpg"
if BACKGROUND_IMAGE_PATH.exists():
    bg_image_encoded = encode_b64(BACKGROUND_IMAGE_PATH)
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_image_encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.warning("Background image not found. Please check path.")

header_logo = encode_b64(WEB_LOGO_PATH)
st.markdown(f"""
<div style="text-align:center">
  <img src="data:image/png;base64,{header_logo}" width="650"
       style="margin-bottom:65px"/>
  <h1 style="font-size:40px;font-weight:800">🌿Soybean SDS Leaf Classifier</h1>
</div>""", unsafe_allow_html=True)

# model & timezone selectors
model_list = sorted(MODEL_ID_MAP.keys())
model_name = st.selectbox("Choose a trained classifier:", model_list)

tz_choices = ["UTC","US/Eastern","US/Central","US/Mountain","US/Pacific",
              "US/Alaska","US/Hawaii"]
tz_name = st.selectbox("Report time-zone:", tz_choices,
                       index=tz_choices.index("US/Central"))

# cube uploader
mats = st.file_uploader("Upload soybean leaf hyperspectral cube(s) (.mat)",
                        type=["mat"], accept_multiple_files=True)

# ─────────────── Run classification ────────────────────────────────
if st.button("Classify Leaf(s)"):
    if not mats:
        st.error("Please add at least one .mat file."); st.stop()

    # load models (download if 1st time)
    try:
        cnn_extractor = load_cnn_extractor()
        clf_model     = load_classifier(model_name)
    except Exception as e:
        st.error(f"Model download/load failed: {e}")
        st.stop()

    res_list = []

    for up in mats:
        st.subheader(f"📁 {up.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(up.read()); tmp_path = tmp.name

        try:
            cube = scipy.io.loadmat(tmp_path)["R_Leaf"][:, :, 6:107]
        except Exception as exc:
            st.error(f"❌ {up.name}: {exc}"); os.remove(tmp_path); continue

        if cube.shape != EXPECTED_SHAPE:
            st.warning(f"{up.name}: expected {EXPECTED_SHAPE}, got {cube.shape}")
            os.remove(tmp_path); continue

        # show RGB preview
        rgb = cube[:, :, RGB_IDX]
        rgb = ((rgb - rgb.min()) / (np.ptp(rgb)+1e-6) * 255).astype(np.uint8)
        st.image(rgb, caption="RGB Visualization")

        # central pixel spectrum
        cpix = cube[cube.shape[0]//2, cube.shape[1]//2, :]
        fig, ax = plt.subplots(); ax.plot(range(6,107), cpix)
        ax.set_xlabel("Band"); ax.set_ylabel("Reflectance")
        ax.set_title("Central Pixel Spectral Profile")
        st.pyplot(fig)

        # prediction
        sel   = cube[:, :, GA_BANDS][None, ...]
        feats = cnn_extractor.predict(sel, verbose=0)
        pred  = "Healthy" if clf_model.predict(feats)[0] == 0 else "Infected (SDS)"
        st.success(f"✅ Prediction: **{pred}**")

        res_list.append({"filename": up.name,
                         "prediction": pred,
                         "spectrum": cpix.tolist()})

        os.remove(tmp_path)

    if res_list:
        pdf = build_pdf(res_list, model_name, tz_name)
        st.download_button("📄 Download PDF Report", pdf,
                           file_name="SDS_Classification_Report.pdf",
                           mime="application/pdf")

# footer
st.markdown("""
<hr style="margin-top:50px"/>
<div style="text-align:center;font-size:14px;color:gray">
© Machine Vision and Optical Sensor (MVOS) Lab, 2025. All rights reserved.
</div>""", unsafe_allow_html=True)
