# app.py ── Soybean SDS Leaf Classifier
# ----------------------------------------------------------------
import streamlit as st
import numpy as np, scipy.io, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib, gdown, tempfile, os, io, base64
from pathlib import Path
from datetime import datetime
import pytz

# =============  PDF imports (unchanged)  =========================
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units  import inch
from reportlab.lib.enums  import TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# =============  Google-Drive ID map (unchanged)  =================
CNN_FILE_ID = "1u2YgYoOfxH34glArx4pLJm2yOUZjWk2u"          # cnn_fold_5.keras
MODEL_ID_MAP = {
    "AdaBoost_fold_5.joblib"         : "1VSL_q8CKZGaX4_f5Hvt4u6aJAQWMm3OG",
    "Decision Tree_fold_5.joblib"    : "1PQCi_QasKDavdCEDU8ySwOISrGo0aaIi",
    "Gaussian Process_fold_5.joblib" : "1PfMq7iqgpWV0p42dBXCDghhtKmVlCiTo",
    "Linear SVM_fold_5.joblib"       : "1x4_Q872CNiFZTUIDWnLI65j2-_2umPyB",
    "Naive Bayes_fold_5.joblib"      : "1RU9wfiWkZJMiz0TUI0UfePiBX1wm3Ljx",
    "Nearest Neighbors_fold_5.joblib": "1id8X_q4s296HxYKEZAkqM4ZRJ37Apd4G",
    "Neural Net_fold_5.joblib"       : "1bhgm-10_1oJxI1juyWRTM1u_RD4mID_w",
    "QDA_fold_5.joblib"              : "13WyKc8vFBLpY0kf4og9KlsHYlpCRQZB3",
    "RBF SVM_fold_5.joblib"          : "1KYo9tQaXmTrKqqfUSSt6Ne26bj2OQAwy",
    "Random Forest_fold_5.joblib"    : "1qbfjSaCKSNHcKrJCuqtSjtK4KQ_HCBmj",
}

# =============  constants & paths  ===============================
REPO_ROOT      = Path(__file__).parent
GA_BANDS       = [14, 25, 53, 72, 90]
RGB_IDX        = [91, 82, 53]
EXPECTED_SHAPE = (125, 100, 101)

LOGO_PATH        = REPO_ROOT / "images" / "sdsu_abe_logo.png"
WEB_LOGO_PATH    = "sdsu_logo.png"
BACKGROUND_IMAGE = REPO_ROOT / "images" / "soybeanfield_homepage.jpg"

TMP_DIR = Path(tempfile.gettempdir())

# =============  helper utilities  ================================
def encode_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def download_from_drive(file_id: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dst),
                   quiet=False)
    return dst

@st.cache_resource(show_spinner=False)
def load_cnn_extractor():
    cnn_fp = download_from_drive(CNN_FILE_ID, TMP_DIR / "cnn_fold_5.keras")
    cnn    = load_model(str(cnn_fp))
    return Sequential(cnn.layers[:-1])

@st.cache_resource(show_spinner=False)
def load_classifier(model_name: str):
    fid = MODEL_ID_MAP[model_name]
    fp  = download_from_drive(fid, TMP_DIR / model_name)
    return joblib.load(fp)

# =============  PDF builder (unchanged)  =========================
def build_pdf(records: list, model_name: str, tz_name: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER)

    sty  = getSampleStyleSheet()
    fsty = ParagraphStyle("file",  parent=sty["Heading2"])
    psty = ParagraphStyle("pred",  parent=sty["Heading3"])
    ssty = ParagraphStyle("small", parent=sty["Normal"],
                          alignment=TA_LEFT, fontSize=10)

    # header
    addr = ("Ag & Biosystems Engineering<br/>"
            "Raven Precision Ag Building&nbsp;104, Box&nbsp;2100<br/>"
            "Brookings, SD&nbsp;57007")
    ts   = datetime.now(pytz.timezone(tz_name)).strftime(
           "%B&nbsp;%d,&nbsp;%Y&nbsp;|&nbsp;%I:%M&nbsp;%p")
    logo = RLImage(str(LOGO_PATH), width=2*inch, height=0.7*inch) \
           if LOGO_PATH.exists() else ""

    head = Table([[Paragraph(addr+"<br/>"+ts, ssty), logo]],
                 colWidths=[4.25*inch, 2*inch])
    head.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))

    story = [head, Spacer(1, 14),
             Paragraph("Soybean SDS Leaf Classification Report", sty["Title"]),
             Spacer(1, 6),
             Paragraph(f"<b>Model Used:</b> {model_name}", ssty),
             Spacer(1, 18)]

    # one page per file
    for i, rec in enumerate(records, 1):
        story += [Paragraph(f"{i}. File: {rec['filename']}", fsty),
                  Paragraph(f"Prediction: <b>{rec['prediction']}</b>", psty),
                  Spacer(1, 10)]
        if rec.get("spectrum"):
            fig, ax = plt.subplots(figsize=(5,2))
            ax.plot(rec["spectrum"], lw=1.3)
            ax.set_xlabel("Band"); ax.set_ylabel("Reflectance")
            ax.set_title("Central-Pixel Spectral Signature")
            plt.tight_layout()
            tmp = io.BytesIO(); fig.savefig(tmp, format="PNG", dpi=160)
            plt.close(fig); tmp.seek(0)
            story += [RLImage(tmp, width=5*inch, height=2.1*inch),
                      Spacer(1, 16)]
        if i < len(records):
            story.append(PageBreak())

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
    buf.seek(0)
    return buf.getvalue()

# =============  Streamlit UI  ====================================
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")

# ---- background + styling ----
if BACKGROUND_IMAGE.exists():
    bg64 = encode_b64(BACKGROUND_IMAGE)
    st.markdown(
        f"""
        <style>
        /* Background with dark overlay */
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                        url("data:image/jpeg;base64,{bg64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Headlines white */
        h1, h2, h3, h4, h5, h6 {{
            color:#ffffff !important;
        }}
        /* Widget “glass cards” */
        .stSelectbox, .stFileUploader, .stTextInput {{
            background: rgba(255,255,255,0.92) !important;
            border-radius: 8px;
        }}
        /* → labels inside those widgets BLACK */
        .stSelectbox label, .stFileUploader label, .stTextInput label {{
            color:#000 !important;
        }}
        /* Classify button: white bg, black text */
        .stButton>button {{
            background: rgba(255,255,255,0.93) !important;
            border-radius:8px;
            color:#000 !important;
            font-weight:600;
        }}
        /* Download button: SDSU blue */
        .stDownloadButton>button {{
            background:#00289c !important;
            color:#fff !important;
            border-radius:8px;
            font-weight:600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- Header ----
header_logo = encode_b64(WEB_LOGO_PATH)
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{header_logo}" width="650"
             style="margin-bottom:65px"/>
        <h1 style="font-size:40px;font-weight:800">🌿 Soybean SDS Leaf Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- model & timezone selectors ----
model_name = st.selectbox("Choose a trained classifier:", sorted(MODEL_ID_MAP))
tz_name    = st.selectbox(
    "Report time-zone:",
    ["UTC","US/Eastern","US/Central","US/Mountain","US/Pacific",
     "US/Alaska","US/Hawaii"],
    index=2,
)

# ---- cube uploader ----
mats = st.file_uploader(
    "Upload soybean leaf hyperspectral cube(s) (.mat)",
    type=["mat"], accept_multiple_files=True
)

# =============  Classification ===================================
if st.button("Classify Leaf(s)"):
    if not mats:
        st.error("Please add at least one .mat file."); st.stop()

    try:
        cnn_extr = load_cnn_extractor()
        clf      = load_classifier(model_name)
    except Exception as e:
        st.error(f"Model download/load failed: {e}"); st.stop()

    records = []

    for up in mats:
        st.subheader(f"📁 {up.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(up.read()); tmp_path = tmp.name

        try:
            cube = scipy.io.loadmat(tmp_path)["R_Leaf"][:, :, 6:107]
        except Exception as err:
            st.error(f"❌ {up.name}: {err}"); os.remove(tmp_path); continue

        if cube.shape != EXPECTED_SHAPE:
            st.warning(f"{up.name}: expected {EXPECTED_SHAPE}, got {cube.shape}")
            os.remove(tmp_path); continue

        # RGB preview
        rgb = cube[:, :, RGB_IDX]
        rgb = ((rgb - rgb.min()) / (np.ptp(rgb)+1e-6)*255).astype(np.uint8)
        st.image(rgb, caption=None)
        st.markdown('<div style="color:white; font-weight:500;">RGB Visualization</div>', unsafe_allow_html=True)

        # central-pixel spectrum
        cpix = cube[cube.shape[0]//2, cube.shape[1]//2, :]
        fig, ax = plt.subplots(); ax.plot(range(6,107), cpix)
        ax.set_xlabel("Band"); ax.set_ylabel("Reflectance")
        ax.set_title("Central Pixel Spectral Profile"); st.pyplot(fig)

        # prediction
        feat  = cnn_extr.predict(cube[:, :, GA_BANDS][None, ...], verbose=0)
        label = "Healthy" if clf.predict(feat)[0] == 0 else "Infected (SDS)"
        st.markdown(f'<div style="color:white; font-size:20px; font-weight:bold;">✅ Prediction: {label}</div>',unsafe_allow_html=True)
        
        records.append({"filename": up.name,
                        "prediction": label,
                        "spectrum" : cpix.tolist()})
        os.remove(tmp_path)

    # PDF download
    if records:
        pdf = build_pdf(records, model_name, tz_name)
        st.download_button("📄 Download PDF Report", pdf,
                           file_name="SDS_Classification_Report.pdf",
                           mime="application/pdf")

# ---- footer ----
st.markdown(
    """
    <hr style="margin-top:50px"/>
    <div style="text-align:center;font-size:14px;color:rgba(255,255,255,0.7)">
        © Machine Vision and Optical Sensor (MVOS) Lab, 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
