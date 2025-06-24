# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 16:43:47 2025
@author: Pappu.Yadav
"""

import streamlit as st
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
import joblib
import tempfile
import os
import datetime
import base64

# Function to encode image to base64 for inline HTML rendering
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Streamlit layout config
st.set_page_config(page_title="🌿 SDS Leaf Classifier", layout="centered")

# Centered logo and title using HTML
logo_path = "sdsu_logo.png"  # Update path if needed
logo_base64 = encode_image_to_base64(logo_path)

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="550" style="margin-bottom: 55px;" />
        <h1 style="font-size: 40px; font-weight: 800;">🌿Soybean SDS Leaf Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Band indices
ga_band_indices = [14, 25, 53, 72, 90]    # 5 selected bands
rgb_indices = [91, 82, 53]                # for RGB visualization

# Uploaders
uploaded_mat_files = st.file_uploader("Upload one or more Leaf Hypercubes (.mat)", type=["mat"], accept_multiple_files=True)
uploaded_model_file = st.file_uploader("Upload Trained Classifier (.joblib)", type=["joblib"])
uploaded_cnn_file = st.file_uploader("Upload CNN Feature Extractor (.keras)", type=["keras"])

# Classify
if st.button("Classify Leaf(s)"):
    if not uploaded_mat_files or not uploaded_model_file or not uploaded_cnn_file:
        st.error("Please upload all required files.")
        st.stop()

    # Load CNN and classifier model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_cnn:
        tmp_cnn.write(uploaded_cnn_file.read())
        cnn_model = load_model(tmp_cnn.name)
        feature_extractor = Sequential(cnn_model.layers[:-1])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_model:
        tmp_model.write(uploaded_model_file.read())
        classifier = joblib.load(tmp_model.name)

    report_lines = []
    for file in uploaded_mat_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp_mat:
                tmp_mat.write(file.read())
                mat = scipy.io.loadmat(tmp_mat.name)
                cube = mat['R_Leaf'][:, :, 6:107]  # shape: (125, 100, 101)

            st.subheader(f"📁 File: {file.name}")

            if cube.shape != (125, 100, 101):
                st.warning(f"Skipping {file.name}: Incorrect shape {cube.shape}")
                continue

            # RGB display
            rgb = cube[:, :, rgb_indices]
            rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255).astype(np.uint8)
            st.image(rgb, caption="RGB Visualization")

            # Central pixel spectral profile
            central_pixel = cube[cube.shape[0] // 2, cube.shape[1] // 2, :]
            fig, ax = plt.subplots()
            ax.plot(range(6, 107), central_pixel)
            ax.set_title("Central Pixel Spectral Profile")
            ax.set_xlabel("Band Index")
            ax.set_ylabel("Reflectance")
            st.pyplot(fig)

            # Classification
            selected = cube[:, :, ga_band_indices]
            selected = np.expand_dims(selected, axis=0)
            features = feature_extractor.predict(selected)
            pred = classifier.predict(features)[0]
            class_name = "Healthy" if pred == 0 else "Infected (SDS)"
            st.success(f"✅ Prediction: **{class_name}**")

            report_lines.append(f"{file.name}: {class_name}")

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")

    # Classification Summary Report
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_text = "\n".join(report_lines)
    st.text_area("📄 Classification Summary", report_text, height=150)
    report_path = os.path.join(tempfile.gettempdir(), f"SDS_Leaf_Report_{now}.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    with open(report_path, "rb") as f:
        st.download_button("📥 Download Report", f, file_name="SDS_Leaf_Report.txt")

# Local network usage tip
st.info("To allow access from other devices:\nRun with:\n`streamlit run yourscript.py --server.enableCORS false --server.address 0.0.0.0`")

# Footer - centered copyright message
st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style="text-align: center; font-size: 14px; color: gray;">
        © Machine Vision and Optical Sensor (MVOS) Lab, 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
