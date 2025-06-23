# 🌱 Soybean SDS Leaf Classifier Web App

An AI-driven Streamlit web application for detecting Sudden Death Syndrome (SDS) in soybean leaves using hyperspectral imaging at early stages — even before visual symptoms appear.

---

## 📌 Background: What is Soybean SDS?

Sudden Death Syndrome (SDS) is one of the **most destructive diseases** of soybean in North America. Caused by the soil-borne fungus *Fusarium virguliforme*, SDS can lead to:

- Yield losses up to **50%** in severely infected fields.
- Early onset leaf chlorosis and necrosis.
- Root rot and plant death in severe cases.

🌾 **Soybean growers** struggle to detect SDS early since symptoms appear **after irreversible damage** has already occurred underground.

---

## 🧠 Role of Hyperspectral Imaging and AI

Hyperspectral imaging captures **reflectance data across hundreds of narrow spectral bands**. It can detect subtle biochemical and physiological changes in plants caused by SDS:

✅ **Before** visible symptoms appear on the leaf surface  
✅ Captures stress-related spectral signatures in infected plants  
✅ Enables **early intervention** and management

This technology, when combined with deep learning (CNN models), can be trained to distinguish **healthy vs. infected leaves** with high accuracy — even at early disease stages.

---

## 💻 About This Web App

This web app allows growers and researchers to:

- **Upload calibrated hyperspectral leaf images** (.hdr and .iso pairs)
- Visualize the hyperspectral cube
- Generate **spectral signatures** of the central pixel
- Classify leaf severity levels into 5 SDS stages (L1–L5)
- Receive predictions using trained CNN + ML classifiers

🧠 The app uses a trained **CNN-based feature extractor** and multiple classical classifiers (like Random Forest, SVM) to predict disease severity from early-stage hyperspectral cubes.

---

## 🌿 Why is This Useful for Growers?

| Problem | Solution |
|--------|----------|
| SDS symptoms appear too late for chemical treatment | Early detection via hyperspectral data |
| Need for non-destructive scouting | Contactless spectral imaging at the leaf level |
| Lack of expertise to analyze hyperspectral data | Simple upload-and-diagnose web interface |
| Limited access to lab software | Free, open-access web tool for SDS detection |

---

## 🖼 Example Images

### SDS-Infected Soybean Field  
![Soybean Field](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Soybean_field_with_SDS.jpg/800px-Soybean_field_with_SDS.jpg)

### Early Symptoms on Leaf  
![SDS Leaf Symptoms](https://www.researchgate.net/profile/Madeline-Lakatos/publication/349101840/figure/fig1/AS:990989046906882@1612479416561/Symptoms-of-sudden-death-syndrome-on-soybean-leaves-Image-credit-M-Lakatos-ISU.ppm)

> *(Note: Replace these with your actual dataset samples if needed.)*

---

## 🚀 Try the Web App

🌐 [Launch the App](https://sdsumvoslabsoybeansds.streamlit.app)

---

## 🧪 Requirements for Local Setup

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
├── app.py                   # Streamlit app
├── models/                  # Saved CNN and classifier models
├── utils/                   # Helper scripts for visualization and inference
├── data_samples/            # Example .hdr/.iso hyperspectral image pairs
├── .streamlit/
│   └── config.toml          # Theme and runtime settings
├── requirements.txt
└── README.md
```

---

## 👨‍🔬 Developed By

Machine Vision and Optical Sensor (MVOS) Lab  
Department of Agricultural and Biosystems Engineering  
South Dakota State University  
[Website](https://www.sdstate.edu/agricultural-biosystems-engineering) | [MVOS Lab GitHub](https://github.com/mvoslab)

---

## 📬 Contact

For questions, collaborations, or dataset access, please reach out to:  
📧 **Dr. Pappu K. Yadav** — pappu.yadav@sdstate.edu


