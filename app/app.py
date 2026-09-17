"""
Interactive Clinical Web Application for TumorGANet.
Provides:
- Real-time Brain MRI Classification (Glioma, Meningioma, Pituitary, No Tumor)
- Explainable AI (LIME) visualization: Correct Region, Highlight Contour, Heatmap (Fig. 11)
- Interactive DCGAN Synthetic Brain MRI Generator
- Published Benchmark & Ablation Study Exploration (Tables 5, 6, 7 & Figs 8, 9, 10)
- Preprocessing Lab (Equations 2-7)
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import create_synthetic_mri_slice
from data.preprocessor import BrainMRIPreprocessor
from models.tumor_ganet import build_tumorganet
from models.gan import BrainMRIGAN
from evaluation.explainability import BrainMRILIMEExplainer
from training.ablation import get_published_ablation_df


st.set_page_config(
    page_title="TumorGANet | Clinical Brain Tumor Classifier & XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3F66;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = build_tumorganet(pretrained=False, device=device)
    # Check if trained weights exist
    best_weights = config.CHECKPOINTS_DIR / "tumorganet_best.pth"
    if best_weights.exists():
        classifier.load_state_dict(torch.load(best_weights, map_location=device))
    classifier.eval()

    gan = BrainMRIGAN(device=device)
    gan.load_models()
    return classifier, gan, device


classifier, gan, device = load_models()
explainer = BrainMRILIMEExplainer(model=classifier, device=torch.device(device))


# Sidebar Navigation & Controls
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=70)
st.sidebar.title("TumorGANet System")
st.sidebar.markdown(f"**Backend Device**: `{device.upper()}`")
st.sidebar.markdown("**Base Paper**: *IEEE Access 2024*")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Sample MRI")
sample_choice = st.sidebar.selectbox(
    "Load representative MRI slice:",
    ["None", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor (Healthy)"]
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Clinical Diagnosis",
    "🔍 Explainable AI (LIME)",
    "🎨 GAN Augmentation Studio",
    "📊 Benchmarks & Ablation",
    "🔬 Preprocessing Lab"
])


# -------------------------------------------------------------
# TAB 1: Clinical Diagnosis
# -------------------------------------------------------------
with tab1:
    st.markdown('<div class="main-header">Brain Tumor MRI Diagnostic Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Assisted Brain MRI Analysis using Transfer Learning (VGG-16 Backbone)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    input_img = None

    with col1:
        st.subheader("1. Upload or Select MRI Scan")
        uploaded_file = st.file_uploader("Upload Brain MRI (JPEG, PNG, DICOM-exported):", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            input_img = Image.open(uploaded_file).convert("RGB")
        elif sample_choice != "None":
            type_map = {
                "Glioma Tumor": "glioma",
                "Meningioma Tumor": "meningioma",
                "Pituitary Tumor": "pituitary",
                "No Tumor (Healthy)": "notumor"
            }
            input_img = create_synthetic_mri_slice(type_map[sample_choice])

        if input_img is not None:
            st.image(input_img, caption="Input MRI Slice (Axial View)", width=320)
        else:
            st.info("Please upload an MRI image or select a sample scan from the sidebar.")

    with col2:
        st.subheader("2. TumorGANet Diagnostic Assessment")
        if input_img is not None:
            with st.spinner("Analyzing MRI features through TumorGANet deep layers..."):
                results = classifier.predict_image(input_img)

            pclass = results["predicted_class"].capitalize()
            conf = results["confidence"] * 100.0

            # Banner
            if results["predicted_class"] == "notumor":
                st.success(f"### Result: **{pclass} (Healthy Brain)**")
            else:
                st.error(f"### Result: **{pclass} Detected**")

            st.metric("Primary Confidence", f"{conf:.2f}%")

            st.write("#### Class Probabilities Breakdown:")
            for cname, prob in results["probabilities"].items():
                st.write(f"**{cname.capitalize()}**: `{prob * 100.0:.2f}%`")
                st.progress(float(prob))

            st.markdown("""
            > **Clinical Note**: TumorGANet utilizes GAN-augmented feature diversity and VGG-16 deep feature representation, achieving **99.53% benchmark accuracy** and a **0.2% Hamming Loss**.
            """)


# -------------------------------------------------------------
# TAB 2: Explainable AI (LIME)
# -------------------------------------------------------------
with tab2:
    st.markdown('<div class="main-header">Explainable AI (LIME) Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Replication of Figure 11 & Section V-G: Local Interpretable Model-Agnostic Explanations</div>', unsafe_allow_html=True)

    if input_img is not None:
        if st.button("Generate LIME Interpretability Maps"):
            with st.spinner("Calculating super-pixel importance weights and local decision boundaries..."):
                explanation = explainer.explain_image(input_img, num_samples=150)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(explanation["original_image"], caption="1. Original Image", use_column_width=True)
                with c2:
                    st.image(explanation["correct_region"], caption="2. Correct Region (Masked)", use_column_width=True)
                with c3:
                    st.image(explanation["highlight_contour"], caption="3. Highlight Contour (Yellow)", use_column_width=True)
                with c4:
                    st.image(explanation["heatmap"], caption="4. Attribution Heatmap", use_column_width=True)

                st.info("""
                **Interpretability Guide**:
                - **Correct Region**: Highlights the specific super-pixels corresponding to the tumor mass.
                - **Highlight Contour**: Highlights the boundary demarcation separating tumor tissue from healthy cerebral cortex.
                - **Attribution Heatmap**: Green super-pixels indicate high probability contribution towards the predicted class; red indicates negative influence.
                """)
    else:
        st.warning("Please upload or select an MRI scan in Tab 1 to run LIME analysis.")


# -------------------------------------------------------------
# TAB 3: GAN Augmentation Studio
# -------------------------------------------------------------
with tab3:
    st.markdown('<div class="main-header">Generative Adversarial Network (DCGAN) Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Synthesis of Diverse Brain MRI Scans for Data Augmentation (Section V-B)</div>', unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([1, 2])

    with gcol1:
        st.subheader("Generator Controls")
        num_gen = st.slider("Number of synthetic scans to generate:", min_value=1, max_value=8, value=4)
        noise_seed = st.number_input("Random Latent Seed:", min_value=0, max_value=99999, value=42)

        if st.button("Generate Synthetic MRI Scans"):
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)
            with st.spinner("Sampling from GAN generator latent space..."):
                samples = gan.generate_synthetic_samples(num_samples=num_gen)
                st.session_state["synth_samples"] = samples

    with gcol2:
        st.subheader("Generated MRI Slices")
        if "synth_samples" in st.session_state:
            cols = st.columns(min(4, num_gen))
            for i, img_arr in enumerate(st.session_state["synth_samples"]):
                with cols[i % len(cols)]:
                    st.image(img_arr, caption=f"Synthetic Slice #{i+1}", use_column_width=True)
        else:
            st.info("Click 'Generate Synthetic MRI Scans' to produce novel brain MRI images.")


# -------------------------------------------------------------
# TAB 4: Paper Benchmarks & Ablation Study
# -------------------------------------------------------------
with tab4:
    st.markdown('<div class="main-header">Published Benchmark & Ablation Study Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation from IEEE Access 2024 paper (Tables 5, 6, 7 and Figure 9)</div>', unsafe_allow_html=True)

    b_subtab1, b_subtab2 = st.tabs(["🏆 10 Transfer Learning Models (Table 5)", "🔬 Optimizer Ablation Study (Table 7)"])

    with b_subtab1:
        st.markdown("### Performance Comparison across 10 Transfer Learning Architectures")
        table_5_data = [
            {"Model": "TumorGANet (Proposed VGG-16)", "Accuracy (%)": 99.53, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-Score (%)": 99.0, "Hamming Loss (%)": 0.20, "MCC (%)": 99.50, "JS (%)": 98.50, "CK (%)": 99.50},
            {"Model": "GANs + ResNet-101", "Accuracy (%)": 98.93, "Precision (%)": 98.93, "Recall (%)": 98.93, "F1-Score (%)": 99.0, "Hamming Loss (%)": 0.50, "MCC (%)": 99.10, "JS (%)": 97.90, "CK (%)": 98.70},
            {"Model": "GANs + Inception-v3", "Accuracy (%)": 99.34, "Precision (%)": 99.34, "Recall (%)": 99.34, "F1-Score (%)": 99.0, "Hamming Loss (%)": 1.02, "MCC (%)": 97.20, "JS (%)": 95.80, "CK (%)": 97.20},
            {"Model": "GANs + AlexNet", "Accuracy (%)": 98.17, "Precision (%)": 98.17, "Recall (%)": 98.17, "F1-Score (%)": 98.0, "Hamming Loss (%)": 0.90, "MCC (%)": 97.55, "JS (%)": 96.20, "CK (%)": 97.50},
            {"Model": "GANs + GoogLeNet", "Accuracy (%)": 97.18, "Precision (%)": 97.18, "Recall (%)": 97.18, "F1-Score (%)": 97.0, "Hamming Loss (%)": 2.44, "MCC (%)": 94.10, "JS (%)": 90.10, "CK (%)": 93.20},
            {"Model": "GANs + VGG-19", "Accuracy (%)": 96.72, "Precision (%)": 96.72, "Recall (%)": 96.72, "F1-Score (%)": 96.5, "Hamming Loss (%)": 1.60, "MCC (%)": 96.00, "JS (%)": 93.10, "CK (%)": 96.00},
            {"Model": "GANs + MobileNet", "Accuracy (%)": 96.41, "Precision (%)": 96.41, "Recall (%)": 96.41, "F1-score (%)": 96.2, "Hamming Loss (%)": 7.90, "MCC (%)": 95.00, "JS (%)": 93.00, "CK (%)": 95.20},
            {"Model": "GANs + MobileNet-v2", "Accuracy (%)": 95.85, "Precision (%)": 95.85, "Recall (%)": 95.85, "F1-score (%)": 95.5, "Hamming Loss (%)": 2.44, "MCC (%)": 94.10, "JS (%)": 90.20, "CK (%)": 93.10},
            {"Model": "GANs + DenseNet-121", "Accuracy (%)": 95.78, "Precision (%)": 95.78, "Recall (%)": 95.78, "F1-score (%)": 95.6, "Hamming Loss (%)": 2.47, "MCC (%)": 94.10, "JS (%)": 90.10, "CK (%)": 93.10},
            {"Model": "GANs + Xception", "Accuracy (%)": 78.01, "Precision (%)": 86.00, "Recall (%)": 77.00, "F1-score (%)": 76.0, "Hamming Loss (%)": 10.90, "MCC (%)": 64.30, "JS (%)": 64.30, "CK (%)": 70.70}
        ]
        st.dataframe(table_5_data, use_container_width=True)

    with b_subtab2:
        st.markdown("### Table 7: Full Ablation Study (5 Optimizers x 3 Learning Rates)")
        ablation_df = get_published_ablation_df()
        st.dataframe(ablation_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: Preprocessing Lab
# -------------------------------------------------------------
with tab5:
    st.markdown('<div class="main-header">MRI Preprocessing & Data Augmentation Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive implementation of Equations (2) through (7)</div>', unsafe_allow_html=True)

    if input_img is not None:
        raw_arr = np.array(input_img)
        preproc = BrainMRIPreprocessor(is_training=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(raw_arr, caption="Original MRI (Up to 512x512)", use_column_width=True)
        with c2:
            st.image(preproc.resize_image(raw_arr, (224, 224)), caption="Resized (224x224, Eq. 2-3)", use_column_width=True)
        with c3:
            st.image(preproc.random_zoom(raw_arr, zoom_range=0.2), caption="Zoomed (20% range, Eq. 6)", use_column_width=True)
        with c4:
            st.image(preproc.horizontal_flip_image(raw_arr), caption="Flipped (Horizontal, Eq. 7)", use_column_width=True)
    else:
        st.info("Select or upload an image in Tab 1 to test preprocessing transformations.")

