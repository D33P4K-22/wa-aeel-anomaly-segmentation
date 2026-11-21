import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="WA-AEEL vs DiffEEL Dashboard", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Comparison", "Performance Statistics"])

st.sidebar.markdown("---")
st.sidebar.info("**Project:** Weather-Aware Adaptive Energy-Entropy Learning (WA-AEEL)\n\n**Course:** CS F415 Data Mining")

# --- HELPER FUNCTIONS (Placeholders for your PyTorch code) ---
def load_image(image_path):
    # Fallback to generating a dummy image if path doesn't exist for the template
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return Image.fromarray(np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8))

def get_dummy_heatmap(image_array, mode="diffeel"):
    """Simulates the model output. Replace this with your actual PyTorch inference."""
    h, w = image_array.shape[:2]
    base_heatmap = np.zeros((h, w))
    
    # Simulate an anomaly in the center
    base_heatmap[h//2-30:h//2+30, w//2-40:w//2+40] = 1.0
    
    if mode == "diffeel":
        # Simulate weather-induced false positives for DiffEEL
        noise = np.random.rand(h, w) * 0.6
        return np.clip(base_heatmap + noise, 0, 1)
    else:
        # Simulate WA-AEEL suppressing the weather noise
        noise = np.random.rand(h, w) * 0.1
        return np.clip(base_heatmap + noise, 0, 1)

# --- PAGE 1: DATASET EXPLORER ---
if page == "Dataset Explorer":
    st.title("Dataset Explorer: Complex Weather Scenarios")
    st.write("Browse the dataset to see how adverse weather conditions affect normal driving scenes, leading to uncertainty in standard models.")
    
    weather_condition = st.selectbox("Select Weather Condition", ["Rain", "Fog", "Night", "Sandstorm", "Blizzard"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Condition (Clear)")
        # Replace with path to your normal dataset images
        img_normal = load_image(f"data/normal_sample.jpg")
        st.image(img_normal, use_container_width=True)
        
    with col2:
        st.subheader(f"Adverse Condition ({weather_condition})")
        # Replace with path to your weather hard-negatives or ComsAmy data
        img_weather = load_image(f"data/{weather_condition.lower()}_sample.jpg")
        st.image(img_weather, use_container_width=True)

# --- PAGE 2: MODEL COMPARISON ---
elif page == "Model Comparison":
    st.title("Segmentation Comparison: DiffEEL vs. WA-AEEL")
    st.write("Compare the open-set anomaly segmentation heatmaps side-by-side.")
    
    # Upload an image or select from a test set
    uploaded_file = st.file_uploader("Upload a test image (or use default)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
    else:
        img = load_image("data/test_sample.jpg")
        
    img_array = np.array(img)
    
    st.subheader("Input Image (Adverse Weather)")
    st.image(img, width=600)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline: DiffEEL")
        st.write("Fixed energy-entropy fusion ($\alpha H - E$). Shows false positives due to weather.")
        heatmap_diffeel = get_dummy_heatmap(img_array, mode="diffeel")
        
        fig, ax = plt.subplots()
        ax.imshow(img_array)
        ax.imshow(heatmap_diffeel, cmap='jet', alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Proposed: WA-AEEL")
        st.write("Adaptive fusion ($w_H H - w_E E$). Suppresses weather-induced uncertainty.")
        heatmap_wa = get_dummy_heatmap(img_array, mode="wa-aeel")
        
        fig2, ax2 = plt.subplots()
        ax2.imshow(img_array)
        ax2.imshow(heatmap_wa, cmap='jet', alpha=0.5)
        ax2.axis('off')
        st.pyplot(fig2)

# --- PAGE 3: PERFORMANCE STATISTICS ---
elif page == "Performance Statistics":
    st.title("Quantitative Results & Statistics")
    st.write("Comparison of False Positive Rates (FPR) and Area Under the ROC Curve (AUROC) across different weather conditions.")
    
    # Mock data based on your presentation hypothesis
    data = {
        "Condition": ["Rain", "Fog", "Night", "Sandstorm", "Blizzard"],
        "DiffEEL FPR (%)": [18.5, 22.1, 15.3, 25.4, 20.0],
        "WA-AEEL FPR (%)": [4.2, 5.8, 3.1, 7.5, 6.2],
        "DiffEEL AUROC": [0.82, 0.79, 0.85, 0.75, 0.80],
        "WA-AEEL AUROC": [0.94, 0.92, 0.95, 0.89, 0.93]
    }
    df = pd.DataFrame(data)
    
    st.subheader("False Positive Rate (Lower is Better)")
    st.bar_chart(df.set_index("Condition")[["DiffEEL FPR (%)", "WA-AEEL FPR (%)"]], height=400)
    
    st.subheader("AUROC (Higher is Better)")
    st.line_chart(df.set_index("Condition")[["DiffEEL AUROC", "WA-AEEL AUROC"]], height=400)
    
    st.subheader("Detailed Metrics Table")
    st.dataframe(df, use_container_width=True)
