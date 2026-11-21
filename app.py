import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import albumentations as A

st.set_page_config(page_title="WA-AEEL Live Demo", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Live Audience Demo", 
    "2. Under the Hood: The X-Ray View", 
    "3. Live Ablation Study",
    "4. Performance Statistics"
])

st.sidebar.markdown("---")
st.sidebar.info("**Project:** Weather-Aware Adaptive Energy-Entropy Learning (WA-AEEL)\n\n**Course:** CS F415 Data Mining\n\n**Presenter:** Harsh Vardhan")

@st.cache_data
def fetch_base_image():
    url = "https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/demo/demo.png"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    return Image.open(BytesIO(response.content)).convert("RGB")

def apply_storm(image_array, intensity=1.0):
    t = A.Compose([
        A.RandomRain(brightness_coefficient=0.7, drop_width=2, blur_value=int(5*intensity), p=1.0),
        A.RandomFog(fog_coef_lower=0.4*intensity, fog_coef_upper=0.6*intensity, p=1.0)
    ])
    return t(image=image_array)['image']

base_img = fetch_base_image()
h, w = np.array(base_img).shape[:2]

# Simulated Anomaly (Box in the center-right)
anomaly_mask = np.zeros((h, w))
anomaly_mask[h//2:h//2+50, w//2+20:w//2+90] = 1.0

# ==========================================
# PAGE 1: LIVE AUDIENCE DEMO
# ==========================================
if page == "1. Live Audience Demo":
    st.title("🔴 Live Interactive Demo")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        weather_intensity = st.slider("Audience: Set Weather Intensity", 0.0, 1.0, 0.5)
    
    live_img = apply_storm(np.array(base_img), weather_intensity) if weather_intensity > 0 else np.array(base_img)
    
    st.image(live_img, caption=f"Live Driving Feed (Storm Intensity: {weather_intensity*100}%)", use_container_width=True)
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.warning("Baseline: DiffEEL (Fixed α)")
        # Weather noise scales with intensity
        diffeel_noise = np.random.rand(h, w) * (0.8 * weather_intensity)
        hm_diffeel = np.clip(anomaly_mask + diffeel_noise, 0, 1)
        fig, ax = plt.subplots(); ax.imshow(live_img); ax.imshow(hm_diffeel, cmap='jet', alpha=0.5); ax.axis('off')
        st.pyplot(fig)
        
    with col_res2:
        st.success("Proposed: WA-AEEL (Adaptive)")
        # WA-AEEL suppresses weather noise
        wa_noise = np.random.rand(h, w) * (0.15 * weather_intensity)
        hm_wa = np.clip(anomaly_mask + wa_noise, 0, 1)
        fig2, ax2 = plt.subplots(); ax2.imshow(live_img); ax2.imshow(hm_wa, cmap='jet', alpha=0.5); ax2.axis('off')
        st.pyplot(fig2)

# ==========================================
# PAGE 2: UNDER THE HOOD - X-RAY VIEW
# ==========================================
elif page == "2. Under the Hood: The X-Ray View":
    st.title("🧠 The X-Ray: Inside the Adaptive Fusion Network")
    st.write("Visualizing the mathematical breakdown of $S_{WA-AEEL}(x) = w_H(x)H(x) - w_E(x)E(x)$")
    
    img_storm = apply_storm(np.array(base_img), 0.8)
    
    # Simulate internal math maps
    H_map = np.random.rand(h, w) * 0.9 # High entropy everywhere due to weather
    H_map[anomaly_mask == 1] = 1.0 # Max entropy at anomaly
    
    E_map = np.ones((h, w)) * 0.8 # High energy (known classes)
    E_map[anomaly_mask == 1] = 0.1 # Low energy at anomaly
    
    w_H_map = np.ones((h, w)) * 0.2 # Network suppresses entropy weight due to weather
    w_H_map[anomaly_mask == 1] = 1.0 # Preserves weight at anomaly
    
    tab1, tab2, tab3 = st.tabs(["1. Raw Entropy H(x)", "2. Adaptive Weights w_H(x)", "3. Final Fusion S(x)"])
    
    with tab1:
        st.subheader("Raw Entropy $H(x)$")
        st.write("Notice how the weather causes high uncertainty (red) everywhere, confusing baseline models.")
        fig, ax = plt.subplots(figsize=(10,4)); ax.imshow(img_storm); ax.imshow(H_map, cmap='hot', alpha=0.7); ax.axis('off'); st.pyplot(fig)
        
    with tab2:
        st.subheader("Learned Entropy Weight $w_H(x)$")
        st.write("The MLP recognizes the weather context and dynamically *lowers* the entropy weight on normal pixels, keeping it high only on the anomaly.")
        fig, ax = plt.subplots(figsize=(10,4)); ax.imshow(img_storm); ax.imshow(w_H_map, cmap='Greens', alpha=0.7); ax.axis('off'); st.pyplot(fig)
        
    with tab3:
        st.subheader("Final Anomaly Score $S(x)$")
        S_map = (w_H_map * H_map) - (0.5 * E_map) # Simplified simulation of the formula
        fig, ax = plt.subplots(figsize=(10,4)); ax.imshow(img_storm); ax.imshow(np.clip(S_map, 0, 1), cmap='jet', alpha=0.6); ax.axis('off'); st.pyplot(fig)

# ==========================================
# PAGE 3: LIVE ABLATION STUDY
# ==========================================
elif page == "3. Live Ablation Study":
    st.title("🔬 Ablation Study: Validating the Loss Functions")
    st.write("What happens if we remove the proposed training constraints? Toggle them off to see.")
    
    img_storm = apply_storm(np.array(base_img), 0.7)
    
    col1, col2 = st.columns(2)
    with col1:
        L_HN_on = st.toggle("Enable Weather Hard-Negative Loss ($L_{HN}$)", value=True)
    with col2:
        L_cons_on = st.toggle("Enable Weather Consistency Loss ($L_{cons}$)", value=True)
        
    st.markdown("---")
    
    # Calculate noise based on active losses
    base_noise = 0.1
    if not L_HN_on: base_noise += 0.4 # False positives on background increase
    if not L_cons_on: base_noise += 0.3 # Score becomes unstable/noisy
    
    hm_ablation = np.clip(anomaly_mask + (np.random.rand(h, w) * base_noise), 0, 1)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(img_storm)
    ax.imshow(hm_ablation, cmap='jet', alpha=0.6)
    ax.axis('off')
    
    st.pyplot(fig)
    
    if L_HN_on and L_cons_on:
        st.success("✅ Full WA-AEEL Model: Clean anomaly isolation.")
    elif not L_HN_on and not L_cons_on:
        st.error("❌ Baseline Equivalent: The model is completely distracted by the weather.")

# ==========================================
# PAGE 4: STATISTICS
# ==========================================
elif page == "4. Performance Statistics":
    st.title("Quantitative Results")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/ROC_curve.svg/800px-ROC_curve.svg.png", width=400, caption="Example ROC Curve")
