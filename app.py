import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Page config (must be first Streamlit command)
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #ffccdd, #ffe6f0);
        color: #880e4f;
    }
    .stApp {
        background: linear-gradient(to bottom right, #ffccdd, #ffe6f0);
    }
    .big-font {
        font-size:30px !important;
        color: #ad1457;
        font-weight: bold;
    }
    .emotion {
        font-size:18px !important;
        color: #d81b60;
        font-style: italic;
        padding-bottom: 20px;
    }
    .input-label {
        font-size:18px !important;
        color: #880e4f;
        font-weight: bold;
    }
    .benign-result {
        font-size:24px !important;
        color: #2e7d32;
        font-weight: bold;
    }
    .malignant-result {
        font-size:24px !important;
        color: #c62828;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #fff0f5;
        color: #880e4f;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #ec407a;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.4em 1em;
    }
    .ribbon-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        overflow: hidden;
    }
    .ribbon {
        position: absolute;
        font-size: 48px;
        animation: float 12s infinite ease-in-out;
        color: #e91e63;
        opacity: 0.2;
    }
    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); }
        100% { transform: translateY(-120vh) rotate(360deg); }
    }
    </style>

    <div class="ribbon-bg">
        <div class="ribbon" style="left: 5%; animation-delay: 0s;">🎀</div>
        <div class="ribbon" style="left: 15%; animation-delay: 3s;">🎀</div>
        <div class="ribbon" style="left: 25%; animation-delay: 6s;">🎀</div>
        <div class="ribbon" style="left: 35%; animation-delay: 9s;">🎀</div>
        <div class="ribbon" style="left: 45%; animation-delay: 12s;">🎀</div>
        <div class="ribbon" style="left: 55%; animation-delay: 15s;">🎀</div>
        <div class="ribbon" style="left: 65%; animation-delay: 2s;">🎀</div>
        <div class="ribbon" style="left: 75%; animation-delay: 5s;">🎀</div>
        <div class="ribbon" style="left: 85%; animation-delay: 8s;">🎀</div>
        <div class="ribbon" style="left: 95%; animation-delay: 11s;">🎀</div>
    </div>
""", unsafe_allow_html=True)

# Load model and scaler
model = load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.save")

# Page content
image_path = "cancer_image.png"
st.image(image_path, use_container_width=True)

st.markdown('<p class="big-font">Breast Cancer Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="emotion">“Early detection saves lives — let us stand together in strength and hope.” 🎗</p>', unsafe_allow_html=True)

# Input section 1: Comma-separated
st.markdown('<p class="input-label">Please enter 30 comma-separated medical features below:</p>', unsafe_allow_html=True)
user_input = st.text_input("", 
    "11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563")

if st.button("Predict", type="primary"):
    try:
        input_data = np.array([float(i.strip()) for i in user_input.split(",")])
        if len(input_data) != 30:
            st.error("Please enter exactly 30 comma-separated values.")
        else:
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            label = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            if label == 1:
                st.markdown(f"""
                    <div style='background-color: #f8bbd0; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: #4caf50;'> Don't worry Tumor is Benign </h2>
                        <p style='font-size: 20px; color: #00695c; font-weight: bold;'>Confidence: {confidence:.2f}%</p>
                        <p style='color: #4e342e;'>You’re doing great! Keep taking care of your health 🌸</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #fce4ec; padding: 25px; border-radius: 12px; text-align: center; animation: popin 0.8s ease;'>
                        <h2 style='color: #c2185b;'>⚠️ Malignant Tumor Detected</h2>
                        <p style='font-size: 20px; color: #880e4f; font-weight: bold;'>Confidence: {confidence:.2f}%</p>
                        <p style='font-size:18px; color:#6a1b9a;'>You are not alone sending you strength to fight this. Talk to a doctor immediately 💗</p>
                        <div style='font-size: 50px; animation: hugPulse 2s infinite;'>🫂</div>
                    </div>

                    <style>
                    @keyframes hugPulse {{
                        0% {{ transform: scale(1); opacity: 0.9; }}
                        50% {{ transform: scale(1.15); opacity: 1; }}
                        100% {{ transform: scale(1); opacity: 0.9; }}
                    }}
                    @keyframes popin {{
                        0% {{ transform: scale(0.8); opacity: 0; }}
                        100% {{ transform: scale(1); opacity: 1; }}
                    }}
                    </style>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------- DETAILED ANALYSIS SECTION ----------------------

st.markdown('<hr style="margin-top: 50px; margin-bottom: 30px;">', unsafe_allow_html=True)
st.markdown('<h4 style="color: #880e4f;">🔬 Detailed Analysis: Enter each feature separately</h4>', unsafe_allow_html=True)

# Pre-fill test values
test_values = [
    22.270, 19.67, 152.80, 1509.0, 0.13260, 0.27680, 0.426400, 0.182300, 0.2556, 0.07039,
    1.2150, 1.5450, 10.050, 170.00, 0.006515, 0.086680, 0.104000, 0.024800, 0.03112, 0.005037,
    28.40, 28.01, 206.80, 2360.0, 0.1701, 0.6997, 0.96080, 0.29100, 0.4055, 0.09789
]

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

custom_input_css = """
<style>
    label[for^="feature_"] {
        color: #6a1b9a !important;
        font-weight: bold;
        font-size: 16px !important;
        text-transform: capitalize;
    }
    input[type="number"] {
        background-color: #ffffff !important;
        color: #c2185b !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_input_css, unsafe_allow_html=True)

detailed_inputs = []
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    display_label = feature[0].upper() + feature[1:]
    with cols[i % 3]:
        # Custom styled label
        st.markdown(f"<div style='color:#6a1b9a; font-weight:bold; font-size:16px;'>{display_label}</div>", unsafe_allow_html=True)
        value = st.number_input("", key=f"feature_{i}", value=float(test_values[i]), format="%.5f")
        detailed_inputs.append(value)


if st.button("Predict (Detailed)", type="secondary"):
    try:
        if len(detailed_inputs) != 30:
            st.error("All 30 features must be entered.")
        else:
            input_scaled = scaler.transform([detailed_inputs])
            prediction = model.predict(input_scaled)
            label = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            if label == 1:
                st.markdown(f"""
                    <div style='background-color: #f8bbd0; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: #4caf50;'> Don't worry Tumor is Benign </h2>
                        <p style='font-size: 20px; color: #00695c; font-weight: bold;'>Confidence: {confidence:.2f}%</p>
                        <p style='color: #4e342e;'>You’re doing great! Keep taking care of your health 🌸</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #fce4ec; padding: 25px; border-radius: 12px; text-align: center; animation: popin 0.8s ease;'>
                        <h2 style='color: #c2185b;'>⚠️ Malignant Tumor Detected</h2>
                        <p style='font-size: 20px; color: #880e4f; font-weight: bold;'>Confidence: {confidence:.2f}%</p>
                        <p style='font-size:18px; color:#6a1b9a;'>You are not alone sending you strength to fight this. Talk to a doctor immediately 💗</p>
                        <div style='font-size: 50px; animation: hugPulse 2s infinite;'>🫂</div>
                    </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")


# Footer
st.markdown("""
    <hr style="border: 1px solid #e91e63;">
    <div style="text-align: center; color: #880e4f; font-size: 14px; padding-top: 10px;">
        © All rights reserved by <strong>Dania 2025</strong>
    </div>
""", unsafe_allow_html=True)
