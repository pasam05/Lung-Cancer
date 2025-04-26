import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="centered")

# Load model
model = load_model('best_folder.h5')

# Define class names
class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma', 'Normal']

# Styling
st.markdown(
    """
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
        color: #4CAF50;
    }
    .small-font {
        font-size:20px !important;
        color: #555;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-top: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px #aaa;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🫁 Lung Cancer Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Upload your Chest X-ray / CT Scan image to predict possible lung diseases.</p>', unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((350, 350))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display Prediction
    result = class_names[predicted_class]

    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    if result == 'Normal':
        st.success(f"🟢 **Result: {result}**")
    else:
        st.error(f"🔴 **Result: {result}**")

    st.info(f"Prediction Confidence: **{confidence * 100:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)
