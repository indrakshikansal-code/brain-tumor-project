import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan to classify it as **Glioma, Meningioma, Pituitary, or No Tumor**.")

MODEL_PATH = "trained_model_finetuned.keras"
DRIVE_FILE_ID = "15fuofDBpEt4gB8tUvZi4WwTN5ou9e06R"
IMG_SIZE = 150
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_trained_model()

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    with st.spinner("Analyzing scan..."):
        prediction = model.predict(img_input)[0]

    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Result")
    if predicted_class == "notumor":
        st.success(f"✅ No Tumor Detected — Confidence: {confidence:.2f}%")
    else:
        st.error(f"⚠️ {predicted_class.capitalize()} Detected — Confidence: {confidence:.2f}%")

    st.subheader("Confidence Breakdown")
    for cls, prob in zip(CLASSES, prediction):
        st.write(f"**{cls.capitalize()}**: {prob*100:.2f}%")
        st.progress(float(prob))

    st.caption("⚠️ This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis.")
