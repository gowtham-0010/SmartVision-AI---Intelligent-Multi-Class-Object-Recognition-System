import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

st.title("🧠 Image Classification")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLASS_MODEL_PATH = ROOT_DIR / "yolov8n-cls.pt"

@st.cache_resource
def load_model():
    if CLASS_MODEL_PATH.exists():
        try:
            return YOLO(str(CLASS_MODEL_PATH))
        except Exception as e:
            st.error(f"⚠️ Classification model error: {e}")
            return None
    st.error(
        "❌ Classification model `yolov8n-cls.pt` not found locally. "
        "Add the local weights file to the repo or deploy with the file present."
    )
    return None

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"⚠️ Could not read the uploaded image: {e}")
    else:
        st.image(image, caption="Uploaded Image", width="stretch")

        img_array = np.array(image)

        with st.spinner("Running classification..."):
            results = model(img_array, verbose=False)

        probs = results[0].probs

        st.subheader("Prediction")

        top1 = probs.top1
        confidence = probs.top1conf

        label = model.names.get(int(top1), str(top1))
        st.success(f"{label} ({float(confidence):.2f})")