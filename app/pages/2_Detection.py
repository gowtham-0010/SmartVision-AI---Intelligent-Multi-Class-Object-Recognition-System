import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

st.title("🔍 Object Detection")

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "runs" / "detect" / "smartvision_yolo_fast" / "weights" / "last.pt"
FALLBACK_MODEL_PATH = ROOT_DIR / "yolov8n.pt"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            st.info("✓ Loaded custom trained YOLOv8 model")
            return YOLO(str(MODEL_PATH))
        except Exception as e:
            st.warning(f"⚠️ Custom model error: {e}")
    if FALLBACK_MODEL_PATH.exists():
        st.info("⚠️ Loaded local fallback YOLOv8 model")
        return YOLO(str(FALLBACK_MODEL_PATH))
    st.error(
        "❌ No local YOLO model was found. "
        "Upload `runs/detect/smartvision_yolo_fast/weights/last.pt` to the repo and redeploy."
    )
    return None

model = load_model()

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)

conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"⚠️ Could not read the uploaded image: {e}")
    else:
        col1.image(image, caption="Original Image", width="stretch")

        img_array = np.array(image)

        with st.spinner("Running detection..."):
            results = model.predict(img_array, conf=conf, verbose=False)

        result_img = results[0].plot()

        # Ultralytics `plot()` returns a BGR numpy image; tell Streamlit accordingly.
        col2.image(result_img, caption="Detected Image", width="stretch", channels="BGR")

        st.subheader("Detected Objects")
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                label = model.names.get(cls, str(cls))
                st.write(f"🔹 {label} ({conf_score:.2f})")
        else:
            st.write("No objects detected.")