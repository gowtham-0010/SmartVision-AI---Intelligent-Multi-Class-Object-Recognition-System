"""
Page 3: Object Detection
Upload a photo with potentially many objects, run YOLO, adjust confidence
threshold live, and see bounding boxes + labels + scores.
"""

import os
import sys
from PIL import Image
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import load_yolo_model
from inference_pipeline import draw_detections

st.set_page_config(page_title="Object Detection - SmartVision AI", layout="wide")
st.title("Object detection")
st.write("Upload a photo. YOLOv8 will locate every object it recognises from the 25 trained classes.")

col_a, col_b = st.columns(2)
with col_a:
    confidence = st.slider("Confidence threshold", 0.05, 0.95, CONFIDENCE_THRESHOLD, 0.05)
with col_b:
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, NMS_IOU_THRESHOLD, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

yolo_model, weights_path = load_yolo_model()

if yolo_model is None:
    st.warning(
        "No trained YOLO weights found in `models/yolo_smartvision/weights/best.pt`. "
        "Run `training/train_yolo.py` first, then reload this page."
    )
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    import numpy as np
    results = yolo_model.predict(source=np.array(image), conf=confidence, iou=iou, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "box": [round(v, 1) for v in xyxy],
            "class_name": result.names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
        })

    annotated = draw_detections(image, detections)

    col_img, col_list = st.columns([2, 1])
    with col_img:
        st.image(annotated, caption=f"{len(detections)} object(s) detected", use_container_width=True)
    with col_list:
        st.markdown("### Detections")
        if detections:
            for det in sorted(detections, key=lambda d: -d["confidence"]):
                st.write(f"**{det['class_name']}** - {det['confidence']*100:.1f}%")
        else:
            st.write("No objects detected above this confidence threshold.")
else:
    st.info("Upload an image above to get started.")
