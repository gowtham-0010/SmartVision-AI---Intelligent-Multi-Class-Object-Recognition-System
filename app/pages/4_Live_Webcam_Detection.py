"""
Page 5: Live Webcam Detection (optional bonus feature, +3 points)
Uses Streamlit's built-in camera_input for a snapshot-based "live" demo -
works in any browser without extra dependencies. For true continuous
video, see the streamlit-webrtc note at the bottom of this file.
"""

import os
import sys
import time
from PIL import Image
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIDENCE_THRESHOLD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import load_yolo_model
from inference_pipeline import draw_detections

st.set_page_config(page_title="Live Webcam Detection - SmartVision AI", layout="wide")
st.title("Live webcam detection (optional)")
st.write(
    "Take a snapshot from your webcam and run object detection on it instantly. "
    "This uses Streamlit's built-in camera widget, so it works with zero extra setup."
)

yolo_model, weights_path = load_yolo_model()

if yolo_model is None:
    st.warning("No trained YOLO weights found. Run `training/train_yolo.py` first.")
else:
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        start = time.time()
        results = yolo_model.predict(source=np.array(image), conf=CONFIDENCE_THRESHOLD, verbose=False)
        elapsed_ms = (time.time() - start) * 1000
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
        st.image(annotated, caption=f"{len(detections)} object(s) - {elapsed_ms:.0f} ms", use_container_width=True)

st.divider()
st.caption(
    "Note: for continuous (non-snapshot) live video, install `streamlit-webrtc` "
    "and swap this page's capture logic for a VideoTransformer that runs "
    "`yolo_model.predict()` per frame. That path needs a GPU-backed host to "
    "hit real-time FPS."
)
