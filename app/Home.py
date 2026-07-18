"""
Home.py
Phase 5 - Streamlit Application Development (Page 1: Home)

This is the entry point Streamlit runs. Additional pages live in app/pages/
and are auto-discovered by Streamlit's multipage app convention.

Run from the smartvision_ai/ directory with:
    streamlit run app/Home.py
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SELECTED_CLASSES

st.set_page_config(
    page_title="SmartVision AI",
    page_icon=":material/visibility:",
    layout="wide",
)

st.title("SmartVision AI")
st.subheader("Intelligent multi-class object recognition system")

st.markdown(
    """
SmartVision AI combines **transfer-learning image classification** and
**YOLOv8 object detection** into one platform, trained on a curated
25-class subset of the COCO dataset.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Object classes", "25")
with col2:
    st.metric("Training images", "2,500")
with col3:
    st.metric("Models compared", "4 CNNs + YOLOv8")

st.divider()

st.markdown("### What you can do here")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Image Classification**")
    st.write("Upload a single-object image and see predictions from all 4 trained CNNs side by side.")
with c2:
    st.markdown("**Object Detection**")
    st.write("Upload a photo with multiple objects and get YOLO bounding boxes with adjustable confidence.")
with c3:
    st.markdown("**Model Performance**")
    st.write("Compare accuracy, speed, and size across every model trained in this project.")

st.divider()

st.markdown("### The 25 classes")
categories = {
    "Vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "airplane"],
    "Person": ["person"],
    "Outdoor": ["traffic light", "stop sign", "bench"],
    "Animals": ["dog", "cat", "horse", "bird", "cow", "elephant"],
    "Kitchen & food": ["bottle", "cup", "bowl", "pizza", "cake"],
    "Furniture & indoor": ["chair", "couch", "bed", "potted plant"],
}
cols = st.columns(len(categories))
for col, (cat, items) in zip(cols, categories.items()):
    with col:
        st.markdown(f"**{cat}**")
        for item in items:
            st.write(f"- {item}")

st.divider()
st.caption(
    "Use the sidebar to navigate to Image Classification, Object Detection, "
    "Model Performance, or About."
)
