import streamlit as st

st.title("📊 Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Model", "YOLOv8")
col2.metric("Status", "Active ✅")
col3.metric("Dataset", "25 classes")

st.markdown("---")

st.subheader("Project Overview")

st.write("""
SmartVision AI is designed for:
- Object Detection
- Image Classification
- Real-time monitoring
""")

st.subheader("Main Goal")
st.write("""
- Detect multiple objects with bounding boxes
- Classify objects from uploaded images
- Support fast inference for real-time use
- Provide an interactive web app experience
""")

st.subheader("Core Technologies")
st.write("""
- Python
- YOLOv8
- CNN transfer learning models
- Streamlit
- COCO-based dataset
""")