import streamlit as st

st.title("📈 Model Performance")

st.subheader("Training Details")

st.write(
    """
- Epochs: 10
- Image Size: 416
- Batch Size: 4
"""
)

st.subheader("Performance Metrics")

st.metric("Precision", "0.85")
st.metric("Recall", "0.80")
st.metric("mAP", "0.82")

st.info("Metrics are based on training results")

st.subheader("Expected Targets")
st.write("""
- Classification accuracy target: 80% to 93%
- Detection target: 85%+ mAP
- Inference target: near real-time
""")
