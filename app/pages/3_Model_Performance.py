"""
Page 4: Model Performance
Dashboard comparing all classification models and the YOLO detector.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import load_classification_metrics, load_yolo_metrics, CLASSIFIER_DISPLAY_NAMES

st.set_page_config(page_title="Model Performance - SmartVision AI", layout="wide")
st.title("Model performance")

cls_metrics = load_classification_metrics()
yolo_metrics = load_yolo_metrics()

st.markdown("### Classification models")
if not cls_metrics:
    st.info("No classification metrics yet. Run `training/train_classifiers.py` to generate them.")
else:
    rows = []
    for key, m in cls_metrics.items():
        rows.append({
            "Model": m["model"],
            "Test accuracy": m["test_accuracy"],
            "Precision (macro)": m["precision_macro"],
            "Recall (macro)": m["recall_macro"],
            "F1 (macro)": m["f1_macro"],
            "Inference (ms/img)": m["avg_inference_ms"],
            "Size (MB)": m["model_size_mb"],
        })
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Accuracy by model**")
        st.bar_chart(df["Test accuracy"])
    with c2:
        st.markdown("**Inference time by model**")
        st.bar_chart(df["Inference (ms/img)"])

st.divider()
st.markdown("### Object detection (YOLOv8)")
if not yolo_metrics:
    st.info("No YOLO metrics yet. Run `training/train_yolo.py` to generate them.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP@0.5", f"{yolo_metrics['map50']*100:.1f}%")
    c2.metric("mAP@0.5:0.95", f"{yolo_metrics['map50_95']*100:.1f}%")
    c3.metric("Precision", f"{yolo_metrics['precision']*100:.1f}%")
    c4.metric("Recall", f"{yolo_metrics['recall']*100:.1f}%")
    if yolo_metrics.get("fps_estimate"):
        st.metric("Estimated FPS", yolo_metrics["fps_estimate"])
