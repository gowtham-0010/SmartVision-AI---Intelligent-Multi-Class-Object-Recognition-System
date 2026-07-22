"""
model_loader.py
Shared, Streamlit-cached model loading used by every page so weights are
only read from disk once per session instead of once per page render.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, METRICS_PATH, YOLO_METRICS_PATH

import streamlit as st

CLASSIFIER_KEYS = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
CLASSIFIER_DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}


@st.cache_resource(show_spinner="Loading classification models...")
def load_all_classifiers():
    """Load every trained classifier .keras file that exists in models/.
    Native .keras format loads the full model (architecture + regularizers +
    weights) with no custom_objects needed. Skips models that haven't been
    trained/saved yet, so the app still runs with partial results."""
    import tensorflow as tf
    models = {}
    for key in CLASSIFIER_KEYS:
        path = os.path.join(MODELS_DIR, f"{key}_best.keras")
        if os.path.exists(path):
            models[key] = tf.keras.models.load_model(path)
    return models


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo_model():
    from ultralytics import YOLO
    candidate_paths = [
        os.path.join(MODELS_DIR, "yolo_smartvision", "weights", "best.pt"),
        os.path.join(MODELS_DIR, "yolov8_best.pt"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return YOLO(path), path
    return None, None


@st.cache_data
def load_classification_metrics():
    """Loads outputs/metrics_<key>.json for every classifier that has one,
    keyed by model key (vgg16, resnet50, mobilenetv2, efficientnetb0)."""
    metrics = {}
    outputs_dir = os.path.dirname(METRICS_PATH)
    for key in CLASSIFIER_KEYS:
        path = os.path.join(outputs_dir, f"metrics_{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[key] = json.load(f)
    return metrics


@st.cache_data
def load_yolo_metrics():
    if os.path.exists(YOLO_METRICS_PATH):
        with open(YOLO_METRICS_PATH) as f:
            return json.load(f)
    return {}
