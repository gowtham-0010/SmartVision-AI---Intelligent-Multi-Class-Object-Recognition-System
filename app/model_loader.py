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
    Uses tf_keras (Keras 2 legacy engine) as a robust fallback for Keras 3 deserialization."""
    import tensorflow as tf
    import keras
    
    # Legacy Keras 2 deserializer engine
    try:
        import tf_keras
    except ImportError:
        tf_keras = None

    models = {}
    for key in CLASSIFIER_KEYS:
        path = os.path.join(MODELS_DIR, f"{key}_best.keras")
        if os.path.exists(path):
            # Attempt 1: Keras 3 native loader
            try:
                models[key] = keras.models.load_model(
                    path, 
                    compile=False, 
                    safe_mode=False
                )
            except Exception:
                # Attempt 2: tf_keras legacy loader (fixes TypeError)
                if tf_keras is not None:
                    try:
                        models[key] = tf_keras.models.load_model(path, compile=False)
                    except Exception as e:
                        print(f"Failed loading {key} with tf_keras: {e}")
                else:
                    # Attempt 3: Standard tf.keras compile=False
                    try:
                        models[key] = tf.keras.models.load_model(path, compile=False, safe_mode=False)
                    except Exception as e:
                        print(f"Failed loading {key}: {e}")

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
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_yolo_metrics():
    if os.path.exists(YOLO_METRICS_PATH):
        with open(YOLO_METRICS_PATH) as f:
            return json.load(f)
    return {}
