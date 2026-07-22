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
    import tensorflow as tf
    import keras
    
    try:
        import tf_keras
    except ImportError:
        tf_keras = None

    models = {}

    # Print current working directory and target models path for debugging
    abs_models_dir = os.path.abspath(MODELS_DIR)
    st.info(f"🔍 Searching for models in: `{abs_models_dir}`")
    
    if os.path.exists(abs_models_dir):
        existing_files = os.listdir(abs_models_dir)
        st.write(f"📁 Files found in models folder: `{existing_files}`")
    else:
        st.error(f"❌ Directory does not exist: `{abs_models_dir}`")

    for key in CLASSIFIER_KEYS:
        expected_filename = f"{key}_best.keras"
        path = os.path.join(abs_models_dir, expected_filename)
        
        if os.path.exists(path):
            st.write(f"⏳ Attempting to load `{expected_filename}`...")
            try:
                models[key] = keras.models.load_model(path, compile=False, safe_mode=False)
                st.success(f"✅ Successfully loaded `{expected_filename}`!")
            except Exception as e1:
                st.warning(f"Keras 3 failed for `{expected_filename}`: {e1}")
                if tf_keras:
                    try:
                        models[key] = tf_keras.models.load_model(path, compile=False)
                        st.success(f"✅ Successfully loaded `{expected_filename}` using tf_keras!")
                    except Exception as e2:
                        st.error(f"❌ tf_keras also failed for `{expected_filename}`: {e2}")
                else:
                    try:
                        models[key] = tf.keras.models.load_model(path, compile=False, safe_mode=False)
                        st.success(f"✅ Loaded `{expected_filename}` via tf.keras!")
                    except Exception as e3:
                        st.error(f"❌ tf.keras failed for `{expected_filename}`: {e3}")
        else:
            st.warning(f"⚠️ Could not find expected file `{expected_filename}` at `{path}`")

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
