"""
Page 2: Image Classification
Upload a single-object image and see predictions from all trained CNNs.
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import classifier_idx_to_display_name

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import load_all_classifiers, CLASSIFIER_DISPLAY_NAMES
from inference_pipeline import load_classifier_preprocess

st.set_page_config(page_title="Image Classification - SmartVision AI", layout="wide")
st.title("Image classification")
st.write(
    "Upload a photo of a single object, tightly cropped around it (this matches how "
    "the models were trained - on individual cropped objects, not full scenes). "
    "Every trained model will predict its class independently."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    models = load_all_classifiers()

    if not models:
        st.warning(
            "No trained classifier weights found in `models/`. "
            "Run notebooks 05-08 in Colab first, then download the resulting "
            "`*_best.keras` files into this project's `models/` folder."
        )
    else:
        with col_results:
            st.markdown("### Predictions by model")

            tabs = st.tabs([CLASSIFIER_DISPLAY_NAMES[k] for k in models.keys()])
            all_top1 = {}

            for tab, (key, model) in zip(tabs, models.items()):
                with tab:
                    # Each model may expect a different input size - resize
                    # per-model based on what it actually expects rather than
                    # assuming a fixed value.
                    input_size = model.input_shape[1]
                    resized = image.resize((input_size, input_size))
                    arr = np.expand_dims(np.array(resized).astype("float32"), axis=0)

                    preprocess_fn = load_classifier_preprocess(key)
                    model_input = preprocess_fn(arr.copy()) if preprocess_fn else arr
                    preds = model.predict(model_input, verbose=0)[0]
                    top5_idx = np.argsort(preds)[::-1][:5]

                    # IMPORTANT: decode indices via classifier_idx_to_display_name(),
                    # NOT SELECTED_CLASSES[idx] directly. The classifier's output
                    # neurons are ordered alphabetically by folder name (how
                    # tf.keras.utils.image_dataset_from_directory assigned them
                    # during training) - a different order from SELECTED_CLASSES'
                    # hand-authored list. Using SELECTED_CLASSES[idx] here would
                    # silently mismap every prediction.
                    top1_name = classifier_idx_to_display_name(top5_idx[0])
                    all_top1[key] = (top1_name, float(preds[top5_idx[0]]))

                    for rank, idx in enumerate(top5_idx, start=1):
                        class_display_name = classifier_idx_to_display_name(idx)
                        st.write(f"{rank}. **{class_display_name}** - {preds[idx]*100:.1f}%")
                        st.progress(float(preds[idx]))

            st.divider()
            st.markdown("### Side-by-side top prediction")
            summary_cols = st.columns(len(all_top1))
            for col, (key, (cls_name, conf)) in zip(summary_cols, all_top1.items()):
                with col:
                    st.metric(CLASSIFIER_DISPLAY_NAMES[key], cls_name, f"{conf*100:.1f}%")
else:
    st.info("Upload an image above to get started.")
