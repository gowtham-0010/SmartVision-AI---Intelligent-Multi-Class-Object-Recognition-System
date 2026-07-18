"""
Page 6: About
Project documentation, dataset info, model architectures, tech stack.
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SELECTED_CLASSES, IMAGES_PER_CLASS

st.set_page_config(page_title="About - SmartVision AI", layout="wide")
st.title("About SmartVision AI")

st.markdown(
    f"""
### Project overview
SmartVision AI is an intelligent multi-class object recognition system that
combines transfer-learning image classification with YOLO-based object
detection, trained on a curated {IMAGES_PER_CLASS}-image-per-class subset
({IMAGES_PER_CLASS * len(SELECTED_CLASSES)} images total) of the COCO 2017
dataset across {len(SELECTED_CLASSES)} classes.

### Model architectures used
- **VGG16** - frozen convolutional base, new dense classification head
- **ResNet50** - fine-tuned last 20 layers
- **MobileNetV2** - frozen base, optimised for inference speed
- **EfficientNetB0** - fine-tuned with mixed precision training
- **YOLOv8** - fine-tuned on the 25-class detection subset for multi-object localisation

### Technical stack
Python - TensorFlow/Keras - PyTorch (Ultralytics YOLO) - OpenCV - Streamlit -
Hugging Face Datasets - scikit-learn - Matplotlib

### Dataset
- Source: `detection-datasets/coco` on Hugging Face (streamed, not fully downloaded)
- 25 curated classes spanning vehicles, animals, people, furniture, and kitchen/food items
- 70% / 15% / 15% train / validation / test split
- Classification set: individual objects cropped to 224x224
- Detection set: full images with YOLO-format bounding box annotations

### Project structure
```
smartvision_ai/
├── config.py
├── data_preparation/
│   ├── prepare_dataset.py
│   └── eda.py
├── training/
│   ├── train_classifiers.py
│   ├── compare_models.py
│   └── train_yolo.py
├── app/
│   ├── Home.py
│   ├── model_loader.py
│   ├── inference_pipeline.py
│   └── pages/
│       ├── 1_Image_Classification.py
│       ├── 2_Object_Detection.py
│       ├── 3_Model_Performance.py
│       ├── 4_Live_Webcam_Detection.py
│       └── 5_About.py
├── models/          (trained weights land here)
├── outputs/          (metrics + charts land here)
├── requirements.txt
└── README.md
```
"""
)
