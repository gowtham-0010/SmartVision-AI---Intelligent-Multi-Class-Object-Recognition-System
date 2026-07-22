"""
config.py
Central configuration for SmartVision AI. All app files import from here so
class lists and paths never drift out of sync.
"""

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "classification_metrics.json")
YOLO_METRICS_PATH = os.path.join(OUTPUTS_DIR, "yolo_metrics.json")

# ---------------------------------------------------------------------------
# The 25 selected COCO classes (must match COCO category names exactly).
#
# IMPORTANT: this is the "human-authored" order, used for YOLO's class IDs
# (CLASS_TO_IDX below) since those were baked into the YOLO label .txt files
# in notebook 10 using this exact order. It is NOT the order the classifiers'
# output neurons use - see CLASSIFIER_CLASS_NAMES further down.
# ---------------------------------------------------------------------------
SELECTED_CLASSES = [
    # Vehicles (6)
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane",
    # Person (1)
    "person",
    # Outdoor (3)
    "traffic light", "stop sign", "bench",
    # Animals (6)
    "dog", "cat", "horse", "bird", "cow", "elephant",
    # Kitchen & food (5)
    "bottle", "cup", "bowl", "pizza", "cake",
    # Furniture & indoor (4)
    "chair", "couch", "bed", "potted plant",
]

assert len(SELECTED_CLASSES) == 25, "Must have exactly 25 classes"

# Used by YOLO only - matches the class IDs baked into the YOLO label .txt
# files (notebook 10) and data.yaml's "names" list, both built from this
# exact SELECTED_CLASSES order.
CLASS_TO_IDX = {name: i for i, name in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(SELECTED_CLASSES)}


def safe_name(class_name: str) -> str:
    """Folder-safe version of a class name (spaces -> underscores), since
    'traffic light' / 'stop sign' / 'potted plant' contain spaces."""
    return class_name.replace(" ", "_")


SAFE_CLASSES = [safe_name(c) for c in SELECTED_CLASSES]

# ---------------------------------------------------------------------------
# CLASSIFIER class ordering - DELIBERATELY DIFFERENT from SELECTED_CLASSES.
#
# tf.keras.utils.image_dataset_from_directory (used by notebooks 05-08 to
# build the training pipeline) assigns each class an index based on
# ALPHABETICAL folder order, not the order classes happen to be listed in
# SELECTED_CLASSES above. That alphabetical order is what each classifier's
# output neurons actually correspond to.
#
# Decoding a classifier's prediction with SELECTED_CLASSES[idx] instead of
# this list silently mismaps every single prediction (e.g. a correctly
# recognized "elephant" gets displayed as "bottle", since both orderings
# happen to disagree at that index). This bug was found and fixed after
# manual testing showed exactly this symptom - see CLASSIFIER_CLASS_NAMES
# usage in app/inference_pipeline.py and app/pages/1_Image_Classification.py.
# ---------------------------------------------------------------------------
CLASSIFIER_CLASS_NAMES = sorted(SAFE_CLASSES)


def classifier_idx_to_display_name(idx: int) -> str:
    """Convert a classifier's raw output index into a human-readable class
    name (e.g. 'traffic_light' -> 'traffic light'). Always use this - never
    index directly into SELECTED_CLASSES with a classifier's prediction index."""
    return CLASSIFIER_CLASS_NAMES[idx].replace("_", " ")


# ---------------------------------------------------------------------------
# Dataset collection settings
# ---------------------------------------------------------------------------
IMAGES_PER_CLASS = 350           # -> 8,750 images total
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLS_IMG_SIZE = 224               # classification input size (single resolution)
YOLO_IMG_SIZE = 640              # YOLO input size

HF_DATASET_NAME = "detection-datasets/coco"

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
CROP_PADDING_RATIO = 0.08        # defensive padding applied around YOLO boxes
                                  # before cropping for classifier re-verification
