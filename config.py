"""
config.py
Central configuration for SmartVision AI.
All scripts (data prep, training, app) import from here so class lists
and paths never drift out of sync.
"""

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "smartvision_dataset")

CLASSIFICATION_DIR = os.path.join(DATASET_DIR, "classification")
CLASSIFICATION_TRAIN_DIR = os.path.join(CLASSIFICATION_DIR, "train")
CLASSIFICATION_VAL_DIR = os.path.join(CLASSIFICATION_DIR, "val")
CLASSIFICATION_TEST_DIR = os.path.join(CLASSIFICATION_DIR, "test")

DETECTION_DIR = os.path.join(DATASET_DIR, "detection")
DETECTION_IMAGES_DIR = os.path.join(DETECTION_DIR, "images")
DETECTION_LABELS_DIR = os.path.join(DETECTION_DIR, "labels")
DETECTION_YAML_PATH = os.path.join(DETECTION_DIR, "data.yaml")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "classification_metrics.json")
YOLO_METRICS_PATH = os.path.join(OUTPUTS_DIR, "yolo_metrics.json")

# ---------------------------------------------------------------------------
# The 25 selected COCO classes (must match COCO category names exactly)
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

CLASS_TO_IDX = {name: i for i, name in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(SELECTED_CLASSES)}

# Folder-safe versions of class names (spaces -> underscores) since
# "traffic light" / "stop sign" / "potted plant" contain spaces.
def safe_name(class_name: str) -> str:
    return class_name.replace(" ", "_")

SAFE_CLASSES = [safe_name(c) for c in SELECTED_CLASSES]

# ---------------------------------------------------------------------------
# Dataset collection settings
# ---------------------------------------------------------------------------
IMAGES_PER_CLASS = 350          # -> 8,750 images total (Option B: up from 100/class)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLS_IMG_SIZE = 224              # Stage 1 (frozen base) classification input size
FINE_TUNE_IMG_SIZE = 384        # Stage 2 (fine-tuning) progressive-resizing input size
YOLO_IMG_SIZE = 640              # YOLO input size

HF_DATASET_NAME = "detection-datasets/coco"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS_FROZEN = 10      # feature-extraction phase
EPOCHS_FINE_TUNE = 10   # fine-tuning phase
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5

YOLO_EPOCHS = 50
YOLO_BATCH = 16

CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
