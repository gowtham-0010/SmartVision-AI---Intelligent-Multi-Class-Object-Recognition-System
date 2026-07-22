"""
inference_pipeline.py
Phase 4 - Model Integration & Pipeline Development

End-to-end prediction pipeline used by the Streamlit app:

    user image -> YOLO detects all objects with bounding boxes
               -> NMS removes duplicate/overlapping detections
               -> confidence threshold filters weak detections
               -> padded crop -> resize -> (optional) best CNN classifier
                  re-verifies each crop
               -> annotated image + structured results returned

This module has NO Streamlit dependency so it can also be unit-tested
or reused in a Flask/FastAPI backend if needed.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLS_IMG_SIZE, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, CROP_PADDING_RATIO,
    classifier_idx_to_display_name,
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SmartVisionPipeline:
    def __init__(self, yolo_weights_path, classifier_weights_path=None,
                 classifier_preprocess_fn=None):
        """
        yolo_weights_path: path to fine-tuned YOLOv8 .pt weights
        classifier_weights_path: optional path to a trained Keras .keras model
                                  (native format) used to re-verify each YOLO crop
        classifier_preprocess_fn: the matching preprocess_input function
                                   for whichever backbone was trained
                                   (e.g. tensorflow.keras.applications.resnet50.preprocess_input)
        """
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_weights_path)

        self.classifier = None
        self.classifier_input_size = CLS_IMG_SIZE
        self.classifier_preprocess_fn = classifier_preprocess_fn
        if classifier_weights_path and os.path.exists(classifier_weights_path):
            import tensorflow as tf
            self.classifier = tf.keras.models.load_model(classifier_weights_path)
            # Read the actual expected input size off the loaded model rather
            # than assuming a fixed value.
            self.classifier_input_size = self.classifier.input_shape[1]

    def detect(self, image: Image.Image, confidence=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD):
        """Run YOLO detection + NMS + confidence filtering.
        Returns list of dicts: {box, class_name, confidence}"""
        start = time.time()
        results = self.yolo.predict(
            source=np.array(image.convert("RGB")),
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        elapsed_ms = (time.time() - start) * 1000

        detections = []
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names.get(cls_id, str(cls_id))
            detections.append({
                "box": [round(v, 1) for v in xyxy],  # [x1,y1,x2,y2]
                "class_name": class_name,
                "confidence": round(conf, 4),
            })

        return detections, round(elapsed_ms, 1)

    def classify_crop(self, image: Image.Image, box, padding_ratio=CROP_PADDING_RATIO):
        """Run the optional CNN classifier on a single YOLO-detected crop
        and return its own top-5 predictions, for verification/display.

        Applies defensive padding around the box before resizing to the
        classifier's actual expected input size.

        IMPORTANT: prediction indices are decoded via
        classifier_idx_to_display_name(), NOT SELECTED_CLASSES[idx] directly.
        The classifier's output neurons are ordered alphabetically by folder
        name (how tf.keras.utils.image_dataset_from_directory assigned them
        during training), which is a different order from SELECTED_CLASSES'
        hand-authored list. Indexing into SELECTED_CLASSES directly silently
        mismaps every prediction (e.g. a correctly recognized "elephant"
        would display as "bottle").
        """
        if self.classifier is None:
            return None

        img_w, img_h = image.size
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        crop = image.convert("RGB").crop((x1, y1, x2, y2)).resize(
            (self.classifier_input_size, self.classifier_input_size)
        )
        arr = np.expand_dims(np.array(crop).astype("float32"), axis=0)
        if self.classifier_preprocess_fn:
            arr = self.classifier_preprocess_fn(arr)

        preds = self.classifier.predict(arr, verbose=0)[0]
        top5_idx = np.argsort(preds)[::-1][:5]
        return [
            {"class_name": classifier_idx_to_display_name(i), "confidence": round(float(preds[i]), 4)}
            for i in top5_idx
        ]

    def run(self, image: Image.Image, confidence=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD, verify_with_classifier=False):
        """Full pipeline: detect -> (optional) classify each crop -> annotate."""
        detections, inference_ms = self.detect(image, confidence, iou)

        if verify_with_classifier and self.classifier is not None:
            for det in detections:
                det["classifier_top5"] = self.classify_crop(image, det["box"])

        annotated = draw_detections(image, detections)
        return {
            "detections": detections,
            "inference_ms": inference_ms,
            "annotated_image": annotated,
        }


def draw_detections(image: Image.Image, detections, box_color=(29, 158, 117), text_color=(255, 255, 255)):
    """Draw bounding boxes + labels + confidence scores on a copy of the image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['confidence']*100:.0f}%"

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1], fill=box_color)
        draw.text((x1 + 4, max(0, y1 - text_h - 4)), label, fill=text_color, font=font)

    return annotated


def load_classifier_preprocess(model_key):
    """Helper to fetch the right preprocess_input for a given model key,
    used when wiring the app together."""
    from tensorflow.keras.applications import vgg16, resnet50, mobilenet_v2, efficientnet
    mapping = {
        "vgg16": vgg16.preprocess_input,
        "resnet50": resnet50.preprocess_input,
        "mobilenetv2": mobilenet_v2.preprocess_input,
        "efficientnetb0": efficientnet.preprocess_input,
    }
    return mapping.get(model_key)
