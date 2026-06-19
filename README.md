---
title: SmartVision AI System
emoji: ⚡
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# SmartVision AI

SmartVision AI is a computer vision system that detects and classifies multiple objects in images using deep learning.

## Main Goal

Build an intelligent system that can:
- Identify objects such as cars, people, and animals
- Detect multiple objects in one image with bounding boxes
- Run fast enough for near real-time usage
- Be deployed as a Streamlit web app

## Core Technologies

- Python
- Deep Learning (CNN models)
- Transfer Learning
- YOLOv8 for object detection
- Streamlit for the web app
- COCO dataset

## Project Workflow

1. Data Preparation
   - Uses COCO subset (25 classes, 2500 images)
   - Splits into train/validation/test
   - Prepares YOLO-format detection labels
   - Prepares cropped classification dataset
2. Classification Models
   - VGG16
   - ResNet50
   - MobileNetV2
   - EfficientNetB0
3. Object Detection
   - YOLOv8 detection with bounding boxes and labels
4. Inference Pipeline
   - User uploads image
   - YOLO detects objects
   - Results shown with confidence scores
5. Web App Pages
   - Home (Dashboard)
   - Classification
   - Detection
   - Performance
   - Webcam
   - About

## Expected Results

- Classification accuracy: 80% to 93%
- YOLO detection: 85%+ mAP target
- Fast inference for near real-time usage

## Real-World Use Cases

- Smart traffic systems
- Retail and inventory
- Security surveillance
- Wildlife monitoring
- Healthcare support systems
- Smart home automation

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run App

```bash
python -m streamlit run app/app.py
```
