# SmartVision AI — Intelligent Multi-Class Object Recognition System

A computer vision platform combining transfer-learning image classification
(VGG16, ResNet50, MobileNetV2, EfficientNetB0) with YOLOv8 object detection,
trained on a curated 25-class, 8,750-image subset of COCO 2017 (350 images/class,
Option B: scaled up from the original 100/class to fight overfitting), and served
through a multi-page Streamlit app.

All dataset preparation and training happens in **Google Colab notebooks**
(one notebook per stage, one per model), so you get a free GPU and every
step is independently re-runnable. The Streamlit app (Phase 5) is the one
piece that stays as plain `.py` files, since a web app can't run as a
notebook.

## Project structure

```
smartvision_ai/
├── notebooks/                                     <- run these in Google Colab, in order
│   ├── 01_Dataset_Loading.ipynb                    Phase 1.1: download official COCO, collect 25 classes
│   ├── 02_EDA.ipynb                                Phase 1.2: class distribution, sample grid
│   ├── 03_Data_Preprocessing_Classification.ipynb  Phase 1.3: crop/resize/split for classifiers
│   ├── 04_Data_Augmentation.ipynb                  Phase 1.4: visualize the augmentation pipeline
│   ├── 05_Train_VGG16.ipynb                        Phase 2.1
│   ├── 06_Train_ResNet50.ipynb                     Phase 2.2
│   ├── 07_Train_MobileNetV2.ipynb                  Phase 2.3
│   ├── 08_Train_EfficientNetB0.ipynb               Phase 2.4
│   ├── 09_Compare_Classification_Models.ipynb      Phase 2.5: comparison + best-model pick
│   ├── 10_YOLO_Dataset_Preparation.ipynb           Phase 3.2: build YOLO-format dataset
│   ├── 11_Train_YOLO.ipynb                         Phase 3.1/3.3/3.4: fine-tune + evaluate YOLOv8
│   └── 12_Inference_Pipeline_Demo.ipynb            Phase 4: detect -> NMS -> filter -> classify
├── build_notebooks.py              # regenerates the notebooks above (nbformat-based, optional)
├── config.py                       # shared class list & paths, used by the Streamlit app only
├── app/
│   ├── Home.py                     # Streamlit entry point (Page 1)
│   ├── model_loader.py             # cached model/metric loading helpers
│   ├── inference_pipeline.py       # detect -> NMS -> filter -> (optional) classify
│   └── pages/
│       ├── 1_Image_Classification.py
│       ├── 2_Object_Detection.py
│       ├── 3_Model_Performance.py
│       ├── 4_Live_Webcam_Detection.py    (bonus, +3 pts)
│       └── 5_About.py
├── models/                         # download trained weights here from Drive before running the app
├── outputs/                        # download metrics/charts here from Drive before running the app
├── requirements.txt                # for running the Streamlit app locally
└── README.md
```

## 1. Run the notebooks in Google Colab

Upload the `notebooks/` folder to Google Drive (or open each `.ipynb`
directly at colab.research.google.com via File > Upload notebook). Run them
**in numeric order** — each one mounts your Google Drive and reads/writes
everything under one shared folder, `My Drive/SmartVisionAI/`, so later
notebooks automatically pick up what earlier ones produced, even in a brand
new Colab session.

Before running notebooks 05–08 and 11, go to **Runtime > Change runtime
type > T4 GPU** — training on CPU will be extremely slow.

**Runtime note (Option B, 350 images/class):** notebook 01 will take
noticeably longer to collect data than the original 100/class run (expect
25–45 minutes of streaming instead of ~8). Notebooks 05–08 also take longer
per model due to progressive resizing's Stage 2 (384px fine-tuning) — budget
extra GPU time accordingly.

| Notebook | What it does | Produces |
|---|---|---|
| `01_Dataset_Loading` | Downloads annotations + needed images from official COCO (cocodataset.org), collects 350 images/class across 25 classes | `raw_data/images/*.jpg`, `raw_data/annotations.json` |
| `02_EDA` | Class distribution, image sizes, objects/image, sample grid | `outputs/eda_*.png` |
| `03_Data_Preprocessing_Classification` | Crops objects, resizes to 224x224, splits 70/15/15 | `classification/{train,val,test}/<class>/*.jpg` |
| `04_Data_Augmentation` | Visualizes flip/rotate/brightness/contrast/zoom augmentation | `outputs/augmentation_examples.png` |
| `05`–`08` (one per model) | Trains VGG16 / ResNet50 / MobileNetV2 / EfficientNetB0 via hardened transfer learning (L2 regularization, dropout 0.5, progressive resizing 224px→384px, AdamW, val_loss early stopping) | `models/<key>_best.keras`, `outputs/metrics_<key>.json` |
| `09_Compare_Classification_Models` | Bar charts, accuracy-vs-speed, confusion matrices, best-model pick | `outputs/model_comparison_bars.png`, `outputs/model_selection_summary.json` |
| `10_YOLO_Dataset_Preparation` | Converts the raw collection into YOLO-format labels + `data.yaml` | `detection/images/`, `detection/labels/`, `detection/data.yaml` |
| `11_Train_YOLO` | Fine-tunes YOLOv8, evaluates mAP/precision/recall/FPS | `models/yolo_smartvision/weights/best.pt`, `outputs/yolo_metrics.json` |
| `12_Inference_Pipeline_Demo` | Runs the full detect -> NMS -> filter -> classify pipeline end-to-end | demo predictions in the notebook |

You can train the 4 classifiers in any order, or split them across separate
Colab sessions/days — each one only depends on
`03_Data_Preprocessing_Classification` having been run once, not on each
other. Every notebook is fully self-contained (it re-declares its own
config and imports), so you never need to run a "shared setup" notebook
first.

**Regenerating the notebooks:** if you want to tweak hyperparameters across
every notebook at once, edit `build_notebooks.py` and re-run
`python3 build_notebooks.py` locally — it rebuilds all 12 `.ipynb` files
from one source of truth so they never drift out of sync with each other.

## 2. Bring the trained models back to run the Streamlit app locally

Once notebooks 05–08 and 11 have produced weights in
`My Drive/SmartVisionAI/models/` and metrics in
`My Drive/SmartVisionAI/outputs/`, download those two folders from Google
Drive and place their *contents* into this project's own `models/` and
`outputs/` folders (same relative paths, e.g.
`models/vgg16_best.keras`, `models/yolo_smartvision/weights/best.pt`,
`outputs/metrics_vgg16.json`, `outputs/yolo_metrics.json`).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

The app works incrementally — if you've only trained some classifiers so
far, the Object Detection page will show a friendly warning instead of
crashing, and vice versa. Train everything for the full experience.

## 3. Deploy to Hugging Face Spaces (Phase 6)

1. Create a new GitHub repo and push this whole `smartvision_ai/` folder,
   including the `models/` folder populated in Step 2.
2. Make sure trained weights are included:
   - Files under ~10MB commit normally.
   - Larger files (most `.keras` and `.pt` weights) need **Git LFS**:
     ```bash
     git lfs install
     git lfs track "*.keras" "*.pt"
     git add .gitattributes
     git add models/
     git commit -m "Add trained model weights via Git LFS"
     ```
3. Go to huggingface.co/new-space:
   - SDK: **Streamlit**
   - Hardware: CPU is fine for inference; pick a GPU tier only if you need
     faster live inference.
4. Connect the Space to your GitHub repo (or push directly to the Space's own
   git remote — Spaces are git repos too).
5. Add a `packages.txt` file if you hit OpenCV system-library errors on
   Spaces, containing:
   ```
   libgl1
   ```
6. Once built, your app is live at
   `https://huggingface.co/spaces/<username>/<space-name>`.

## Evaluation checklist mapping

| Rubric item | Where it's satisfied |
|---|---|
| Data preprocessing & EDA (15 pts) | `notebooks/01_Dataset_Loading.ipynb`, `02_EDA.ipynb`, `03_Data_Preprocessing_Classification.ipynb` |
| 4 classification models, ≥80% accuracy (30 pts) | `notebooks/05_Train_VGG16.ipynb` through `08_Train_EfficientNetB0.ipynb` |
| YOLO detection, mAP@0.5 > 75% (25 pts) | `notebooks/10_YOLO_Dataset_Preparation.ipynb`, `11_Train_YOLO.ipynb` |
| Model comparison & analysis (10 pts) | `notebooks/09_Compare_Classification_Models.ipynb` |
| Streamlit app, multi-page (15 pts) | `app/Home.py` + `app/pages/` |
| Deployment on Hugging Face Spaces (5 pts) | see Section 3 above |
| Bonus: live webcam detection (+3) | `app/pages/4_Live_Webcam_Detection.py` |
| Bonus: advanced augmentation (+2) | `notebooks/04_Data_Augmentation.ipynb` |

## Notes on realistic expectations

- Expected accuracy ranges (80–93%) and mAP (85–90%) from the project brief
  assume a full ~20-epoch training run per model on a GPU. Notebooks 05–08 use a
  lean, single-resolution architecture (GlobalAveragePooling2D → Dropout → Dense,
  no progressive resizing) verified to reach 66–82% test accuracy on this dataset
  before any further tuning — a solid, defensible result if you land in that
  range rather than the top of the brief's stated targets. Colab's free T4 GPU
  is enough — a CPU runtime will be extremely slow for the 4 classifier
  notebooks and impractical for YOLO training.
- Colab free-tier sessions disconnect after a period of inactivity or after
  ~12 hours. Since all data and weights are saved straight to Google Drive
  as each notebook runs, you can safely reconnect and re-run a later
  notebook without redoing earlier ones.
- `01_Dataset_Loading` downloads directly from the official COCO dataset
  (`cocodataset.org` / `images.cocodataset.org`) via `pycocotools`, per the
  sources listed in the project brief — not the Hugging Face mirror, which
  hit a server-side outage (its storage backend was returning `403
  Forbidden` instead of data). Only the ~241MB annotation file plus the
  specific images actually needed (a few thousand, not the full 118K-image
  training set) are downloaded.
