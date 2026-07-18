"""
build_notebooks.py
Generates all SmartVision AI Colab notebooks as valid .ipynb files.
Run once: python3 build_notebooks.py
"""

import nbformat as nbf
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)


def new_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    }
    return nb


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


def save(name, cells):
    nb = new_notebook(cells)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Shared boilerplate cells reused (with small variations) across notebooks
# ---------------------------------------------------------------------------

def setup_cells(extra_pip=""):
    pip_line = (
        "!pip install -q datasets pyyaml tqdm scikit-learn matplotlib pandas "
        "opencv-python-headless" + (" " + extra_pip if extra_pip else "")
    )
    return [
        code(pip_line),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
        ),
    ]


CONFIG_CELL_SOURCE = '''\
import os

# ---------------------------------------------------------------------------
# Project configuration - shared across every SmartVision AI notebook.
# All notebooks read/write under this same Google Drive folder so that
# work done in one notebook (e.g. dataset collection) is available to the
# next one (e.g. training), even across separate Colab sessions.
# ---------------------------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/SmartVisionAI"

RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")
RAW_ANNOTATIONS_PATH = os.path.join(RAW_DATA_DIR, "annotations.json")

CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classification")
CLASSIFICATION_TRAIN_DIR = os.path.join(CLASSIFICATION_DIR, "train")
CLASSIFICATION_VAL_DIR = os.path.join(CLASSIFICATION_DIR, "val")
CLASSIFICATION_TEST_DIR = os.path.join(CLASSIFICATION_DIR, "test")

DETECTION_DIR = os.path.join(BASE_DIR, "detection")
DETECTION_IMAGES_DIR = os.path.join(DETECTION_DIR, "images")
DETECTION_LABELS_DIR = os.path.join(DETECTION_DIR, "labels")
DETECTION_YAML_PATH = os.path.join(DETECTION_DIR, "data.yaml")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in [BASE_DIR, RAW_DATA_DIR, RAW_IMAGES_DIR, CLASSIFICATION_DIR, DETECTION_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# The 25 selected COCO classes (must match COCO category names exactly)
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
assert len(SELECTED_CLASSES) == 25

CLASS_TO_IDX = {name: i for i, name in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(SELECTED_CLASSES)}

def safe_name(class_name):
    return class_name.replace(" ", "_")

IMAGES_PER_CLASS = 350        # -> 8,750 images total (up from 100/class to fight overfitting)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

CLS_IMG_SIZE = 224            # Classification input resolution (single-resolution throughout)
FINE_TUNE_IMG_SIZE = 384      # Unused by classifier training (reverted to single-resolution); kept for compatibility
YOLO_IMG_SIZE = 640
BATCH_SIZE = 32                # Stage 1 batch size
BATCH_SIZE_STAGE2 = 16         # Smaller batch at 384x384 to fit GPU memory (~2.9x pixels/image)

HF_DATASET_NAME = "detection-datasets/coco"

print("BASE_DIR:", BASE_DIR)
print("Classes:", len(SELECTED_CLASSES))
'''


def config_cell():
    return code(CONFIG_CELL_SOURCE)


# ===========================================================================
# NOTEBOOK 1 — Dataset Loading
# ===========================================================================
nb1_cells = [
    md(
        "# 01 - Dataset Loading\n\n"
        "**Phase 1, Step 1.1 - Dataset Loading**\n\n"
        "**Source: official COCO 2017 dataset (`cocodataset.org`)**, not the Hugging Face "
        "streaming mirror. The `detection-datasets/coco` HF mirror hit a server-side outage "
        "(its storage backend was returning `403 Forbidden` / malformed HTML instead of data), "
        "so this notebook uses the same source named in the project brief instead: "
        "`https://cocodataset.org/` via the official `pycocotools` API.\n\n"
        "This approach downloads only the ~241MB annotation file (not the full ~18GB image "
        "set), uses it to look up exactly which images contain our 25 target classes, and "
        "downloads only those specific images (a few thousand, not all 118K) directly from "
        "`images.cocodataset.org` in parallel.\n\n"
        "**Bonus:** the official annotation JSON includes real category names directly - no "
        "more guessing at category ID mappings. Bounding boxes are converted from COCO's "
        "official `[x, y, width, height]` format to `[x_min, y_min, x_max, y_max]` at "
        "collection time here, so every downstream notebook (02, 03, 10) keeps working "
        "completely unchanged - they already expect the corner format.\n\n"
        "Everything is saved to **Google Drive** under `SmartVisionAI/raw_data/` so that "
        "every later notebook (EDA, preprocessing, YOLO dataset prep) can reuse this exact "
        "same raw collection - even in a brand new Colab session."
    ),
    md("### 1. Install dependencies and mount Drive"),
    *setup_cells(extra_pip="pycocotools"),
    md("### 2. Project configuration"),
    config_cell(),
    md(
        "### 3. Download the official COCO 2017 annotations\n\n"
        "Only `instances_train2017.json` is extracted from the annotations zip - the "
        "captions/keypoints files inside the same archive aren\'t needed and are discarded "
        "to save disk space."
    ),
    code(
        "import os\n"
        "import zipfile\n"
        "import urllib.request\n\n"
        "COCO_ANNOTATIONS_ZIP_URL = \"http://images.cocodataset.org/annotations/annotations_trainval2017.zip\"\n"
        "COCO_LOCAL_DIR = \"/content/coco_annotations\"\n"
        "os.makedirs(COCO_LOCAL_DIR, exist_ok=True)\n\n"
        "INSTANCES_JSON_PATH = os.path.join(COCO_LOCAL_DIR, \"annotations\", \"instances_train2017.json\")\n\n"
        "if not os.path.exists(INSTANCES_JSON_PATH):\n"
        "    zip_path = os.path.join(COCO_LOCAL_DIR, \"annotations_trainval2017.zip\")\n"
        "    print(\"Downloading official COCO 2017 annotations (~241MB)...\")\n"
        "    urllib.request.urlretrieve(COCO_ANNOTATIONS_ZIP_URL, zip_path)\n"
        "    print(\"Extracting instances_train2017.json...\")\n"
        "    with zipfile.ZipFile(zip_path, \"r\") as zf:\n"
        "        zf.extract(\"annotations/instances_train2017.json\", COCO_LOCAL_DIR)\n"
        "    os.remove(zip_path)  # free disk space - we only needed one file from the zip\n"
        "    print(\"Done.\")\n"
        "else:\n"
        "    print(\"Annotations already downloaded.\")\n"
    ),
    md("### 4. Load annotations with pycocotools and resolve our 25 classes"),
    code(
        "from pycocotools.coco import COCO\n\n"
        "coco = COCO(INSTANCES_JSON_PATH)\n\n"
        "all_cats = coco.loadCats(coco.getCatIds())\n"
        "cat_id_to_name = {c[\"id\"]: c[\"name\"] for c in all_cats}\n"
        "name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}\n\n"
        "missing = [c for c in SELECTED_CLASSES if c not in name_to_cat_id]\n"
        "assert not missing, f\"Classes not found in official COCO categories: {missing}\"\n"
        "print(f\"Resolved all {len(SELECTED_CLASSES)} classes to official COCO category IDs.\")\n"
    ),
    md(
        "### 5. Determine exactly which images to download (metadata only, no downloads yet)\n\n"
        "`pycocotools` gives us direct index lookups (which images contain which "
        "categories), so - unlike streaming - we know in advance exactly which images we "
        "need, with no wasted scanning."
    ),
    code(
        "import random\n"
        "random.seed(42)\n\n"
        "collected_counts = {c: 0 for c in SELECTED_CLASSES}\n"
        "images_to_download = {}  # image_id -> {\"file\", \"width\", \"height\", \"boxes\", \"url\"}\n\n"
        "for cname in SELECTED_CLASSES:\n"
        "    cid = name_to_cat_id[cname]\n"
        "    img_ids = coco.getImgIds(catIds=[cid])\n"
        "    random.shuffle(img_ids)\n\n"
        "    for img_id in img_ids:\n"
        "        if collected_counts[cname] >= IMAGES_PER_CLASS:\n"
        "            break\n"
        "        if img_id not in images_to_download:\n"
        "            img_info = coco.loadImgs(img_id)[0]\n"
        "            ann_ids = coco.getAnnIds(imgIds=img_id)\n"
        "            anns = coco.loadAnns(ann_ids)\n\n"
        "            image_boxes = []\n"
        "            for a in anns:\n"
        "                box_cname = cat_id_to_name.get(a[\"category_id\"])\n"
        "                if box_cname in SELECTED_CLASSES:\n"
        "                    # Official COCO format is [x, y, width, height] - convert to\n"
        "                    # [x_min, y_min, x_max, y_max] here so every downstream notebook\n"
        "                    # (02, 03, 10) keeps working unchanged, since they already expect\n"
        "                    # the corner format.\n"
        "                    x, y, w, h = a[\"bbox\"]\n"
        "                    image_boxes.append({\"class\": box_cname, \"bbox\": [x, y, x + w, y + h]})\n\n"
        "            if not image_boxes:\n"
        "                continue\n\n"
        "            images_to_download[img_id] = {\n"
        "                \"width\": img_info[\"width\"],\n"
        "                \"height\": img_info[\"height\"],\n"
        "                \"boxes\": image_boxes,\n"
        "                \"url\": img_info[\"coco_url\"],\n"
        "            }\n\n"
        "        # Count this image toward every under-quota class it contains\n"
        "        for box in images_to_download[img_id][\"boxes\"]:\n"
        "            if collected_counts[box[\"class\"]] < IMAGES_PER_CLASS:\n"
        "                collected_counts[box[\"class\"]] += 1\n\n"
        "print(f\"Need to download {len(images_to_download)} unique images.\")\n"
        "for c, n in collected_counts.items():\n"
        "    flag = \"\" if n >= IMAGES_PER_CLASS else \"  <-- short\"\n"
        "    print(f\"  {c:<15} {n}{flag}\")\n"
    ),
    md(
        "### 6. Download the needed images in parallel\n\n"
        "Only the images identified above are downloaded (a few thousand, not the full "
        "118K-image train2017 set), using a thread pool since this is a network-bound task."
    ),
    code(
        "from concurrent.futures import ThreadPoolExecutor, as_completed\n"
        "from tqdm import tqdm\n"
        "import json\n\n"
        "def download_one(item):\n"
        "    img_id, meta, fname = item\n"
        "    local_path = os.path.join(RAW_IMAGES_DIR, fname)\n"
        "    try:\n"
        "        urllib.request.urlretrieve(meta[\"url\"], local_path)\n"
        "        return (img_id, fname, meta, True, None)\n"
        "    except Exception as e:\n"
        "        return (img_id, fname, meta, False, str(e))\n\n"
        "download_jobs = [\n"
        "    (img_id, meta, f\"img_{i:05d}.jpg\")\n"
        "    for i, (img_id, meta) in enumerate(images_to_download.items())\n"
        "]\n\n"
        "annotations = []\n"
        "failed = 0\n\n"
        "with ThreadPoolExecutor(max_workers=32) as executor:\n"
        "    futures = [executor.submit(download_one, job) for job in download_jobs]\n"
        "    for future in tqdm(as_completed(futures), total=len(futures)):\n"
        "        img_id, fname, meta, ok, err = future.result()\n"
        "        if ok:\n"
        "            annotations.append({\n"
        "                \"file\": fname,\n"
        "                \"width\": meta[\"width\"],\n"
        "                \"height\": meta[\"height\"],\n"
        "                \"boxes\": meta[\"boxes\"],\n"
        "            })\n"
        "        else:\n"
        "            failed += 1\n\n"
        "print(f\"Downloaded {len(annotations)} images successfully ({failed} failed).\")\n"
    ),
    md("### 7. Save annotations and a metadata summary"),
    code(
        "with open(RAW_ANNOTATIONS_PATH, \"w\") as f:\n"
        "    json.dump(annotations, f)\n\n"
        "# Recompute final per-class counts from what was actually downloaded successfully\n"
        "final_counts = {c: 0 for c in SELECTED_CLASSES}\n"
        "for ann in annotations:\n"
        "    for box in ann[\"boxes\"]:\n"
        "        final_counts[box[\"class\"]] += 1\n\n"
        "metadata = {\n"
        "    \"classes\": SELECTED_CLASSES,\n"
        "    \"images_per_class_target\": IMAGES_PER_CLASS,\n"
        "    \"actual_counts\": final_counts,\n"
        "    \"total_unique_images\": len(annotations),\n"
        "    \"source\": \"official COCO 2017 (cocodataset.org) via pycocotools\",\n"
        "}\n"
        "with open(os.path.join(RAW_DATA_DIR, \"metadata.json\"), \"w\") as f:\n"
        "    json.dump(metadata, f, indent=2)\n\n"
        "print(\"Final per-class counts (boxes actually downloaded):\")\n"
        "for c, n in final_counts.items():\n"
        "    print(f\"  {c:<15} {n}\")\n\n"
        "print(f\"\\nSaved annotations to {RAW_ANNOTATIONS_PATH}\")\n"
        "print(f\"Saved images to {RAW_IMAGES_DIR}\")\n"
    ),
    md(
        "**Next notebook:** `02_EDA.ipynb` analyzes this raw collection before we crop "
        "anything for classification or build the YOLO detection dataset."
    ),
]
save("01_Dataset_Loading.ipynb", nb1_cells)


# ===========================================================================
# NOTEBOOK 2 — EDA
# ===========================================================================
nb2_cells = [
    md(
        "# 02 - Exploratory Data Analysis\n\n"
        "**Phase 1, Step 1.2 - EDA**\n\n"
        "Run this AFTER `01_Dataset_Loading.ipynb`. Analyzes class distribution, image "
        "sizes, objects-per-image, and visualizes sample images with their annotated "
        "bounding boxes."
    ),
    md("### 1. Mount Drive and load configuration"),
    *setup_cells(extra_pip="pillow"),
    config_cell(),
    code(
        "import json\n"
        "from PIL import Image, ImageDraw\n"
        "import matplotlib.pyplot as plt\n"
        "from collections import Counter\n\n"
        "with open(RAW_ANNOTATIONS_PATH) as f:\n"
        "    annotations = json.load(f)\n"
        "print(f\"Loaded {len(annotations)} annotated images.\")\n"
    ),
    md("### 2. Class distribution (by bounding box count, not image count)"),
    code(
        "class_counts = Counter()\n"
        "for ann in annotations:\n"
        "    for box in ann[\"boxes\"]:\n"
        "        class_counts[box[\"class\"]] += 1\n\n"
        "counts_ordered = [class_counts.get(c, 0) for c in SELECTED_CLASSES]\n\n"
        "plt.figure(figsize=(12, 6))\n"
        "plt.bar(SELECTED_CLASSES, counts_ordered, color=\"#377a8a\")\n"
        "plt.xticks(rotation=75, ha=\"right\")\n"
        "plt.ylabel(\"Bounding box count\")\n"
        "plt.title(\"Class distribution across collected images\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"eda_class_distribution.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md("### 3. Image size distribution"),
    code(
        "widths = [a[\"width\"] for a in annotations]\n"
        "heights = [a[\"height\"] for a in annotations]\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "axes[0].hist(widths, bins=30, color=\"#639922\")\n"
        "axes[0].set_title(\"Image width distribution\")\n"
        "axes[1].hist(heights, bins=30, color=\"#c0622f\")\n"
        "axes[1].set_title(\"Image height distribution\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"eda_image_sizes.png\"), dpi=150)\n"
        "plt.show()\n\n"
        "print(f\"Width  - min={min(widths)} max={max(widths)} mean={sum(widths)/len(widths):.0f}\")\n"
        "print(f\"Height - min={min(heights)} max={max(heights)} mean={sum(heights)/len(heights):.0f}\")\n"
    ),
    md("### 4. Objects-per-image distribution"),
    code(
        "objs_per_image = [len(a[\"boxes\"]) for a in annotations]\n\n"
        "plt.figure(figsize=(8, 5))\n"
        "plt.hist(objs_per_image, bins=range(1, max(objs_per_image) + 2), color=\"#a3487e\", edgecolor=\"white\")\n"
        "plt.xlabel(\"Objects per image\")\n"
        "plt.ylabel(\"Number of images\")\n"
        "plt.title(\"Objects-per-image distribution\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"eda_objects_per_image.png\"), dpi=150)\n"
        "plt.show()\n\n"
        "print(f\"Average objects/image: {sum(objs_per_image)/len(objs_per_image):.2f}\")\n"
    ),
    md("### 5. Sample images with bounding box annotations"),
    code(
        "# NOTE: detection-datasets/coco stores \"bbox\" as [x_min, y_min, x_max, y_max]\n"
        "# (corner format), NOT the original COCO [x, y, width, height] format. This was\n"
        "# confirmed empirically - see the format-diagnostic check in the project README.\n"
        "import random\n"
        "random.seed(42)\n"
        "samples = random.sample(annotations, min(9, len(annotations)))\n\n"
        "fig, axes = plt.subplots(3, 3, figsize=(12, 12))\n"
        "for ax, ann in zip(axes.flat, samples):\n"
        "    img = Image.open(os.path.join(RAW_IMAGES_DIR, ann[\"file\"])).convert(\"RGB\")\n"
        "    draw = ImageDraw.Draw(img)\n"
        "    for box in ann[\"boxes\"]:\n"
        "        x1, y1, x2, y2 = box[\"bbox\"]  # x_min, y_min, x_max, y_max\n"
        "        draw.rectangle([x1, y1, x2, y2], outline=\"red\", width=3)\n"
        "        draw.text((x1, max(0, y1 - 12)), box[\"class\"], fill=\"red\")\n"
        "    ax.imshow(img)\n"
        "    ax.set_title(f\"{len(ann['boxes'])} object(s)\", fontsize=10)\n"
        "    ax.axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"eda_sample_grid.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md(
        "**Next notebook:** `03_Data_Preprocessing_Classification.ipynb` crops each "
        "object out of these images to build the classification dataset."
    ),
]
save("02_EDA.ipynb", nb2_cells)


# ===========================================================================
# NOTEBOOK 3 — Data Preprocessing for Classification
# ===========================================================================
nb3_cells = [
    md(
        "# 03 - Data Preprocessing for Classification\n\n"
        "**Phase 1, Step 1.3 - Data Preprocessing for Classification**\n\n"
        "Extracts bounding boxes from the raw annotations, crops each object into its "
        "own image, resizes to 224x224, organizes into 25 class folders, and creates "
        "train/val/test splits (70/15/15). This is the dataset the four CNN classifiers "
        "will train on."
    ),
    md("### 1. Mount Drive and load configuration"),
    *setup_cells(extra_pip="pillow"),
    config_cell(),
    code(
        "import json\n"
        "import random\n"
        "from PIL import Image\n\n"
        "random.seed(42)\n\n"
        "with open(RAW_ANNOTATIONS_PATH) as f:\n"
        "    annotations = json.load(f)\n"
        "print(f\"Loaded {len(annotations)} annotated images.\")\n"
    ),
    md(
        "### 2. Group crops by class (capped at IMAGES_PER_CLASS each)\n\n"
        "**Confirmed bbox format:** `detection-datasets/coco` stores `bbox` as "
        "`[x_min, y_min, x_max, y_max]` (corner format) - NOT the original COCO "
        "`[x, y, width, height]` format. This was verified empirically (100% of boxes "
        "are valid under the xyxy interpretation vs. only ~24% under xywh). Every crop "
        "below uses the corner-format interpretation accordingly."
    ),
    code(
        "crops_by_class = {c: [] for c in SELECTED_CLASSES}\n\n"
        "for ann in annotations:\n"
        "    img_path = os.path.join(RAW_IMAGES_DIR, ann[\"file\"])\n"
        "    img_w, img_h = ann[\"width\"], ann[\"height\"]\n"
        "    for box in ann[\"boxes\"]:\n"
        "        cname = box[\"class\"]\n"
        "        if len(crops_by_class[cname]) >= IMAGES_PER_CLASS:\n"
        "            continue\n"
        "        x1, y1, x2, y2 = box[\"bbox\"]  # x_min, y_min, x_max, y_max\n"
        "        # Defensive clamp in case of minor annotation overshoot at image edges\n"
        "        x1 = max(0, min(x1, img_w))\n"
        "        y1 = max(0, min(y1, img_h))\n"
        "        x2 = max(0, min(x2, img_w))\n"
        "        y2 = max(0, min(y2, img_h))\n"
        "        if x2 <= x1 or y2 <= y1:\n"
        "            continue  # degenerate box, skip\n"
        "        crops_by_class[cname].append({\"img_path\": img_path, \"bbox\": (x1, y1, x2, y2)})\n\n"
        "for c, items in crops_by_class.items():\n"
        "    print(f\"{c:<15} {len(items)} crops\")\n"
    ),
    md("### 3. Crop, resize, split 70/15/15, and save into class folders"),
    code(
        "import shutil as _shutil\n"
        "# Clear any stale crops from a previous run before rebuilding, so leftover\n"
        "# images from an earlier (possibly differently-labeled) run never mix in.\n"
        "_shutil.rmtree(CLASSIFICATION_DIR, ignore_errors=True)\n\n"
        "for split_dir in (CLASSIFICATION_TRAIN_DIR, CLASSIFICATION_VAL_DIR, CLASSIFICATION_TEST_DIR):\n"
        "    for cname in SELECTED_CLASSES:\n"
        "        os.makedirs(os.path.join(split_dir, safe_name(cname)), exist_ok=True)\n\n"
        "split_dirs = {\"train\": CLASSIFICATION_TRAIN_DIR, \"val\": CLASSIFICATION_VAL_DIR, \"test\": CLASSIFICATION_TEST_DIR}\n\n"
        "for cname, items in crops_by_class.items():\n"
        "    random.shuffle(items)\n"
        "    n = len(items)\n"
        "    n_train = int(n * TRAIN_SPLIT)\n"
        "    n_val = int(n * VAL_SPLIT)\n"
        "    splits = {\n"
        "        \"train\": items[:n_train],\n"
        "        \"val\": items[n_train:n_train + n_val],\n"
        "        \"test\": items[n_train + n_val:],\n"
        "    }\n"
        "    for split_name, split_items in splits.items():\n"
        "        out_dir = os.path.join(split_dirs[split_name], safe_name(cname))\n"
        "        for i, item in enumerate(split_items):\n"
        "            x1, y1, x2, y2 = item[\"bbox\"]  # x_min, y_min, x_max, y_max\n"
        "            img = Image.open(item[\"img_path\"]).convert(\"RGB\")\n"
        "            crop = img.crop((x1, y1, x2, y2)).resize((CLS_IMG_SIZE, CLS_IMG_SIZE))\n"
        "            crop.save(os.path.join(out_dir, f\"{safe_name(cname)}_{split_name}_{i:04d}.jpg\"))\n"
        "    print(f\"{cname:<15} train={len(splits['train']):3d}  val={len(splits['val']):3d}  test={len(splits['test']):3d}\")\n"
    ),
    md("### 4. Sanity check: view a few processed crops"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import glob\n\n"
        "fig, axes = plt.subplots(1, 5, figsize=(15, 3))\n"
        "for ax, cname in zip(axes, SELECTED_CLASSES[:5]):\n"
        "    files = glob.glob(os.path.join(CLASSIFICATION_TRAIN_DIR, safe_name(cname), \"*.jpg\"))\n"
        "    if files:\n"
        "        ax.imshow(Image.open(files[0]))\n"
        "    ax.set_title(cname)\n"
        "    ax.axis(\"off\")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    md(
        "**Next notebook:** `04_Data_Augmentation.ipynb` visualizes the augmentation "
        "pipeline that each training notebook applies on top of this data."
    ),
]
save("03_Data_Preprocessing_Classification.ipynb", nb3_cells)


# ===========================================================================
# NOTEBOOK 4 — Data Augmentation (demo)
# ===========================================================================
nb4_cells = [
    md(
        "# 04 - Data Augmentation\n\n"
        "**Phase 1, Step 1.4 - Data Augmentation**\n\n"
        "Defines and visualizes the augmentation pipeline used by every classifier "
        "training notebook: random horizontal flip, rotation (+/-15 degrees), "
        "brightness adjustment (+/-20%), contrast adjustment, random zoom, and color "
        "jittering. This notebook is a visual sanity check - the actual augmentation "
        "layer is re-declared inside each training notebook so they stay fully "
        "self-contained."
    ),
    md("### 1. Mount Drive and load configuration"),
    *setup_cells(extra_pip="tensorflow pillow"),
    config_cell(),
    md("### 2. Define the augmentation pipeline"),
    code(
        "import tensorflow as tf\n"
        "from tensorflow.keras import layers\n\n"
        "augmentation_pipeline = tf.keras.Sequential([\n"
        "    layers.RandomFlip(\"horizontal\"),\n"
        "    layers.RandomRotation(0.04),     # ~ +/-15 degrees\n"
        "    layers.RandomBrightness(0.2),    # +/-20%\n"
        "    layers.RandomContrast(0.2),\n"
        "    layers.RandomZoom(0.1),\n"
        "], name=\"augmentation_pipeline\")\n\n"
        "print(augmentation_pipeline.summary())\n"
    ),
    md("### 3. Visualize augmented versions of a sample image"),
    code(
        "import glob\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n\n"
        "sample_class = safe_name(SELECTED_CLASSES[0])\n"
        "files = glob.glob(os.path.join(CLASSIFICATION_TRAIN_DIR, sample_class, \"*.jpg\"))\n"
        "assert files, \"Run 03_Data_Preprocessing_Classification.ipynb first.\"\n\n"
        "img = np.array(Image.open(files[0]).convert(\"RGB\")).astype(\"float32\")\n"
        "batch = np.expand_dims(img, 0)\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "axes[0, 0].imshow(img.astype(\"uint8\"))\n"
        "axes[0, 0].set_title(\"Original\")\n"
        "axes[0, 0].axis(\"off\")\n\n"
        "for ax in axes.flat[1:]:\n"
        "    augmented = augmentation_pipeline(batch, training=True)[0].numpy()\n"
        "    augmented = np.clip(augmented, 0, 255).astype(\"uint8\")\n"
        "    ax.imshow(augmented)\n"
        "    ax.set_title(\"Augmented\")\n"
        "    ax.axis(\"off\")\n\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"augmentation_examples.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md(
        "**Next notebooks:** `05_Train_VGG16.ipynb`, `06_Train_ResNet50.ipynb`, "
        "`07_Train_MobileNetV2.ipynb`, `08_Train_EfficientNetB0.ipynb` - each trains "
        "one classifier independently, reusing this same augmentation pipeline "
        "definition inline."
    ),
]
save("04_Data_Augmentation.ipynb", nb4_cells)


# ===========================================================================
# Shared code block for classifier training notebooks
# ===========================================================================

def classifier_training_notebook(model_key, model_name, keras_app_module, keras_app_class,
                                  fine_tune_layers, expected_range, phase_step, notes):
    """Build one self-contained classifier training notebook.

    Simplified, proven-working architecture (replaces the earlier progressive-resizing
    approach, which destabilized every model's pretrained features via an abrupt
    224px->384px jump). This version trains and evaluates entirely at a single
    resolution (CLS_IMG_SIZE, 224px), uses a lean GAP -> Dropout -> Dense head, and
    only fine-tunes the two architectures that benefit from it (ResNet50,
    EfficientNetB0) - each still gets a safety net that keeps Stage 1's weights if
    Stage 2 doesn't actually improve val_accuracy.
    """

    data_pipeline_code = f'''\
import tensorflow as tf
from tensorflow.keras import layers

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    CLASSIFICATION_TRAIN_DIR, image_size=(CLS_IMG_SIZE, CLS_IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    CLASSIFICATION_VAL_DIR, image_size=(CLS_IMG_SIZE, CLS_IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    CLASSIFICATION_TEST_DIR, image_size=(CLS_IMG_SIZE, CLS_IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
)

class_names = train_ds_raw.class_names
print("Classes found:", len(class_names))

# Augmentation pipeline (matches the project brief: flip, rotation, brightness,
# contrast, zoom). Gaussian noise was dropped - it wasn't in the proven-working
# reference and added complexity without a demonstrated benefit.
augmentation_pipeline = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),        # ~ +/-15 degrees
    layers.RandomBrightness(0.2),       # +/-20%
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.1),             # 10% zoom
], name="augmentation_pipeline")

from tensorflow.keras.applications import {keras_app_module}

def prep_train(x, y):
    x = augmentation_pipeline(x, training=True)
    return {keras_app_module}.preprocess_input(x), y

def prep_eval(x, y):
    return {keras_app_module}.preprocess_input(x), y

train_ds = train_ds_raw.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_raw.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds_raw.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
'''

    build_model_code = f'''\
# Lean classification head: GlobalAveragePooling2D -> Dropout -> Dense.
# The earlier version added an extra Dense(256)+L2+BatchNormalization bottleneck
# before this; dropping it (matching the proven-working reference) means fewer
# parameters to overfit on a modest dataset, relying on Dropout alone plus the
# frozen backbone for regularization.
NUM_CLASSES = len(class_names)
DROPOUT_RATE = 0.3

def build_{model_key}(unfreeze_last_n=0):
    base_model = {keras_app_module}.{keras_app_class}(
        include_top=False, weights="imagenet", input_shape=(CLS_IMG_SIZE, CLS_IMG_SIZE, 3)
    )
    if unfreeze_last_n > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False
        # Keep BatchNorm frozen even within the fine-tuned range - standard Keras
        # fine-tuning practice, prevents small-batch statistics from destabilizing
        # the model regardless of which layers are otherwise trainable.
        for layer in base_model.layers[-unfreeze_last_n:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(CLS_IMG_SIZE, CLS_IMG_SIZE, 3))
    # training=False always - keeps BatchNorm in inference mode throughout, which
    # is what actually keeps fine-tuning stable (mixing this up was the root cause
    # of the earlier progressive-resizing collapse).
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

model = build_{model_key}(unfreeze_last_n=0)
model.summary()
'''

    weights_path_stage1 = f'STAGE1_WEIGHTS_PATH = os.path.join(MODELS_DIR, "{model_key}_stage1.keras")'
    weights_path_final = f'WEIGHTS_PATH = os.path.join(MODELS_DIR, "{model_key}_best.keras")'

    stage1_code = f'''\
{weights_path_stage1}

from tensorflow.keras import callbacks

# Plain Adam, EarlyStopping on val_accuracy - matches the configuration that was
# empirically verified to work well on this dataset.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cbs_stage1 = [
    callbacks.ModelCheckpoint(STAGE1_WEIGHTS_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
]

history_stage1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs_stage1)
'''

    stage2_code = f'''\
{weights_path_final}

# Fine-tune the last {fine_tune_layers} base layers - SAME resolution as Stage 1
# (no progressive resizing). This is the key fix: fine-tuning at the same
# resolution the base was pretrained at keeps BatchNorm statistics and feature
# scales consistent, instead of forcing the model to re-equilibrate to a new
# input size at the same time it's adapting pretrained weights.
model = build_{model_key}(unfreeze_last_n={fine_tune_layers})
model.load_weights(STAGE1_WEIGHTS_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cbs_stage2 = [
    callbacks.ModelCheckpoint(WEIGHTS_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
]

history_stage2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cbs_stage2)

# Safety net: only keep Stage 2 if it actually beat Stage 1. With same-resolution
# fine-tuning this should succeed far more often than the old progressive-resizing
# approach did, but the safety net stays as cheap insurance either way.
stage1_best_val_acc = max(history_stage1.history["val_accuracy"])
stage2_best_val_acc = max(history_stage2.history["val_accuracy"])

if stage2_best_val_acc > stage1_best_val_acc:
    print(f"Stage 2 improved val_accuracy ({{stage1_best_val_acc:.4f}} -> {{stage2_best_val_acc:.4f}}). Keeping Stage 2 model.")
else:
    print(f"Stage 2 did NOT improve val_accuracy (Stage 1: {{stage1_best_val_acc:.4f}}, Stage 2: {{stage2_best_val_acc:.4f}}). Reverting to Stage 1 model.")
    import shutil as _shutil
    _shutil.copy(STAGE1_WEIGHTS_PATH, WEIGHTS_PATH)
    model = tf.keras.models.load_model(WEIGHTS_PATH)
'''

    frozen_only_save_code = f'''\
{weights_path_final}
# {model_name} stays frozen-only (matches the project brief) - Stage 1's checkpoint
# IS the final model. Copy it to the canonical WEIGHTS_PATH name for consistency
# with models that do have a Stage 2.
import shutil as _shutil
_shutil.copy(STAGE1_WEIGHTS_PATH, WEIGHTS_PATH)
model = tf.keras.models.load_model(WEIGHTS_PATH)
'''

    eval_code = f'''\
import time
import numpy as np
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

y_true, y_pred = [], []
start = time.time()
n_images = 0
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    n_images += x_batch.shape[0]
elapsed = time.time() - start
ms_per_image = (elapsed / max(n_images, 1)) * 1000

accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)

metrics = {{
    "model": "{model_name}",
    "test_accuracy": round(accuracy, 4),
    "precision_macro": round(float(precision), 4),
    "recall_macro": round(float(recall), 4),
    "f1_macro": round(float(f1), 4),
    "avg_inference_ms": round(ms_per_image, 2),
    "model_size_mb": round(size_mb, 2),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "class_names": class_names,
}}

metrics_path = os.path.join(OUTPUTS_DIR, "metrics_{model_key}.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Test accuracy: {{accuracy:.4f}}  (expected range: {expected_range})")
print(f"Precision (macro): {{precision:.4f}}   Recall (macro): {{recall:.4f}}   F1 (macro): {{f1:.4f}}")
print(f"Inference:     {{ms_per_image:.1f}} ms/image")
print(f"Model size:    {{size_mb:.1f}} MB")
print(f"Metrics saved to {{metrics_path}}")
'''

    plot_code = '''\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
histories = [history_stage1] + ([history_stage2] if "history_stage2" in dir() else [])
acc, val_acc, loss, val_loss = [], [], [], []
for h in histories:
    acc += h.history["accuracy"]
    val_acc += h.history["val_accuracy"]
    loss += h.history["loss"]
    val_loss += h.history["val_loss"]

axes[0].plot(acc, label="train"); axes[0].plot(val_acc, label="val")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(loss, label="train"); axes[1].plot(val_loss, label="val")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "training_curves_{}.png".format("{model_key}")), dpi=150)
plt.show()

train_val_gap = max(acc) - max(val_acc)
print(f"Train/val accuracy gap: {train_val_gap*100:.1f} points")
'''

    cells = [
        md(
            f"# {phase_step} - {model_name}\n\n"
            f"Trains **{model_name}** via transfer learning on the classification dataset "
            f"built in `03_Data_Preprocessing_Classification.ipynb`, using a **lean, "
            f"single-resolution architecture** (GlobalAveragePooling2D -> Dropout(0.3) -> "
            f"Dense) - matching the configuration empirically verified to reach strong "
            f"accuracy on this dataset, in place of an earlier progressive-resizing "
            f"approach that destabilized every model's pretrained features.\n\n"
            f"{notes}\n\n"
            f"Expected performance: **{expected_range}**."
        ),
        md("### 1. Install dependencies and mount Drive"),
        *setup_cells(extra_pip="tensorflow"),
        md("### 2. Project configuration"),
        config_cell(),
        md("### 3. Build the data pipeline (single resolution, brief-matching augmentation)"),
        code(data_pipeline_code),
        md(f"### 4. Build the {model_name} model (lean head)"),
        code(build_model_code),
        md("### 5. Stage 1 - train the new classification head (frozen base)"),
        code(stage1_code),
    ]

    if fine_tune_layers > 0:
        cells += [
            md(f"### 6. Stage 2 - fine-tune the last {fine_tune_layers} base layers (same resolution)"),
            code(stage2_code),
        ]
    else:
        cells += [
            md(f"### 6. {model_name} stays frozen-only"),
            code(frozen_only_save_code),
        ]

    cells += [
        md("### 7. Evaluate on the held-out test set"),
        code(eval_code),
        md("### 8. Plot training curves and train/val gap"),
        code(plot_code.replace('"{model_key}"', f'"{model_key}"')),
        md(
            "**Next:** once all 4 classifiers have been trained, run "
            "`09_Compare_Classification_Models.ipynb` to compare them and pick the best one."
        ),
    ]
    return cells


save("05_Train_VGG16.ipynb", classifier_training_notebook(
    model_key="vgg16", model_name="VGG16", keras_app_module="vgg16", keras_app_class="VGG16",
    fine_tune_layers=0, expected_range="80-85%",
    phase_step="05 - Train Classifier",
    notes="VGG16's convolutional base stays **fully frozen** (matches the project brief) - only the new head is trained.",
))

save("06_Train_ResNet50.ipynb", classifier_training_notebook(
    model_key="resnet50", model_name="ResNet50", keras_app_module="resnet50", keras_app_class="ResNet50",
    fine_tune_layers=20, expected_range="85-90%",
    phase_step="06 - Train Classifier",
    notes="The last **20 layers** of ResNet50's base are unfrozen and fine-tuned in Stage 2 at the same resolution, low learning rate.",
))

save("07_Train_MobileNetV2.ipynb", classifier_training_notebook(
    model_key="mobilenetv2", model_name="MobileNetV2", keras_app_module="mobilenet_v2", keras_app_class="MobileNetV2",
    fine_tune_layers=0, expected_range="82-87%",
    phase_step="07 - Train Classifier",
    notes="MobileNetV2's base stays **fully frozen** (matches the project brief), optimizing this model for inference speed.",
))

save("08_Train_EfficientNetB0.ipynb", classifier_training_notebook(
    model_key="efficientnetb0", model_name="EfficientNetB0", keras_app_module="efficientnet", keras_app_class="EfficientNetB0",
    fine_tune_layers=30, expected_range="88-93%",
    phase_step="08 - Train Classifier",
    notes="The last **30 layers** of EfficientNetB0's base are unfrozen and fine-tuned in Stage 2 at the same resolution, aiming for the best overall accuracy.",
))

# ===========================================================================
# NOTEBOOK 9 — Compare Classification Models
# ===========================================================================
nb9_cells = [
    md(
        "# 09 - Compare Classification Models\n\n"
        "**Phase 2, Step 2.5 - Model Comparison & Selection**\n\n"
        "Run this AFTER training all 4 classifiers (notebooks 05-08). Loads each "
        "model's `outputs/metrics_<key>.json`, builds comparison charts, and "
        "recommends a best model based on an accuracy/speed tradeoff score."
    ),
    md("### 1. Mount Drive and load configuration"),
    *setup_cells(),
    config_cell(),
    code(
        "import json\n"
        "import glob\n\n"
        "MODEL_KEYS = [\"vgg16\", \"resnet50\", \"mobilenetv2\", \"efficientnetb0\"]\n\n"
        "all_metrics = {}\n"
        "for key in MODEL_KEYS:\n"
        "    path = os.path.join(OUTPUTS_DIR, f\"metrics_{key}.json\")\n"
        "    if os.path.exists(path):\n"
        "        with open(path) as f:\n"
        "            all_metrics[key] = json.load(f)\n"
        "    else:\n"
        "        print(f\"WARNING: {path} not found - train that model first.\")\n\n"
        "print(f\"Loaded metrics for: {list(all_metrics.keys())}\")\n"
    ),
    md("### 2. Bar chart: accuracy, precision, recall, F1 across models"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n\n"
        "names = [m[\"model\"] for m in all_metrics.values()]\n"
        "metrics_to_plot = [\"test_accuracy\", \"precision_macro\", \"recall_macro\", \"f1_macro\"]\n"
        "x = np.arange(len(names))\n"
        "width = 0.2\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "for i, metric in enumerate(metrics_to_plot):\n"
        "    values = [m[metric] for m in all_metrics.values()]\n"
        "    plt.bar(x + i * width, values, width, label=metric)\n"
        "plt.xticks(x + width * 1.5, names)\n"
        "plt.ylabel(\"Score\")\n"
        "plt.ylim(0, 1.0)\n"
        "plt.title(\"Classification model comparison\")\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"model_comparison_bars.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md("### 3. Accuracy vs. inference speed tradeoff"),
    code(
        "plt.figure(figsize=(8, 6))\n"
        "for key, m in all_metrics.items():\n"
        "    plt.scatter(m[\"avg_inference_ms\"], m[\"test_accuracy\"], s=120)\n"
        "    plt.annotate(m[\"model\"], (m[\"avg_inference_ms\"], m[\"test_accuracy\"]), textcoords=\"offset points\", xytext=(8, 5))\n"
        "plt.xlabel(\"Avg inference time (ms/image)\")\n"
        "plt.ylabel(\"Test accuracy\")\n"
        "plt.title(\"Accuracy vs. inference speed tradeoff\")\n"
        "plt.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"accuracy_vs_speed.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md("### 4. Confusion matrices"),
    code(
        "n = len(all_metrics)\n"
        "fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))\n"
        "if n == 1:\n"
        "    axes = [axes]\n"
        "for ax, (key, m) in zip(axes, all_metrics.items()):\n"
        "    cm = np.array(m[\"confusion_matrix\"])\n"
        "    ax.imshow(cm, cmap=\"Blues\")\n"
        "    ax.set_title(m[\"model\"])\n"
        "    ax.set_xlabel(\"Predicted\")\n"
        "    ax.set_ylabel(\"Actual\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"confusion_matrices.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md("### 5. Select the best model (weighted accuracy + speed score)"),
    code(
        "def select_best(all_metrics, accuracy_weight=0.7, speed_weight=0.3):\n"
        "    accs = [m[\"test_accuracy\"] for m in all_metrics.values()]\n"
        "    speeds = [m[\"avg_inference_ms\"] for m in all_metrics.values()]\n"
        "    acc_min, acc_max = min(accs), max(accs)\n"
        "    speed_min, speed_max = min(speeds), max(speeds)\n\n"
        "    scores = {}\n"
        "    for key, m in all_metrics.items():\n"
        "        norm_acc = (m[\"test_accuracy\"] - acc_min) / (acc_max - acc_min + 1e-9)\n"
        "        norm_speed = 1 - (m[\"avg_inference_ms\"] - speed_min) / (speed_max - speed_min + 1e-9)\n"
        "        scores[key] = accuracy_weight * norm_acc + speed_weight * norm_speed\n"
        "    best_key = max(scores, key=scores.get)\n"
        "    return best_key, scores\n\n"
        "best_key, scores = select_best(all_metrics)\n\n"
        "summary = {\n"
        "    \"scores\": scores,\n"
        "    \"recommended_model\": best_key,\n"
        "    \"recommended_model_name\": all_metrics[best_key][\"model\"],\n"
        "    \"reasoning\": \"Weighted 70% test accuracy / 30% inference speed.\",\n"
        "}\n"
        "with open(os.path.join(OUTPUTS_DIR, \"model_selection_summary.json\"), \"w\") as f:\n"
        "    json.dump(summary, f, indent=2)\n\n"
        "print(\"Model comparison summary:\")\n"
        "for key, m in all_metrics.items():\n"
        "    print(f\"  {m['model']:<15} acc={m['test_accuracy']:.4f}  speed={m['avg_inference_ms']:.1f}ms  score={scores[key]:.4f}\")\n"
        "print(f\"\\nRecommended model: {all_metrics[best_key]['model']}\")\n"
    ),
    md(
        "**Next notebook:** `10_YOLO_Dataset_Preparation.ipynb` builds the detection "
        "dataset so we can train YOLOv8 for multi-object localization."
    ),
]
save("09_Compare_Classification_Models.ipynb", nb9_cells)


# ===========================================================================
# NOTEBOOK 10 — YOLO Dataset Preparation
# ===========================================================================
nb10_cells = [
    md(
        "# 10 - YOLO Dataset Preparation\n\n"
        "**Phase 3, Step 3.2 - Dataset Preparation for Detection**\n\n"
        "Reuses the raw collection from `01_Dataset_Loading.ipynb` (full images + all "
        "matching boxes) and converts it into YOLO format: normalized "
        "`class x_center y_center width height` label files, an images/labels train/val/test "
        "split, and a `data.yaml` config for Ultralytics YOLOv8."
    ),
    md("### 1. Mount Drive and load configuration"),
    *setup_cells(extra_pip="pyyaml pillow"),
    config_cell(),
    code(
        "import json\n"
        "import random\n"
        "import shutil\n"
        "import yaml\n\n"
        "random.seed(42)\n\n"
        "with open(RAW_ANNOTATIONS_PATH) as f:\n"
        "    annotations = json.load(f)\n"
        "print(f\"Loaded {len(annotations)} annotated images.\")\n"
    ),
    md(
        "### 2. Convert corner-format boxes to normalized YOLO format\n\n"
        "**Confirmed bbox format:** `detection-datasets/coco` stores `bbox` as "
        "`[x_min, y_min, x_max, y_max]` (corner format), verified empirically - 100% of "
        "boxes are valid under this interpretation vs. only ~24% under the originally "
        "assumed `[x, y, width, height]` format. Getting this wrong is what caused "
        "YOLOv8 to discard the majority of images as \"corrupt\" (out-of-bounds "
        "normalized coordinates) in earlier runs."
    ),
    code(
        "import shutil as _shutil\n"
        "# Clear any stale detection data from a previous run before rebuilding.\n"
        "# Without this, re-running this notebook after 01_Dataset_Loading collected a\n"
        "# different number of images would leave old files mixed in with new ones -\n"
        "# including old files from before any bbox-format fix, silently corrupting results.\n"
        "_shutil.rmtree(DETECTION_IMAGES_DIR, ignore_errors=True)\n"
        "_shutil.rmtree(DETECTION_LABELS_DIR, ignore_errors=True)\n"
        "os.makedirs(DETECTION_IMAGES_DIR, exist_ok=True)\n"
        "os.makedirs(DETECTION_LABELS_DIR, exist_ok=True)\n\n"
        "yolo_records = []  # (image_filename, list_of_label_lines)\n"
        "skipped_degenerate = 0\n\n"
        "for ann in annotations:\n"
        "    w, h = ann[\"width\"], ann[\"height\"]\n"
        "    label_lines = []\n"
        "    for box in ann[\"boxes\"]:\n"
        "        x1, y1, x2, y2 = box[\"bbox\"]  # x_min, y_min, x_max, y_max\n"
        "        # Defensive clamp in case of minor annotation overshoot at image edges\n"
        "        x1 = max(0.0, min(x1, w))\n"
        "        y1 = max(0.0, min(y1, h))\n"
        "        x2 = max(0.0, min(x2, w))\n"
        "        y2 = max(0.0, min(y2, h))\n"
        "        bw, bh = x2 - x1, y2 - y1\n"
        "        if bw <= 0 or bh <= 0:\n"
        "            skipped_degenerate += 1\n"
        "            continue\n\n"
        "        x_center = min(max((x1 + bw / 2) / w, 0.0), 1.0)\n"
        "        y_center = min(max((y1 + bh / 2) / h, 0.0), 1.0)\n"
        "        norm_w = min(max(bw / w, 0.0), 1.0)\n"
        "        norm_h = min(max(bh / h, 0.0), 1.0)\n"
        "        cls_id = CLASS_TO_IDX[box[\"class\"]]\n"
        "        label_lines.append(f\"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\")\n"
        "    yolo_records.append((ann[\"file\"], label_lines))\n\n"
        "print(f\"Prepared YOLO labels for {len(yolo_records)} images.\")\n"
        "print(f\"Skipped {skipped_degenerate} degenerate boxes (zero or negative width/height).\")\n"
    ),
    md("### 3. Split into train/val/test and write images + label files"),
    code(
        "random.shuffle(yolo_records)\n"
        "n = len(yolo_records)\n"
        "n_train = int(n * TRAIN_SPLIT)\n"
        "n_val = int(n * VAL_SPLIT)\n"
        "split_map = {\n"
        "    \"train\": yolo_records[:n_train],\n"
        "    \"val\": yolo_records[n_train:n_train + n_val],\n"
        "    \"test\": yolo_records[n_train + n_val:],\n"
        "}\n\n"
        "for split_name, records in split_map.items():\n"
        "    img_split_dir = os.path.join(DETECTION_IMAGES_DIR, split_name)\n"
        "    lbl_split_dir = os.path.join(DETECTION_LABELS_DIR, split_name)\n"
        "    os.makedirs(img_split_dir, exist_ok=True)\n"
        "    os.makedirs(lbl_split_dir, exist_ok=True)\n"
        "    for fname, label_lines in records:\n"
        "        stem = fname.rsplit(\".\", 1)[0]\n"
        "        shutil.copy(os.path.join(RAW_IMAGES_DIR, fname), os.path.join(img_split_dir, fname))\n"
        "        with open(os.path.join(lbl_split_dir, f\"{stem}.txt\"), \"w\") as f:\n"
        "            f.write(\"\\n\".join(label_lines))\n"
        "    print(f\"{split_name}: {len(records)} images\")\n"
    ),
    md("### 4. Write the data.yaml config for YOLOv8"),
    code(
        "data_yaml = {\n"
        "    \"path\": DETECTION_DIR,\n"
        "    \"train\": \"images/train\",\n"
        "    \"val\": \"images/val\",\n"
        "    \"test\": \"images/test\",\n"
        "    \"nc\": len(SELECTED_CLASSES),\n"
        "    \"names\": SELECTED_CLASSES,\n"
        "}\n"
        "with open(DETECTION_YAML_PATH, \"w\") as f:\n"
        "    yaml.dump(data_yaml, f, sort_keys=False)\n\n"
        "print(f\"Wrote {DETECTION_YAML_PATH}\")\n"
        "print(open(DETECTION_YAML_PATH).read())\n"
    ),
    md(
        "**Next notebook:** `11_Train_YOLO.ipynb` fine-tunes YOLOv8 on this exact "
        "dataset and directory structure."
    ),
]
save("10_YOLO_Dataset_Preparation.ipynb", nb10_cells)


# ===========================================================================
# NOTEBOOK 11 — Train YOLO
# ===========================================================================
nb11_cells = [
    md(
        "# 11 - Train YOLOv8\n\n"
        "**Phase 3, Steps 3.1, 3.3, 3.4 - YOLO Setup, Training, and Evaluation**\n\n"
        "Fine-tunes a pretrained YOLOv8 checkpoint on the 25-class detection dataset "
        "built in `10_YOLO_Dataset_Preparation.ipynb`, then evaluates mAP@0.5, "
        "mAP@0.5:0.95, precision, recall, and inference speed.\n\n"
        "**Tip:** In Colab, go to Runtime > Change runtime type > GPU (T4) before "
        "running this notebook."
    ),
    md("### 1. Install Ultralytics and mount Drive"),
    *setup_cells(extra_pip="ultralytics"),
    config_cell(),
    code(
        "import torch\n"
        "print(\"CUDA available:\", torch.cuda.is_available())\n"
        "print(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")\n"
    ),
    md("### 2. Load a pretrained YOLOv8 checkpoint"),
    code(
        "from ultralytics import YOLO\n\n"
        "# yolov8n.pt = nano (fastest to train/run)\n"
        "# yolov8s.pt = small (more accurate, still fast)\n"
        "YOLO_BASE_WEIGHTS = \"yolov8n.pt\"\n"
        "model = YOLO(YOLO_BASE_WEIGHTS)\n"
    ),
    md("### 3. Fine-tune on the 25-class subset"),
    code(
        "results = model.train(\n"
        "    data=DETECTION_YAML_PATH,\n"
        "    epochs=50,\n"
        "    imgsz=YOLO_IMG_SIZE,\n"
        "    batch=16,\n"
        "    project=MODELS_DIR,\n"
        "    name=\"yolo_smartvision\",\n"
        "    patience=10,\n"
        "    optimizer=\"auto\",\n"
        "    exist_ok=True,\n"
        ")\n\n"
        "BEST_WEIGHTS_PATH = os.path.join(MODELS_DIR, \"yolo_smartvision\", \"weights\", \"best.pt\")\n"
        "print(\"Best weights saved to:\", BEST_WEIGHTS_PATH)\n"
    ),
    md("### 4. Evaluate on the validation set"),
    code(
        "import json\n\n"
        "eval_model = YOLO(BEST_WEIGHTS_PATH)\n"
        "metrics = eval_model.val(data=DETECTION_YAML_PATH)\n\n"
        "inference_ms = None\n"
        "try:\n"
        "    inference_ms = metrics.speed.get(\"inference\")\n"
        "except Exception:\n"
        "    pass\n\n"
        "summary = {\n"
        "    \"map50\": float(metrics.box.map50),\n"
        "    \"map50_95\": float(metrics.box.map),\n"
        "    \"precision\": float(metrics.box.mp),\n"
        "    \"recall\": float(metrics.box.mr),\n"
        "    \"inference_ms_per_image\": inference_ms,\n"
        "    \"fps_estimate\": round(1000 / inference_ms, 1) if inference_ms else None,\n"
        "    \"weights_path\": BEST_WEIGHTS_PATH,\n"
        "}\n\n"
        "with open(os.path.join(OUTPUTS_DIR, \"yolo_metrics.json\"), \"w\") as f:\n"
        "    json.dump(summary, f, indent=2)\n\n"
        "print(\"YOLO evaluation summary:\")\n"
        "print(f\"  mAP@0.5      : {summary['map50']:.4f}  (target: >0.75)\")\n"
        "print(f\"  mAP@0.5:0.95 : {summary['map50_95']:.4f}\")\n"
        "print(f\"  Precision    : {summary['precision']:.4f}\")\n"
        "print(f\"  Recall       : {summary['recall']:.4f}\")\n"
        "if summary[\"fps_estimate\"]:\n"
        "    print(f\"  Est. FPS     : {summary['fps_estimate']}\")\n"
    ),
    md("### 5. Visualize predictions on a few validation images"),
    code(
        "import glob\n"
        "from PIL import Image\n"
        "import matplotlib.pyplot as plt\n\n"
        "val_images = glob.glob(os.path.join(DETECTION_IMAGES_DIR, \"val\", \"*.jpg\"))[:6]\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n\n"
        "for ax, img_path in zip(axes.flat, val_images):\n"
        "    result = eval_model.predict(source=img_path, conf=0.5, verbose=False)[0]\n"
        "    annotated = result.plot()  # returns a numpy array (BGR)\n"
        "    ax.imshow(annotated[:, :, ::-1])  # BGR -> RGB\n"
        "    ax.axis(\"off\")\n\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUTPUTS_DIR, \"yolo_sample_predictions.png\"), dpi=150)\n"
        "plt.show()\n"
    ),
    md(
        "**Next notebook:** `12_Inference_Pipeline_Demo.ipynb` chains this YOLO model "
        "with a trained classifier for the full detect-then-verify pipeline, before "
        "wiring both into the Streamlit app."
    ),
]
save("11_Train_YOLO.ipynb", nb11_cells)


# ===========================================================================
# NOTEBOOK 12 — Inference Pipeline Demo
# ===========================================================================
nb12_cells = [
    md(
        "# 12 - Inference Pipeline Demo\n\n"
        "**Phase 4 - Model Integration & Pipeline Development**\n\n"
        "Unified pipeline tying the whole SmartVision AI system together:\n\n"
        "1. Load YOLOv8 from `best.pt` and the champion classifier from its native `.keras` file\n"
        "2. Run YOLOv8 detection on a test image\n"
        "3. Apply NMS + confidence filtering\n"
        "4. Parse each box as `[x_min, y_min, x_max, y_max]` (matching the corner format "
        "confirmed for `detection-datasets/coco` - see project report)\n"
        "5. Crop with defensive padding, resize, and scale pixels for the classifier\n"
        "6. Re-verify the crop with the EfficientNetB0 `.keras` model\n"
        "7. Print the combined YOLO + classifier verdict for each detection\n\n"
        "This is the same logic that powers the Streamlit app's Object Detection page."
    ),
    md("### 1. Install dependencies and mount Drive"),
    *setup_cells(extra_pip="ultralytics tensorflow pillow"),
    config_cell(),
    md(
        "### 2. Load the YOLOv8 detector (`.pt`) and the champion classifier (`.keras`)\n\n"
        "YOLOv8 weights stay in Ultralytics' native PyTorch `.pt` format - that's the format "
        "the framework trains, saves, and reloads with, and it's what Ultralytics' own "
        "`YOLO(path)` loader expects. The classifier uses Keras's native `.keras` format, "
        "which serializes the full model (architecture + regularizers + weights) as a single "
        "file - no `custom_objects` argument needed on reload, since every layer used in the "
        "hardened head (GlobalAveragePooling2D, Dense with L2 regularizer, BatchNormalization, "
        "Dropout) is a standard Keras layer with a built-in `get_config()`."
    ),
    code(
        "from ultralytics import YOLO\n"
        "import tensorflow as tf\n"
        "import numpy as np\n"
        "from PIL import Image, ImageDraw, ImageFont\n"
        "import json\n\n"
        "# --- Step 1: load YOLOv8 from best.pt ---\n"
        "YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, \"yolo_smartvision\", \"weights\", \"best.pt\")\n"
        "yolo_model = YOLO(YOLO_WEIGHTS_PATH)\n\n"
        "# --- Step 1: load the champion classifier from its native .keras file ---\n"
        "# Whichever model 09_Compare_Classification_Models.ipynb recommended is loaded here.\n"
        "# Falls back to EfficientNetB0 by name if no summary file is present yet.\n"
        "summary_path = os.path.join(OUTPUTS_DIR, \"model_selection_summary.json\")\n"
        "if os.path.exists(summary_path):\n"
        "    with open(summary_path) as f:\n"
        "        classifier_key = json.load(f)[\"recommended_model\"]\n"
        "else:\n"
        "    classifier_key = \"efficientnetb0\"\n\n"
        "CLASSIFIER_WEIGHTS_PATH = os.path.join(MODELS_DIR, f\"{classifier_key}_best.keras\")\n"
        "classifier_model = tf.keras.models.load_model(CLASSIFIER_WEIGHTS_PATH)\n\n"
        "PREPROCESS_FNS = {\n"
        "    \"vgg16\": tf.keras.applications.vgg16.preprocess_input,\n"
        "    \"resnet50\": tf.keras.applications.resnet50.preprocess_input,\n"
        "    \"mobilenetv2\": tf.keras.applications.mobilenet_v2.preprocess_input,\n"
        "    \"efficientnetb0\": tf.keras.applications.efficientnet.preprocess_input,\n"
        "}\n"
        "classifier_preprocess = PREPROCESS_FNS[classifier_key]\n\n"
        "# The classifier's expected input size is read straight off the loaded model, so\n"
        "# this works whether it was ultimately saved at 224px or the 384px fine-tune size.\n"
        "CLASSIFIER_INPUT_SIZE = classifier_model.input_shape[1]\n\n"
        "print(f\"Loaded YOLO detector from: {YOLO_WEIGHTS_PATH}\")\n"
        "print(f\"Loaded champion classifier: {classifier_key} (input size {CLASSIFIER_INPUT_SIZE}px)\")\n"
    ),
    md(
        "### 3. Unified detect -> NMS -> filter -> crop -> classify pipeline\n\n"
        "`yolo_model.predict(..., conf=..., iou=...)` applies confidence filtering and NMS "
        "internally (Steps 2-3), so the detections returned are already deduplicated and "
        "thresholded. Everything from Step 4 onward (corner-format parsing, padded crop, "
        "resize, classifier re-verification) is implemented explicitly below."
    ),
    code(
        "CONFIDENCE_THRESHOLD = 0.5\n"
        "NMS_IOU_THRESHOLD = 0.45\n"
        "CROP_PADDING_RATIO = 0.08  # 8% padding on each side of the detected box\n\n"
        "\n"
        "def run_smartvision_pipeline(image_path, confidence=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD):\n"
        "    \"\"\"\n"
        "    Full SmartVision AI inference pipeline for a single image.\n"
        "    Returns (PIL.Image original, list of detection dicts).\n"
        "    \"\"\"\n"
        "    image = Image.open(image_path).convert(\"RGB\")\n"
        "    img_w, img_h = image.size\n\n"
        "    # --- Steps 2 & 3: YOLOv8 detection with built-in confidence filter + NMS ---\n"
        "    results = yolo_model.predict(\n"
        "        source=np.array(image), conf=confidence, iou=iou, verbose=False,\n"
        "    )\n"
        "    result = results[0]\n\n"
        "    detections = []\n"
        "    for box in result.boxes:\n"
        "        # --- Step 4: parse coordinates as [x_min, y_min, x_max, y_max] ---\n"
        "        # Ultralytics' box.xyxy is already in this corner format natively - it\n"
        "        # matches the same [x_min, y_min, x_max, y_max] convention confirmed for\n"
        "        # detection-datasets/coco's raw annotations in the project report.\n"
        "        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()\n"
        "        yolo_cls_id = int(box.cls[0])\n"
        "        yolo_confidence = float(box.conf[0])\n"
        "        yolo_label = result.names.get(yolo_cls_id, str(yolo_cls_id))\n\n"
        "        # --- Step 5a: defensive padding around the box before cropping ---\n"
        "        box_w = x_max - x_min\n"
        "        box_h = y_max - y_min\n"
        "        pad_x = box_w * CROP_PADDING_RATIO\n"
        "        pad_y = box_h * CROP_PADDING_RATIO\n"
        "        crop_x1 = max(0, x_min - pad_x)\n"
        "        crop_y1 = max(0, y_min - pad_y)\n"
        "        crop_x2 = min(img_w, x_max + pad_x)\n"
        "        crop_y2 = min(img_h, y_max + pad_y)\n\n"
        "        # --- Step 5b: crop, resize to classifier input size, scale pixels ---\n"
        "        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))\n"
        "        crop_resized = crop.resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))\n"
        "        crop_array = np.expand_dims(np.array(crop_resized).astype(\"float32\"), axis=0)\n"
        "        crop_array = classifier_preprocess(crop_array)  # model-specific pixel scaling\n\n"
        "        # --- Step 6: re-verify with the classifier ---\n"
        "        class_probs = classifier_model.predict(crop_array, verbose=0)[0]\n"
        "        top_idx = int(np.argmax(class_probs))\n"
        "        classifier_label = SELECTED_CLASSES[top_idx]\n"
        "        classifier_confidence = float(class_probs[top_idx])\n\n"
        "        detections.append({\n"
        "            \"box\": [x_min, y_min, x_max, y_max],\n"
        "            \"yolo_label\": yolo_label,\n"
        "            \"yolo_confidence\": yolo_confidence,\n"
        "            \"classifier_label\": classifier_label,\n"
        "            \"classifier_confidence\": classifier_confidence,\n"
        "        })\n\n"
        "    return image, detections\n"
    ),
    md("### 4. Draw annotated boxes with both YOLO and classifier labels"),
    code(
        "def draw_detections(image, detections):\n"
        "    annotated = image.copy()\n"
        "    draw = ImageDraw.Draw(annotated)\n"
        "    try:\n"
        "        font = ImageFont.truetype(\"DejaVuSans-Bold.ttf\", 14)\n"
        "    except Exception:\n"
        "        font = ImageFont.load_default()\n\n"
        "    for det in detections:\n"
        "        x1, y1, x2, y2 = det[\"box\"]\n"
        "        label = f\"{det['yolo_label']} -> {det['classifier_label']}\"\n"
        "        draw.rectangle([x1, y1, x2, y2], outline=(29, 158, 117), width=3)\n"
        "        draw.rectangle([x1, max(0, y1 - 18), x1 + len(label) * 7, y1], fill=(29, 158, 117))\n"
        "        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=(255, 255, 255), font=font)\n"
        "    return annotated\n"
    ),
    md(
        "### 5. Run it on a sample test image\n\n"
        "**Step 7:** prints the final combined output for every detection: "
        "`Detected [YOLO Label] with X% confidence -> Re-verified by {classifier} as "
        "[Classifier Label] with Y% confidence`."
    ),
    code(
        "import glob\n"
        "import matplotlib.pyplot as plt\n\n"
        "sample_images = glob.glob(os.path.join(DETECTION_IMAGES_DIR, \"test\", \"*.jpg\"))\n"
        "assert sample_images, \"Run 10_YOLO_Dataset_Preparation.ipynb first.\"\n\n"
        "image, detections = run_smartvision_pipeline(sample_images[0])\n"
        "annotated = draw_detections(image, detections)\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "plt.imshow(annotated)\n"
        "plt.axis(\"off\")\n"
        "plt.title(f\"{len(detections)} object(s) detected\")\n"
        "plt.show()\n\n"
        "# --- Step 7: final combined print output ---\n"
        "classifier_display_name = classifier_key.replace(\"efficientnetb0\", \"EfficientNetB0\") \\\n"
        "                                          .replace(\"resnet50\", \"ResNet50\") \\\n"
        "                                          .replace(\"vgg16\", \"VGG16\") \\\n"
        "                                          .replace(\"mobilenetv2\", \"MobileNetV2\")\n\n"
        "if not detections:\n"
        "    print(\"No objects detected above the confidence threshold.\")\n"
        "for det in detections:\n"
        "    print(\n"
        "        f\"Detected {det['yolo_label']} with {det['yolo_confidence']*100:.1f}% confidence \"\n"
        "        f\"-> Re-verified by {classifier_display_name} as {det['classifier_label']} \"\n"
        "        f\"with {det['classifier_confidence']*100:.1f}% confidence\"\n"
        "    )\n"
    ),
    md(
        "### Next step: the Streamlit app\n\n"
        "This same detect -> NMS -> filter -> pad -> crop -> classify logic is what powers "
        "`app/inference_pipeline.py` in the Streamlit application. After downloading "
        "`SmartVisionAI/models/` and `SmartVisionAI/outputs/` from Google Drive into the "
        "local project's `models/` and `outputs/` folders, run:\n\n"
        "```bash\n"
        "streamlit run app/Home.py\n"
        "```\n"
        "See the project README for full deployment instructions to Hugging Face Spaces."
    ),
]
save("12_Inference_Pipeline_Demo.ipynb", nb12_cells)

print("\nAll notebooks generated successfully.")
