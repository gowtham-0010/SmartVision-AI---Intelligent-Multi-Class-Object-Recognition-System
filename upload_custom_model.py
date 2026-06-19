from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
ROOT_DIR = Path(__file__).resolve().parents[0]
WEIGHTS_FOLDER = ROOT_DIR / "runs" / "detect" / "smartvision_yolo_fast" / "weights"
REPO_ID = "Gowtham-0010/SmartVision-AI-System"
REPO_TYPE = "space"

if not WEIGHTS_FOLDER.exists() or not WEIGHTS_FOLDER.is_dir():
    raise FileNotFoundError(f"Weights folder not found: {WEIGHTS_FOLDER}")

files = [p for p in WEIGHTS_FOLDER.rglob("*") if p.is_file()]
if not files:
    raise FileNotFoundError(f"No files found in weights folder: {WEIGHTS_FOLDER}")

print(f"Uploading {len(files)} file(s) from {WEIGHTS_FOLDER}")
for path in files:
    rel = path.relative_to(WEIGHTS_FOLDER)
    print(f" - {rel} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

api.upload_folder(
    repo_id=REPO_ID,
    folder_path=WEIGHTS_FOLDER,
    path_in_repo="runs/detect/smartvision_yolo_fast/weights",
    repo_type=REPO_TYPE,
    commit_message="Upload custom trained detection model folder",
)
print('✓ Custom model folder uploaded successfully')
