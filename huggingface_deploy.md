# Deploy SmartVision AI to Hugging Face Spaces

## 1. What this deployment contains

- Streamlit app entrypoint: `app/app.py`
- Model files used by the app:
  - `yolov8n-cls.pt`
  - `runs/detect/smartvision_yolo_fast/weights/last.pt`
- Python dependencies from `requirements.txt`
- Exposes Streamlit at port `8501`

## 2. Add the Dockerfile

The repository now includes a `Dockerfile` configured for Hugging Face Spaces.

## 3. Hugging Face repo setup

1. Create a new Hugging Face Space:
   - Go to https://huggingface.co/spaces
   - Click **Create new Space**
   - Choose a name and select **Docker** as the hardware option

2. Clone the new Space repo locally:

```bash
git clone https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>
cd <SPACE_NAME>
```

3. Copy your project files into the `Space` repo.
   - Ensure the following files are present:
     - `Dockerfile`
     - `requirements.txt`
     - `app/app.py`
     - `app/pages/*.py`
     - `yolov8n-cls.pt`
     - `runs/detect/smartvision_yolo_fast/weights/last.pt`

4. Add and commit all files:

```bash
git add .
git commit -m "Deploy SmartVision AI to Hugging Face Spaces"
git push
```

## 4. Large file handling

Your YOLO model weight `runs/detect/smartvision_yolo_fast/weights/last.pt` is ~22 MB and `yolov8n-cls.pt` is ~5.6 MB.

If pushing directly fails due to size, use Git LFS:

```bash
git lfs install

git lfs track "*.pt"
git add .gitattributes
git add *.pt
git commit -m "Track model weights with Git LFS"
git push
```

> Note: Hugging Face Spaces supports Git LFS and large weights, but public Spaces normally allow a limited repo size. Use LFS for any files above standard Git limits.

## 5. Recommended `README.md` update

Add a short note to your repo README:

```md
## Running locally

python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

## 6. Troubleshooting

- If the Space fails to build, check the build logs in your Hugging Face Space.
- Confirm `Dockerfile` is present at the repository root.
- Verify `app/app.py` runs locally first:

```bash
python -m streamlit run app/app.py
```

- If model files are not found in the Space, ensure they are copied into the repo and tracked by Git/Git LFS.
