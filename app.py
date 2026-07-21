import os
import subprocess

if __name__ == "__main__":
    # Launch Streamlit on port 7860
    subprocess.run([
        "streamlit", "run", "app/Home.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ])
