import subprocess

# Satisfy Hugging Face ZeroGPU check at startup
try:
    import spaces

    @spaces.GPU(duration=5)
    def initialize_zerogpu():
        print("ZeroGPU initialized successfully for SmartVision!")
        return True

    initialize_zerogpu()
except Exception as e:
    print(f"ZeroGPU initialization skipped or failed: {e}")

if __name__ == "__main__":
    # Launch Streamlit with CORS & XSRF disabled for Hugging Face uploads
    subprocess.run([
        "streamlit", "run", "app/Home.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ])
