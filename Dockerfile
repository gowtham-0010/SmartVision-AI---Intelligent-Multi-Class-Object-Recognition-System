FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install system dependencies required for OpenCV and WebRTC binaries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and run pip install explicitly inside the image layer
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the remaining workspace files into the container
COPY . ./

# Hugging Face Spaces strictly requires port 7860 to route traffic to the container
EXPOSE 7860

# Run streamlit with XSRF protection disabled to permanently fix the 403 error on file uploads
CMD ["sh", "-c", "streamlit run app/app.py --server.port 7860 --server.address 0.0.0.0 --server.enableXsrfProtection false"]