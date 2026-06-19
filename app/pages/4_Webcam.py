import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("🎥 Real-Time Webcam Detection")

current_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
MODEL_PATH = repo_root / "runs" / "detect" / "smartvision_yolo_fast" / "weights" / "last.pt"

with st.expander("🛠️ System Diagnostics", expanded=False):
    st.text(f"Current script directory: {current_dir}")
    st.text(f"Detected Repository Root: {repo_root}")
    st.text(f"Looking for weights at absolute path: {MODEL_PATH}")

    if MODEL_PATH.exists():
        st.success(f"📁 SUCCESS: Found weights file! Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        st.error("❌ ERROR: The weights file does not exist at this path inside the running container!")
        all_pt_files = list(repo_root.glob("**/*.pt"))
        if all_pt_files:
            st.write("Found these alternative .pt files in your repo:")
            for file in all_pt_files:
                st.code(str(file.relative_to(repo_root)))
        else:
            st.error("No .pt files found anywhere in the entire repository directory!")


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return YOLO(str(MODEL_PATH)), "🎯 Successfully initialized custom model weights!"
        except Exception as e:
            return None, f"💥 Failed to parse weights file: {e}"
    else:
        return None, "⚠️ Running with empty initialization because file path was not found."


model, status_msg = load_model()
if model:
    st.info(status_msg)
    with st.expander("📋 Classes this model can detect"):
        st.json(model.names)
else:
    st.error(status_msg)
    st.stop()


st.write("### 📡 Live Detection")
conf_threshold = st.slider("Model Confidence Threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.05)


@st.cache_resource
def get_rtc_configuration():
    """Use Twilio TURN credentials if configured as HF Space secrets, otherwise fall back to public STUN."""
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if account_sid and auth_token:
        try:
            from twilio.rest import Client
            token = Client(account_sid, auth_token).tokens.create()
            return {"iceServers": token.ice_servers}
        except Exception as e:
            st.warning(f"Twilio TURN setup failed, falling back to STUN only: {e}")
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # YOLO expects BGR (same as cv2.imread) -- no conversion needed

    results = model(img, conf=conf_threshold, verbose=False)
    annotated = results[0].plot()  # comes back in the same BGR order, ready to send straight back

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


webrtc_streamer(
    key="realtime-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=get_rtc_configuration(),
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption(
    "Click START above to begin. If the feed stays on 'Connecting...' for more than "
    "10-15 seconds, your network needs a TURN server -- see the notes below the code."
)