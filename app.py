"""
streamlit run app.py

"""
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
from common import picking_point

st.set_page_config(page_title="Tea shoot plucking points")


@st.cache_resource
def load_model(path):
    return YOLO(path)


@st.cache_resource
def settings(): 
    return {"conf": 0.4}


def annotate(model, img, conf):
    r = model(img, conf=conf, imgsz=640, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return img, 0
    for (x1, y1, x2, y2), k in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        p = picking_point(k)
        if p:
            c = (int(p[0]), int(p[1]))
            cv2.circle(img, c, 6, (0, 0, 255), -1)       
            cv2.circle(img, c, 7, (255, 255, 255), 1)   
    return img, len(r.boxes)


st.title("Tea shoot plucking points")
model = load_model(st.sidebar.text_input("Model weights", "best.pt"))
cfg = settings()
cfg["conf"] = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
mode = st.radio("Camera", ["Live video", "Single photo"], horizontal=True)

if mode == "Live video":
    def callback(frame):
        img, _ = annotate(model, frame.to_ndarray(format="bgr24"), cfg["conf"])
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="tea",
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )
    st.caption("Uses the phone's rear camera. The page must be opened over HTTPS.")
else:
    shot = st.camera_input("Take a photo of the shoots")
    if shot:
        img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img, n = annotate(model, img, cfg["conf"])
        st.image(img[:, :, ::-1], caption=f"{n} shoots found")
