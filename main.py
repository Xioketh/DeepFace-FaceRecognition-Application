import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Custom Modules
from config import MODEL_NAME, DETECTOR_BACKEND_ENROLLMENT
from database import save_face
from utils import get_ice_servers
from processor import FaceRecognitionProcessor

st.title("Real-Time AI Identity System")
mode = st.sidebar.selectbox("Mode", ["Enrollment", "Real-Time Recognition"])

# --- MODE: ENROLLMENT ---
if mode == "Enrollment":
    st.header("Enrollment")
    name = st.text_input("Name")
    img_file = st.camera_input("Register Face")
    
    if st.button("Save") and img_file and name:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Using RetinaFace for enrollment (slower but more accurate)
            emb = DeepFace.represent(
                img, 
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND_ENROLLMENT, 
                enforce_detection=True
            )[0]["embedding"]

            # if face_area < 15000: # arbitrary pixel count
            #     st.warning("Move closer to the camera")
            
            save_face(name, img, emb)
            st.success(f"User {name} registered successfully!")
            
        except ValueError:
            st.error("Face not detected! Please ensure your face is clearly visible.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- MODE: RECOGNITION ---
elif mode == "Real-Time Recognition":
    st.header("Live Feed")
    st.info("Processing runs every ~30 frames to maintain performance.")
    
    ice_servers = get_ice_servers()
    
    webrtc_streamer(
        key="realtime-face",
        video_processor_factory=FaceRecognitionProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )