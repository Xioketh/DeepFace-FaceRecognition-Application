import streamlit as st
import cv2
import numpy as np
import faiss
import pickle
import os
from deepface import DeepFace
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from twilio.rest import Client
from dotenv import load_dotenv

# --- CONFIGURATION ---
DB_PATH = "db"
MODEL_NAME = "ArcFace"  # Switching to ArcFace for better accuracy (Vector Size: 512)
DISTANCE_METRIC = "cosine"

load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# FAISS requires knowing the vector dimension beforehand
# ArcFace = 512, VGG-Face = 2622, FaceNet = 128
VECTOR_DIMENSION = 512 

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def get_ice_servers():
    """
    Fetches the TURN server credentials from Twilio.
    This solves the 'Connection taking too long' error on mobile.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Could not fetch TURN servers: {e}. Falling back to STUN (might fail on mobile).")
        # Fallback to Google's STUN if Twilio fails
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# --- FAISS CORE ---
def init_faiss_index():
    """
    Loads all .pkl files and builds a FAISS index in RAM.
    Returns: (index, names_list)
    """
    # Create a Flat L2 Index (Euclidean Distance)
    # Note: FAISS uses Euclidean by default. For Cosine, we usually normalize vectors first.
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    
    names = []
    vectors = []
    
    # Load all vectors from disk
    if os.path.exists(DB_PATH):
        for file in os.listdir(DB_PATH):
            if file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                with open(os.path.join(DB_PATH, file), "rb") as f:
                    emb = pickle.load(f)
                    # FAISS expects float32 numpy arrays
                    vectors.append(np.array(emb, dtype='float32'))
                    names.append(name)
    
    if vectors:
        # Convert list of arrays to a single matrix (N, 512)
        vectors_matrix = np.array(vectors)
        index.add(vectors_matrix)
    
    return index, names

def save_face(name, image_array, embedding):
    img_path = os.path.join(DB_PATH, f"{name}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    vector_path = os.path.join(DB_PATH, f"{name}.pkl")
    with open(vector_path, "wb") as f:
        pickle.dump(embedding, f)

# --- REAL-TIME PROCESSING CLASS ---
class FaceRecognitionProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detected_name = "Scanning..."
        # Load index once at startup
        self.index, self.names = init_faiss_index()

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Optimization: Only run recognition every 30 frames (approx 1 sec)
        # Running it every frame will lag the video heavily on CPU
        if self.frame_count % 30 == 0:
            try:
                # 1. Detect & Embed
                embedding_objs = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL_NAME,
                    enforce_detection=False # Don't crash if no face
                )
                
                if embedding_objs:
                    target_emb = embedding_objs[0]["embedding"]
                    target_emb = np.array([target_emb], dtype='float32')
                    
                    # 2. FAISS Search
                    # k=1 means "Find the 1 closest match"
                    if self.index.ntotal > 0:
                        distances, indices = self.index.search(target_emb, k=1)
                        
                        best_idx = indices[0][0]
                        best_dist = distances[0][0]
                        
                        # Thresholding (Adjust based on model/metric)
                        # ArcFace L2 distance < 10 is usually a good match guess
                        if best_dist < 100: 
                             self.detected_name = f"{self.names[best_idx]} ({int(best_dist)})"
                        else:
                            self.detected_name = "Unknown"
            except Exception as e:
                pass # Ignore errors to keep video smooth

        self.frame_count += 1
        
        # Draw the result on the frame
        cv2.putText(img, self.detected_name, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# --- UI LAYOUT ---
st.title("Real-Time AI Identity System")
mode = st.sidebar.selectbox("Mode", ["Enrollment", "Real-Time Recognition"])

if mode == "Enrollment":
    st.header("Enrollment")
    name = st.text_input("Name")
    img_file = st.camera_input("Register Face")
    
    if st.button("Save") and img_file and name:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            emb = DeepFace.represent(img, model_name=MODEL_NAME,detector_backend="retinaface",  # <--- The Fix
                    enforce_detection=True)[0]["embedding"]
            save_face(name, img, emb)
            st.success(f"User {name} registered successfully!")
        except ValueError:
            st.error("Face not detected! Please ensure your face is clearly visible and well-lit.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif mode == "Real-Time Recognition":
    st.header("Live Feed")
    st.info("Processing runs every ~1 second to maintain FPS.")
    ice_servers = get_ice_servers()
    
    # The WebRTC Magic
    webrtc_streamer(
        key="realtime-face",
        video_processor_factory=FaceRecognitionProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )