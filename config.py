import os
from dotenv import load_dotenv

load_dotenv()

# File Paths
DB_PATH = "db"

# Model Configs
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"
VECTOR_DIMENSION = 512
DETECTOR_BACKEND_REALTIME = "mediapipe" # Faster
DETECTOR_BACKEND_ENROLLMENT = "retinaface" # More accurate

# Thresholds
# Note: ArcFace cosine threshold is usually around 0.68, 
# but your code used L2 distance logic (250). Adjust as needed.
FACE_MATCH_THRESHOLD = 0.40

# Credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")