import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
import pickle
from scipy.spatial.distance import cosine

# --- CONFIGURATION ---
DB_PATH = "db"
# VGG-Face is the default model. 
# For VGG-Face + Cosine Distance, a good threshold is usually 0.40
# MODEL_NAME = "VGG-Face"
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40

# Ensure DB folder exists
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# --- HELPER FUNCTIONS ---

def save_face(name, image_array, embedding):
    """
    Saves the raw image and the embedding vector.
    """
    # 1. Save Image
    img_path = os.path.join(DB_PATH, f"{name}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    # 2. Save Embedding (Vector) to a pickle file (acting as your JSON/DB)
    # We use pickle here because it handles numpy arrays better than standard JSON
    vector_path = os.path.join(DB_PATH, f"{name}.pkl")
    with open(vector_path, "wb") as f:
        pickle.dump(embedding, f)

def load_db():
    """
    Loads all saved vectors from the db folder.
    Returns a dictionary: { "User1": [vector], "User2": [vector] }
    """
    database = {}
    for file in os.listdir(DB_PATH):
        if file.endswith(".pkl"):
            name = file.replace(".pkl", "")
            with open(os.path.join(DB_PATH, file), "rb") as f:
                database[name] = pickle.load(f)
    return database

def get_embedding(image_array):
    """
    The 'Magic Step': Detects face -> Aligns -> Vectorizes.
    Returns the embedding list.
    """
    try:
        # DeepFace.represent performs detection, alignment, and vectorization
        # enforce_detection=True throws an error if no face is found (Flow A requirement)
        embedding_objs = DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        # DeepFace returns a list (in case of multiple faces), we take the first one
        return embedding_objs[0]["embedding"]
    except ValueError:
        return None

# --- STREAMLIT UI ---

st.title("User Identity System")
st.write(f"Using Model: **{MODEL_NAME}** | Threshold: **{THRESHOLD}**")

# Sidebar to switch modes
mode = st.sidebar.selectbox("Select Mode", ["Enrollment", "Recognition"])

# ==========================================
# FLOW A: ENROLLMENT
# ==========================================
if mode == "Enrollment":
    st.header("Enrollment Phase")
    
    # User Input
    user_name = st.text_input("Enter User Name", placeholder="e.g., User1")
    
    # Capture
    img_file = st.camera_input("Take a picture to register")
    
    if img_file is not None and user_name:
        if st.button("Register User"):
            # Convert Streamlit buffer to OpenCV format
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with st.spinner("Processing face..."):
                # Vectorization (The Magic Step)
                embedding = get_embedding(img)
                
                if embedding is None:
                    st.error("No face detected! Please look clearly at the camera.")
                else:
                    # Storage
                    save_face(user_name, img, embedding)
                    st.success(f"User '{user_name}' registered successfully!")
                    st.json(embedding[:5]) # Show first 5 numbers of vector for demo

# ==========================================
# FLOW B: RECOGNITION
# ==========================================
elif mode == "Recognition":
    st.header("Recognition Phase")
    
    # Capture
    img_file = st.camera_input("Take a picture to login")
    
    if img_file is not None:
        # Convert Streamlit buffer to OpenCV format
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with st.spinner("Identifying..."):
            # Vectorization
            target_embedding = get_embedding(img)
            
            if target_embedding is None:
                st.error("No face detected.")
            else:
                # Load Database
                db = load_db()
                best_match_name = "Unknown"
                min_distance = float("inf")
                
                # Search (Distance Calculation)
                for name, saved_embedding in db.items():
                    # Calculate Cosine Distance
                    dist = cosine(target_embedding, saved_embedding)
                    
                    # Logic to find the closest match
                    if dist < min_distance:
                        min_distance = dist
                        best_match_name = name

                # Decision (Thresholding)
                if min_distance <= THRESHOLD:
                    st.success(f"Match Found! Welcome, **{best_match_name}**.")
                    st.info(f"Distance Score: {min_distance:.4f} (Lower is better)")
                else:
                    st.warning("Access Denied: Unknown User.")
                    st.info(f"Closest match was {best_match_name} with score {min_distance:.4f} (Too High)")