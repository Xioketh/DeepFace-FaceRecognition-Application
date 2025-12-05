import streamlit as st
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
import numpy as np
import faiss
import cv2

def get_ice_servers():
    """Fetches TURN servers from Twilio for WebRTC traversal."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Could not fetch TURN servers: {e}. Falling back to STUN.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    


def calculate_centroid(embeddings_list):
    """
    Combines multiple face vectors into one 'Master Vector'.
    """
    # 1. Stack them into a matrix
    mat = np.array(embeddings_list, dtype='float32')
    
    # 2. Normalize each vector individually (Length = 1)
    faiss.normalize_L2(mat)
    
    # 3. Calculate the Mean Vector (Average)
    centroid = np.mean(mat, axis=0).reshape(1, -1)
    
    # 4. Normalize the Centroid (Crucial step for ArcFace!)
    faiss.normalize_L2(centroid)
    
    return centroid.flatten() # Return as 1D array



def get_image_quality(img):
    """
    Returns a quality score (0-100) based on sharpness and brightness.
    """
    # 1. Blur Detection (Laplacian Variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Thresholds: < 100 is usually blurry for face rec
    # We map 0-500 to 0-100 range roughly
    sharpness = min(100, blur_score / 5)

    # 2. Brightness Check
    avg_brightness = np.mean(gray)
    if avg_brightness < 40 or avg_brightness > 220:
        return 0 # Too dark or too washed out
        
    return sharpness