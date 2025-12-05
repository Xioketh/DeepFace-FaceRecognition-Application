import os
import pickle
import cv2
import numpy as np
import faiss
from config import DB_PATH, VECTOR_DIMENSION

# Ensure DB exists on import
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def init_faiss_index():
    """Loads existing embeddings and builds the FAISS index."""
    index = faiss.IndexFlatIP(VECTOR_DIMENSION)
    names = []
    vectors = []
    
    if os.path.exists(DB_PATH):
        for file in os.listdir(DB_PATH):
            if file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                vector_path = os.path.join(DB_PATH, file)
                
                try:
                    with open(vector_path, "rb") as f:
                        emb = pickle.load(f)
                        emb_np = np.array(emb, dtype='float32')
                        faiss.normalize_L2(emb_np.reshape(1, -1))
                        vectors.append(emb_np.flatten())
                        names.append(name)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    if vectors:
        vectors_matrix = np.array(vectors)
        vectors_matrix = vectors_matrix.reshape(-1, VECTOR_DIMENSION)
        index.add(vectors_matrix)
    
    return index, names

def save_face(name, image_array, embedding):
    img_path = os.path.join(DB_PATH, f"{name}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    vector_path = os.path.join(DB_PATH, f"{name}.pkl")
    with open(vector_path, "wb") as f:
        pickle.dump(embedding, f)