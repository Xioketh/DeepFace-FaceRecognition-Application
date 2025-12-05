# processor.py
import time
import cv2
import numpy as np
import faiss 
from deepface import DeepFace
from streamlit_webrtc import VideoTransformerBase
from database import init_faiss_index
from config import MODEL_NAME, DETECTOR_BACKEND_REALTIME, FACE_MATCH_THRESHOLD

class FaceRecognitionProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detected_name = "Scanning..."
        self.index, self.names = init_faiss_index()
        self.last_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        if self.frame_count % 30 == 0:
            self.last_time = current_time

            try:
                embedding_objs = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND_REALTIME,
                    enforce_detection=True
                )
                
                if embedding_objs:
                    target_emb = embedding_objs[0]["embedding"]
                    target_emb = np.array([target_emb], dtype='float32').reshape(1, -1)
                    
                    faiss.normalize_L2(target_emb)
                    
                    if self.index.ntotal > 0:
                        # Search
                        similarities, indices = self.index.search(target_emb, k=1)
                        
                        best_idx = indices[0][0]
                        best_similarity = similarities[0][0] # This is now a Score (-1 to 1)

                        print(f"Match: {self.names[best_idx]} | Score: {best_similarity:.4f}")

                        if best_similarity > FACE_MATCH_THRESHOLD: 
                             self.detected_name = f"{self.names[best_idx]} ({int(best_similarity*100)}%)"
                        else:
                            self.detected_name = "Unknown User"
            except ValueError:
                self.detected_name = "No Face Found"
            except Exception as e:
                print(f"Error in processor: {e}")

        self.frame_count += 1

        # Draw UI
        cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Red for Unknown/No Face, Green for Known
        if "Unknown" in self.detected_name or "No Face" in self.detected_name:
            color = (0, 0, 255) # Red
        else:
            color = (0, 255, 0) # Green
        
        cv2.putText(img, self.detected_name, (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return img