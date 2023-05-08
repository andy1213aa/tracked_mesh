import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceMesh():
    
    def __init__(self):
        
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5)
    def detect(self, image):
        
        kpt = np.zeros((478, 2))
        
        # Convert the BGR image to RGB before processing.
        results = self.face_mesh.process(image)

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return False, kpt
        
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            
            for idx, data_point in enumerate(face_landmarks.landmark):
                kpt[idx, 0] = data_point.x * annotated_image.shape[1]
                kpt[idx, 1] = data_point.y * annotated_image.shape[0]
        
        return True, kpt

