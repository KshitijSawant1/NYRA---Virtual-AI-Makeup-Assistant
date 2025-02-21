import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def read_landmarks(frame):
    """Extract facial landmarks from the given frame."""
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    return {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(len(landmarks))}

def apply_makeup(frame, points, color, alpha=0.6):
    """Apply a transparent colored mask on the given facial region."""
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], color)
    return cv2.addWeighted(frame, 1, mask, alpha, 0)
