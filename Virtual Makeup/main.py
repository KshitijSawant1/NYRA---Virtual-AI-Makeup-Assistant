import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define Facial Landmarks
LIP_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
LIP_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]
LEFT_EYE = [263, 362, 373, 374, 380, 381, 382, 386, 387, 388, 390, 398, 466, 362]
RIGHT_EYE = [33, 133, 160, 161, 159, 158, 157, 154, 153, 145, 144, 163, 246, 33]

# Define Colors (BGR)
LIP_COLOR = (0, 0, 255)  # Red
LEFT_EYESHADOW_COLOR = (255, 20, 147)  # Pink
RIGHT_EYESHADOW_COLOR = (147, 112, 219)  # Purple
EYE_LANDMARK_COLOR = (0, 255, 255)  # Yellow for landmarks
ALPHA = 0.6  # Transparency level

def read_landmarks(frame):
    """Extracts facial landmarks from a frame using Mediapipe."""
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    return {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(len(landmarks))}

def apply_makeup(frame, points, color, alpha=ALPHA):
    """Applies a transparent makeup mask over the given points."""
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], color)
    return cv2.addWeighted(frame, 1, mask, alpha, 0)

# Initialize Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = read_landmarks(frame)

    if landmarks:
        # Apply Lipstick
        lip_upper = [landmarks[i] for i in LIP_UPPER]
        lip_lower = [landmarks[i] for i in LIP_LOWER]
        frame = apply_makeup(frame, lip_upper, LIP_COLOR)
        frame = apply_makeup(frame, lip_lower, LIP_COLOR)

        # Apply Eyeshadow
        left_eyeshadow = [landmarks[i] for i in LEFT_EYESHADOW]
        right_eyeshadow = [landmarks[i] for i in RIGHT_EYESHADOW]
        frame = apply_makeup(frame, left_eyeshadow, LEFT_EYESHADOW_COLOR)
        frame = apply_makeup(frame, right_eyeshadow, RIGHT_EYESHADOW_COLOR)

        # Draw Eye Landmarks
        for i in LEFT_EYE + RIGHT_EYE:
            x, y = landmarks[i]
            cv2.circle(frame, (x, y), 2, EYE_LANDMARK_COLOR, -1)

    # Display Output
    cv2.imshow("Virtual Makeup Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
