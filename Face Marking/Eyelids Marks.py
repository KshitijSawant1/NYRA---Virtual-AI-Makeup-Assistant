import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Left and Right Eyeshadow Regions (Fully Enclosed Polygon)
LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]

# Eyeshadow Colors
LEFT_EYESHADOW_COLOR = (255, 20, 147)  # Pink in BGR
RIGHT_EYESHADOW_COLOR = (147, 112, 219)  # Purple in BGR

# Transparency Level
ALPHA = 0.6

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eyeshadow_points = []
            right_eyeshadow_points = []

            # Get Left Eyeshadow Points
            for idx in LEFT_EYESHADOW:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                left_eyeshadow_points.append((x, y))

            # Get Right Eyeshadow Points
            for idx in RIGHT_EYESHADOW:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                right_eyeshadow_points.append((x, y))

            # Apply Left Eyeshadow
            if len(left_eyeshadow_points) > 2:
                left_mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(left_mask, [np.array(left_eyeshadow_points, np.int32)], LEFT_EYESHADOW_COLOR)
                frame = cv2.addWeighted(frame, 1, left_mask, ALPHA, 0)

            # Apply Right Eyeshadow
            if len(right_eyeshadow_points) > 2:
                right_mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(right_mask, [np.array(right_eyeshadow_points, np.int32)], RIGHT_EYESHADOW_COLOR)
                frame = cv2.addWeighted(frame, 1, right_mask, ALPHA, 0)

    # Display output
    cv2.imshow("Dual Eyeshadow Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
