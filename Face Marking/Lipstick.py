import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define Lip Landmarks (Upper & Lower)
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]

# Lipstick Color & Transparency
LIPSTICK_COLOR = (0, 0, 255)  # Red in BGR
ALPHA = 0.6  # Transparency level

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
            upper_lip_points = []
            lower_lip_points = []

            # Get Upper Lip Points
            for idx in UPPER_LIP:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                upper_lip_points.append((x, y))

            # Get Lower Lip Points
            for idx in LOWER_LIP:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                lower_lip_points.append((x, y))

            if len(upper_lip_points) > 2 and len(lower_lip_points) > 2:
                # Create mask for lips
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(upper_lip_points, np.int32)], LIPSTICK_COLOR)
                cv2.fillPoly(mask, [np.array(lower_lip_points, np.int32)], LIPSTICK_COLOR)

                # Apply transparency blending
                frame = cv2.addWeighted(frame, 1, mask, ALPHA, 0)

    # Display output
    cv2.imshow("Virtual Lipstick", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
