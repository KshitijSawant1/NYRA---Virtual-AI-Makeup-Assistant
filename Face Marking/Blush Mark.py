import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Left & Right Cheek Blush Areas
LEFT_CHEEK_POLYGONS = [
    [50, 101, 205], [205, 50, 187], [187, 205, 206], [206, 187, 216], [216, 206, 207]
]

RIGHT_CHEEK_POLYGONS = [
    [280, 425, 426], [426, 280, 366], [366, 426, 432], [432, 366, 436], [436, 432, 423]
]

# Blush Color (Soft Pink) & Transparency
BLUSH_COLOR = (147, 58, 103)  # Rosy Pink in BGR
ALPHA = 0.5  # Transparency

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
            landmark_coords = {idx: (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                               for idx, lm in enumerate(face_landmarks.landmark)}

            # Create a mask for blush effect
            mask = np.zeros_like(frame, dtype=np.uint8)

            # Draw **both cheek blush areas**
            for poly in LEFT_CHEEK_POLYGONS + RIGHT_CHEEK_POLYGONS:
                pts = np.array([landmark_coords[i] for i in poly], np.int32)
                cv2.fillPoly(mask, [pts], color=BLUSH_COLOR)  # Fill blush area

            # **Smooth the blush effect**
            mask = cv2.GaussianBlur(mask, (35, 35), 15)  # Soft airbrushed effect
            frame = cv2.addWeighted(frame, 1.0, mask, ALPHA, 0)  # Adjust transparency

    # Display output
    cv2.imshow("Blush Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
