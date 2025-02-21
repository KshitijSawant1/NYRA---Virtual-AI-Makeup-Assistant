import mediapipe as mp
import cv2

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define more precise eye landmarks
LEFT_EYE = [263, 362, 373, 374, 380, 381, 382, 386, 387, 388, 390, 398, 466, 362]  # Left eye including outer corner
RIGHT_EYE = [33, 133, 160, 161, 159, 158, 157, 154, 153, 145, 144, 163, 246, 33]  # Right eye including outer corner

# Inner and outer eye corners for higher precision
LEFT_EYE_CORNERS = [263, 362]  # Left eye outer & inner corner
RIGHT_EYE_CORNERS = [33, 133]  # Right eye outer & inner corner

# Initialize Video Capture (0 for webcam)
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
            # Draw landmarks for the left eye (Blue)
            for idx in LEFT_EYE:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue color for left eye

            # Draw landmarks for the right eye (Red)
            for idx in RIGHT_EYE:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red color for right eye

            # Mark inner and outer corners for better accuracy
            for idx in LEFT_EYE_CORNERS:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)  # Yellow for left eye corners

            for idx in RIGHT_EYE_CORNERS:
                lm = face_landmarks.landmark[idx]
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)  # Yellow for right eye corners

    # Display output
    cv2.imshow("Improved Eye Landmark Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
