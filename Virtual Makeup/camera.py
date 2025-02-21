import cv2
from utils import *

# Define makeup features and colors
face_elements = ["LIP_LOWER", "LIP_UPPER", "EYEBROW_LEFT", "EYEBROW_RIGHT"]
colors_map = {
    "LIP_UPPER": [0, 0, 255],  # Red Lips
    "LIP_LOWER": [0, 0, 255],
    "EYEBROW_LEFT": [19, 69, 139],  # Dark Brown Eyebrows
    "EYEBROW_RIGHT": [19, 69, 139]
}

# Extract landmark points and colors
face_connections = [face_points[idx] for idx in face_elements]
colors = [colors_map[idx] for idx in face_elements]

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    success, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    if success:
        mask = np.zeros_like(frame)
        face_landmarks = read_landmarks(frame)

        if not face_landmarks:
            print("⚠ No face detected, skipping frame.")
            continue  # Skip this frame

        mask = add_mask(mask, face_landmarks, face_connections, colors)
        output = cv2.addWeighted(frame, 1.0, mask, 0.3, 0)

        cv2.imshow("Virtual Makeup Try-On", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
