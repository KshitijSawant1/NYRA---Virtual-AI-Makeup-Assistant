import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Nose Contour Polygons (Selected Landmark Indices)
NOSE_CONTOUR_POLYGONS = [
    [417, 168, 8], [417, 351, 168], [168, 193, 8], [193, 168, 122], [168, 6, 122],
    [351, 6, 168], [351, 197, 6], [197, 122, 6], [197, 196, 122], [197, 195, 196],
    [196, 195, 3], [3, 195, 5], [197, 351, 419], [399, 248, 419], [195, 197, 419],
    [248, 5, 195], [248, 197, 419], [419, 248, 195], [456, 281, 248], [281, 5, 248],
    [281, 275, 5], [4, 5, 275], [245, 193, 122], [188, 245, 122], [196, 188, 122],
    [174, 188, 196], [188, 128, 245], [128, 188, 114], [188, 174, 114], [217, 114, 174],
    [47, 114, 217], [217, 126, 47], [217, 198, 126], [236, 198, 217], [126, 198, 209],
    [126, 209, 142], [209, 131, 49], [209, 49, 142], [220, 115, 131], [115, 131, 220],
    [237, 220, 45], [4, 1, 45], [1, 44, 4], [45, 5, 4], [4, 51, 5], [4, 45, 51],
    [134, 51, 45], [129, 64, 203], [198, 134, 131], [236, 51, 134], [3, 195, 51],
    [51, 195, 5], [236, 3, 51], [51, 134, 236], [236, 174, 3], [3, 174, 196],
    [217, 174, 236], [198, 236, 134], [198, 131, 209], [131, 134, 220], [220, 134, 45]
]

# Define a **Bright Pink** contouring shade (BGR Format)
BRIGHT_PINK = (180, 50, 255)  # Hot Pink in BGR
ALPHA = 0.6  # Transparency for blending

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

            # Create a mask for the nose contour
            mask = np.zeros_like(frame, dtype=np.uint8)

            # Draw **both sides** of the nose contour
            for poly in NOSE_CONTOUR_POLYGONS:
                pts = np.array([landmark_coords[i] for i in poly], np.int32)
                cv2.fillPoly(mask, [pts], color=BRIGHT_PINK)  # Fill contour with bright pink

            # **Smooth the contouring effect**
            mask = cv2.GaussianBlur(mask, (35, 35), 15)  # Increased blur for a natural effect
            frame = cv2.addWeighted(frame, 1.0, mask, ALPHA, 0)  # Adjust transparency

    # Display output
    cv2.imshow("Balanced Bright Pink Nose Contour", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
