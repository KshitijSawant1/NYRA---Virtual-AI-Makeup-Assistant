import mediapipe as mp
import cv2

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize Video Capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Store clicked points
selected_points = []

def click_event(event, x, y, flags, param):
    """
    Mouse click event to select landmark points.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float("inf")
        selected_idx = None

        # Find the closest landmark
        for idx, (lx, ly) in enumerate(landmark_coords):
            dist = (x - lx) ** 2 + (y - ly) ** 2
            if dist < min_dist:
                min_dist = dist
                selected_idx = idx

        if selected_idx is not None:
            selected_points.append((landmark_coords[selected_idx], selected_idx))
            print(f"Selected Landmark: {selected_idx}")  # Print selected landmark number

# Create a window and set mouse callback
cv2.namedWindow("Face Landmark Selector")
cv2.setMouseCallback("Face Landmark Selector", click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    landmark_coords = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_coords.append((x, y))
                
                # Draw normal landmarks
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Green dots

    # Highlight selected points
    for (sx, sy), sidx in selected_points:
        cv2.circle(frame, (sx, sy), 5, (0, 0, 255), -1)  # Red for selected

    # Display output
    cv2.imshow("Face Landmark Selector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
