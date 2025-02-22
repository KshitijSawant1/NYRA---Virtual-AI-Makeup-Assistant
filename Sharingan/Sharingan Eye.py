import cv2
import numpy as np
import mediapipe as mp

# Load MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load Sharingan Image
sharingan_img = cv2.imread("Sharingan/image2.png", cv2.IMREAD_UNCHANGED)

if sharingan_img is None:
    print("Error: Could not load Sharingan image. Check the file path!")
    exit()

# Iris Landmark Indices from MediaPipe
LEFT_IRIS = [469, 470, 471, 472]  # Left iris landmarks
RIGHT_IRIS = [474, 475, 476, 477] # Right iris landmarks

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract Iris Positions
            left_iris_pts = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_IRIS])
            right_iris_pts = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_IRIS])

            # Calculate Bounding Box of Left & Right Iris
            lx, ly, lw, lh = cv2.boundingRect(left_iris_pts)
            rx, ry, rw, rh = cv2.boundingRect(right_iris_pts)

            # Resize Sharingan to Match Iris Size
            sharingan_resized = cv2.resize(sharingan_img, (lw, lh))
            
            # Apply Sharingan on Left Iris
            for i in range(lh):
                for j in range(lw):
                    if sharingan_resized[i, j, 3] > 0:  # Check Alpha Channel
                        frame[ly + i, lx + j] = sharingan_resized[i, j, :3]

            # Resize Sharingan for Right Iris
            sharingan_resized = cv2.resize(sharingan_img, (rw, rh))

            # Apply Sharingan on Right Iris
            for i in range(rh):
                for j in range(rw):
                    if sharingan_resized[i, j, 3] > 0:  # Check Alpha Channel
                        frame[ry + i, rx + j] = sharingan_resized[i, j, :3]

    # Show Output
    cv2.imshow("Sharingan Eye Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
