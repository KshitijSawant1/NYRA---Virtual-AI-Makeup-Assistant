import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

selected_polygons = []  # Store selected polygon landmark indices
selected_triangles = []  # Store the actual triangle indices
landmark_coords = {}  # Store landmark coordinates

def inside_polygon(triangle, point):
    """ Check if a point is inside a given triangle (Barycentric method). """
    A, B, C = np.array(triangle, dtype=np.float32)
    P = np.array(point, dtype=np.float32)

    # Compute Barycentric Coordinates
    v0 = B - A
    v1 = C - A
    v2 = P - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v < 1)

def select_polygon(event, x, y, flags, param):
    """ Mouse callback to select a polygon when clicking inside it. """
    global selected_polygons, selected_triangles

    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, tri in enumerate(selected_triangles):
            triangle_points = [landmark_coords[i] for i in tri]
            if inside_polygon(triangle_points, (x, y)):
                if tri not in selected_polygons:
                    selected_polygons.append(tri)
                    print(f"Selected Polygon: {tri}")  # Print selected polygon indices
                return

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Polygon Selection on Face Mesh")
cv2.setMouseCallback("Polygon Selection on Face Mesh", select_polygon)

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

            # Convert facial landmarks to a NumPy array for Delaunay Triangulation
            landmark_array = np.array(list(landmark_coords.values()), dtype=np.int32)
            if len(landmark_array) > 0:
                delaunay_tri = Delaunay(landmark_array)

                # Store Delaunay triangles (indices referring to landmarks)
                selected_triangles = delaunay_tri.simplices.tolist()

                # Draw facial mesh
                for tri in selected_triangles:
                    pts = np.array([landmark_coords[i] for i in tri], np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

                # Draw selected polygons
                for tri in selected_polygons:
                    pts = np.array([landmark_coords[i] for i in tri], np.int32)
                    cv2.fillPoly(frame, [pts], color=(255, 0, 0))  # Blue fill for selected areas

    # Display output
    cv2.imshow("Polygon Selection on Face Mesh", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        print("\nFinal Selected Polygons:")
        print(selected_polygons)  # Print all selected polygons at the end
        break
    elif key == ord('c'):  # Clear selections
        selected_polygons.clear()
        print("Cleared all selections.")

cap.release()
cv2.destroyAllWindows()
