import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os

# Set up the capture directory
capture_directory = "Captured_Images"
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

# Initialize Streamlit app
st.set_page_config(page_title="Iris Color Changer", layout="wide")

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

# Iris landmark indices (Mediapipe uses these indices for eyes)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Sidebar controls
st.sidebar.title("🎨 Iris Color Changer")
selected_color = st.sidebar.color_picker("Pick Iris Color", "#00FF00")
blur_intensity = st.sidebar.slider("Blur Intensity", 1, 25, 5, step=2)

# Webcam controls
if "webcam_enabled" not in st.session_state:
    st.session_state.webcam_enabled = False

if st.sidebar.button(
    "🎥 Start Webcam" if not st.session_state.webcam_enabled else "🛑 Stop Webcam",
    key="webcam_toggle"
):
    st.session_state.webcam_enabled = not st.session_state.webcam_enabled

# Convert hex color to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)

# Function to apply iris color change
def apply_iris_color(frame, iris_points, iris_color, blur_value):
    mask = np.zeros_like(frame)
    iris_center = np.mean(iris_points, axis=0).astype(int)
    iris_radius = int(np.linalg.norm(iris_points[0] - iris_points[2]) // 2)

    # Draw a filled circle for the iris color
    cv2.circle(mask, tuple(iris_center), iris_radius, iris_color, -1)

    # Apply Gaussian blur to smooth out the edges
    mask = cv2.GaussianBlur(mask, (blur_value, blur_value), 0)

    # Blend the mask with the frame
    blended_frame = cv2.addWeighted(frame, 1, mask, 0.4, 0)
    return blended_frame

# Streamlit frame display
FRAME_WINDOW = st.image([])

if st.session_state.webcam_enabled:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access the webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                iris_bgr = hex_to_bgr(selected_color)

                # Get iris points for both eyes
                left_iris_points = np.array(
                    [
                        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                        for i in LEFT_IRIS
                    ]
                )
                right_iris_points = np.array(
                    [
                        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                        for i in RIGHT_IRIS
                    ]
                )

                # Apply iris color changes
                frame = apply_iris_color(frame, left_iris_points, iris_bgr, blur_intensity)
                frame = apply_iris_color(frame, right_iris_points, iris_bgr, blur_intensity)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
else:
    st.info("👆 Turn on the webcam to start changing iris colors.")
