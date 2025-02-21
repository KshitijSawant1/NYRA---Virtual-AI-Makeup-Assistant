import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define Facial Feature Landmarks
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
CHEEKS = [50, 280]  # Blush Area

# Streamlit UI
st.title("💄 Virtual Makeup App")
st.sidebar.title("🔧 Adjust Makeup Settings")

# Makeup Color Pickers
lipstick_color = st.sidebar.color_picker("💋 Lipstick Color", "#FF0000")
blush_color = st.sidebar.color_picker("🌸 Blush Color", "#FF69B4")

# Opacity Slider
opacity = st.sidebar.slider("Makeup Intensity", 0.1, 1.0, 0.6)

# Convert Hex to BGR for OpenCV
def hex_to_bgr(hex_color):
    """Converts Hex to BGR color format for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Fix incorrect ordering
    return (b, g, r)  # Convert RGB to BGR

lipstick_bgr = hex_to_bgr(lipstick_color)
blush_bgr = hex_to_bgr(blush_color)

# File Upload Feature
uploaded_file = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file))

    # Convert to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process image with Mediapipe
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        mask = np.zeros_like(image, dtype=np.uint8)

        for face_landmarks in results.multi_face_landmarks:
            # Extract Lip Points
            lip_points = np.array([[int(face_landmarks.landmark[i].x * image.shape[1]), 
                                    int(face_landmarks.landmark[i].y * image.shape[0])] for i in LIPS], np.int32)
            if len(lip_points) > 2:
                cv2.fillPoly(mask, [lip_points], lipstick_bgr)

            # Extract Blush (Cheeks)
            cheek_points = np.array([[int(face_landmarks.landmark[i].x * image.shape[1]), 
                                      int(face_landmarks.landmark[i].y * image.shape[0])] for i in CHEEKS], np.int32)
            for pt in cheek_points:
                cv2.circle(mask, tuple(pt), 40, blush_bgr, -1)

        # **Fix Negative Color Issue by Properly Merging**
        mask = cv2.GaussianBlur(mask, (15, 15), 10)

        # Proper Blending Using Alpha
        blended_image = cv2.addWeighted(image, 1 - opacity, mask, opacity, 0)

        # Convert back to RGB for Display
        blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        st.image(blended_image, caption="💄 Virtual Makeup Applied", use_column_width=True)

# Webcam Feature
use_webcam = st.sidebar.checkbox("📷 Use Webcam")
if use_webcam:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam!")
            break

        # Convert frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in results.multi_face_landmarks:
                lip_points = np.array([[int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                        int(face_landmarks.landmark[i].y * frame.shape[0])] for i in LIPS], np.int32)
                if len(lip_points) > 2:
                    cv2.fillPoly(mask, [lip_points], lipstick_bgr)

                cheek_points = np.array([[int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                          int(face_landmarks.landmark[i].y * frame.shape[0])] for i in CHEEKS], np.int32)
                for pt in cheek_points:
                    cv2.circle(mask, tuple(pt), 40, blush_bgr, -1)

            # **Fix Negative Colors by Correctly Merging Mask**
            mask = cv2.GaussianBlur(mask, (15, 15), 10)
            frame = cv2.addWeighted(frame, 1 - opacity, mask, opacity, 0)

        # Show live video in Streamlit
        FRAME_WINDOW.image(frame, channels="RGB")

    cap.release()
