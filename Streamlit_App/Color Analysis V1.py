import streamlit as st
import cohere
import cv2
import numpy as np
import mediapipe as mp

# Initialize Cohere Client
api_key = st.sidebar.text_input("Enter Cohere API Key", type="password")
if api_key:
    co = cohere.Client(api_key)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define landmarks for skin, iris, lips
SKIN_LANDMARK = 10  # Tip of the nose
LEFT_IRIS = 468
RIGHT_IRIS = 473
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Function to convert BGR to HEX
def bgr_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])

# Function to display color swatches in Streamlit
def color_swatch(hex_code):
    return f'<div style="display:inline-block;width:20px;height:20px;background-color:{hex_code};border-radius:3px;margin-right:5px;"></div>'

# Streamlit App UI
st.title("🎨 AI-Based Personal Color Analysis")
st.sidebar.title("📷 Webcam Settings")
use_webcam = st.sidebar.checkbox("Enable Webcam")

# Initialize state to avoid multiple analyses per frame
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# Add a button to trigger color analysis
generate_analysis = st.sidebar.button("Generate Color Analysis")

if use_webcam:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    skin_hex, iris_hex, lips_hex = None, None, None

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

                # Extract colors without displaying points
                skin_point = face_landmarks.landmark[SKIN_LANDMARK]
                skin_x, skin_y = int(skin_point.x * w), int(skin_point.y * h)
                skin_color = frame[skin_y, skin_x]

                iris_point = face_landmarks.landmark[LEFT_IRIS]
                iris_x, iris_y = int(iris_point.x * w), int(iris_point.y * h)
                iris_color = frame[iris_y, iris_x]

                lip_points = np.array(
                    [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in LIPS], np.int32
                )
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [lip_points], 255)
                mean_lip_color = cv2.mean(frame, mask=mask)[:3]

                # Convert colors to HEX
                skin_hex = bgr_to_hex(skin_color)
                iris_hex = bgr_to_hex(iris_color)
                lips_hex = bgr_to_hex((int(mean_lip_color[0]), int(mean_lip_color[1]), int(mean_lip_color[2])))

        # Show the webcam feed without landmark points
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Trigger color analysis on button click
        if generate_analysis and api_key and not st.session_state.analysis_done:
            if skin_hex and iris_hex and lips_hex:
                prompt = (
                    f"Provide an in-depth fashion, makeup, and color palette analysis for someone with "
                    f"skin tone {skin_hex}, iris color {iris_hex}, and lip color {lips_hex}. Suggest suitable "
                    f"clothing colors, makeup shades, and hair colors based on seasonal color theory."
                    f"Also keep everyting concise and provide result in points"
                )
                response = co.generate(
                    model='command-xlarge-nightly',
                    prompt=prompt,
                    max_tokens=400  # Increased token count for full analysis
                )
                analysis = response.generations[0].text
                st.session_state.analysis_done = True  # Prevent repeated analysis per frame

                # Display the AI response with color swatches
                st.subheader("🌈 AI Color Analysis Result")
                st.markdown(f"**Skin Color:** {color_swatch(skin_hex)} {skin_hex}", unsafe_allow_html=True)
                st.markdown(f"**Iris Color:** {color_swatch(iris_hex)} {iris_hex}", unsafe_allow_html=True)
                st.markdown(f"**Lip Color:** {color_swatch(lips_hex)} {lips_hex}", unsafe_allow_html=True)
                st.markdown(analysis)

    cap.release()
    st.session_state.analysis_done = False  # Reset analysis state after release
