import streamlit as st
import cohere
import cv2
import numpy as np
import mediapipe as mp

# Initialize Cohere Client
api_key = st.sidebar.text_input("apikey", type="password")
if api_key:
    co = cohere.Client(api_key)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define landmarks for skin, iris, lips, and eyebrows
SKIN_LANDMARK = 10  # Tip of the nose
LEFT_IRIS = 468
RIGHT_IRIS = 473
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Function to convert BGR to HEX
def bgr_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])

# Streamlit App UI
st.title("🎨 Color Analysis Using AI")
st.sidebar.title("📷 Webcam Settings")
use_webcam = st.sidebar.checkbox("Enable Webcam")

if use_webcam:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

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

                # Extract color points
                skin_point = face_landmarks.landmark[SKIN_LANDMARK]
                skin_x, skin_y = int(skin_point.x * w), int(skin_point.y * h)
                skin_color = frame[skin_y, skin_x]

                iris_point = face_landmarks.landmark[LEFT_IRIS]
                iris_x, iris_y = int(iris_point.x * w), int(iris_point.y * h)
                iris_color = frame[iris_y, iris_x]

                lip_points = np.array(
                    [[int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h)]
                     for i in LIPS], np.int32)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [lip_points], 255)
                mean_lip_color = cv2.mean(frame, mask=mask)[:3]

                # Convert colors to HEX
                skin_hex = bgr_to_hex(skin_color)
                iris_hex = bgr_to_hex(iris_color)
                lips_hex = bgr_to_hex((int(mean_lip_color[0]), int(mean_lip_color[1]), int(mean_lip_color[2])))

                # Generate Color Analysis using Cohere
                if api_key:
                    prompt = f"Provide a fashion and makeup color analysis for a person with skin color {skin_hex}, iris color {iris_hex}, and lip color {lips_hex}."
                    response = co.generate(
                        model='command-xlarge-nightly',
                        prompt=prompt,
                        max_tokens=100
                    )
                    analysis = response.generations[0].text
                    st.subheader("🌈 AI Color Analysis")
                    st.write(analysis)

                # Draw circles on detected points
                cv2.circle(frame, (skin_x, skin_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (iris_x, iris_y), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [lip_points], isClosed=True, color=(0, 0, 255), thickness=1)

        # Show webcam feed
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
