import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import google.generativeai as genai
import re
import time
import os

# Ensure the directory exists
capture_directory = "Captured_Images"
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

st.set_page_config(
    page_title="NYRA",
    page_icon="https://github.com/user-attachments/assets/d63378a3-baaf-4afe-9540-cb5674a73e12",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define Facial Landmarks
LEFT_CHEEK_CONTOUR = [234, 93, 132, 58, 172, 136, 150]
RIGHT_CHEEK_CONTOUR = [454, 323, 361, 288, 397, 365, 379]
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]

st.markdown(
    """
    <style>
    hr {
        border: 1px solid #eee;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("NYRA - Virtual AI Makeup Assistant")
# 📷 Webcam Controls at the Top
st.sidebar.title("📷 Webcam Controls")
if "webcam_enabled" not in st.session_state:
    st.session_state.webcam_enabled = False

# Toggle Button for Webcam
if st.sidebar.button("🎥 Start Webcam" if not st.session_state.webcam_enabled else "🛑 Stop Webcam"):
    st.session_state.webcam_enabled = not st.session_state.webcam_enabled

# Webcam Status Display
if st.session_state.webcam_enabled:
    st.sidebar.success("Webcam is ON")
else:
    st.sidebar.error("Webcam is OFF")
    
# Button to capture image with session state
if st.sidebar.button("📸 Capture & Save Image"):
    st.session_state.capture_image = True  # Set the flag to True when button is clicked
# Add a state for capturing images
if "capture_image" not in st.session_state:
    st.session_state.capture_image = False


st.sidebar.markdown("---")
# Add a toggle for AI color analysis
st.sidebar.title("🌈 AI Color Analysis")
# Initialize Cohere Client
api_key = st.sidebar.text_input("Enter Cohere API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# Button to trigger color analysis
generate_analysis = st.sidebar.button("Generate Color Analysis")
if st.sidebar.button("🔄 Reset AI Analysis", key="reset_ai_analysis"):
    st.session_state.analysis_done = False

# Define landmarks for skin, iris, lips
SKIN_LANDMARK = 10  # Tip of the nose
LEFT_IRIS = 468
RIGHT_IRIS = 473
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Convert BGR to HEX
def bgr_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])

# Display color swatches in Streamlit
def color_swatch(hex_code):
    return f'<div style="display:inline-block;width:20px;height:20px;background-color:{hex_code};border-radius:3px;margin-right:5px;"></div>'

st.sidebar.markdown("---")
# Makeup Feature Toggles
st.sidebar.title("💄 Makeup Controls")


# Blush Settings with Visual Color Swatches
# Blush Toggle
st.sidebar.title("✨ Blush Settings")
if "blush_enabled" not in st.session_state:
    st.session_state.blush_enabled = False
if st.sidebar.button("Enable Blush" if not st.session_state.blush_enabled else "❌ Disable Blush"):
    st.session_state.blush_enabled = not st.session_state.blush_enabled
opacity_blush = st.sidebar.slider("Blush Intensity", 0.1, 1.0, 0.5)

# Predefined skin tone colors with color boxes
predefined_skin_tones = {
    "Light Beige": "#FAD7B6",
    "Medium Beige": "#D2A679",
    "Olive": "#A67B5B",
    "Tan": "#C68642",
    "Deep Bronze": "#8D5524"
}

# Store selected blush color in session state
if "blush_color" not in st.session_state:
    st.session_state.blush_color = "#8D5524"  # Default color

blush_color_mode = st.sidebar.radio("Blush Color Mode", ["Match Skin Tone", "Predefined Color", "Custom Color"])

# Function to create color swatches using HTML
def color_box(color_hex):
    return f"""
    <div style="
        display: inline-block;
        width: 20px;
        height: 20px;
        background-color: {color_hex};
        border-radius: 3px;
        margin-right: 10px;
        border: 1px solid #000;
    "></div>
    """

# Predefined Color Selection (Side by Side)
if blush_color_mode == "Predefined Color":
    st.sidebar.markdown("**Choose a Skin Tone:**")
    # Arrange swatches side by side
    cols = st.sidebar.columns(2)
    for index, (tone_name, tone_hex) in enumerate(predefined_skin_tones.items()):
        with cols[index % 2]:  # Alternating between two columns
            # Show color buttons without extra text
            if st.button(f"{tone_name}", key=tone_name):
                st.session_state.blush_color = tone_hex  # Update the session state immediately
            st.markdown(color_box(tone_hex), unsafe_allow_html=True)

elif blush_color_mode == "Custom Color":
    st.session_state.blush_color = st.sidebar.color_picker("Choose Blush Color", "#FF69B4")

# Use the selected blush color from the session state
blush_color = st.session_state.blush_color
st.sidebar.markdown("---")
# Lipstick Toggle
st.sidebar.title("✨ Lip Color Settings")
if "lipstick_enabled" not in st.session_state:
    st.session_state.lipstick_enabled = False
if st.sidebar.button("Enable Lipstick" if not st.session_state.lipstick_enabled else "❌ Disable Lipstick"):
    st.session_state.lipstick_enabled = not st.session_state.lipstick_enabled
lipstick_opacity = st.sidebar.slider("Lipstick Intensity", 0.1, 0.5, 0.2)

# Predefined lipstick shades
predefined_lipstick_colors = {
    "Classic Red": "#FF0000",
    "Berry Pink": "#C71585",
    "Nude": "#E3AA94",
    "Coral": "#FF7F50",
    "Wine": "#722F37"
}

# Store selected lipstick color in session state
if "lipstick_color" not in st.session_state:
    st.session_state.lipstick_color = "#FF0000"  # Default color

lipstick_color_mode = st.sidebar.radio("Lipstick Color Mode", ["Predefined Color", "Custom Color"])

# Function to create color swatches for lipstick
def lipstick_color_box(color_hex):
    return f"""
    <div style="
        display: inline-block;
        width: 20px;
        height: 20px;
        background-color: {color_hex};
        border-radius: 3px;
        margin-right: 10px;
        border: 1px solid #000;
    "></div>
    """

# Predefined Color Selection for Lipstick (Side by Side)
if lipstick_color_mode == "Predefined Color":
    st.sidebar.markdown("**Choose a Lipstick Shade:**")
    # Arrange swatches side by side
    cols = st.sidebar.columns(2)
    for index, (shade_name, shade_hex) in enumerate(predefined_lipstick_colors.items()):
        with cols[index % 2]:  # Alternating between two columns
            if st.button(f"{shade_name}", key=f"lipstick_{shade_name}"):
                st.session_state.lipstick_color = shade_hex
            st.markdown(lipstick_color_box(shade_hex), unsafe_allow_html=True)

elif lipstick_color_mode == "Custom Color":
    st.session_state.lipstick_color = st.sidebar.color_picker("Choose Custom Lipstick Color", "#FF0000")

# Use the selected lipstick color from the session state
lipstick_color = st.session_state.lipstick_color
st.sidebar.markdown("---")
# Eyeshadow Toggle
st.sidebar.title("✨ Eye Shadow Settings")
if "eyeshadow_enabled" not in st.session_state:
    st.session_state.eyeshadow_enabled = False
if st.sidebar.button("Enable Eyeshadow" if not st.session_state.eyeshadow_enabled else "❌ Disable Eyeshadow"):
    st.session_state.eyeshadow_enabled = not st.session_state.eyeshadow_enabled
eyeshadow_color = st.sidebar.color_picker("Eyeshadow Color", "#FF69B4")
eyeshadow_opacity = st.sidebar.slider("Eyeshadow Intensity", 0.1, 0.3, 0.2)

# Convert Hex Color to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

# Function to get average skin tone from contour points
def get_average_skin_tone(frame, points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    mean_val = cv2.mean(frame, mask=mask)
    return (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))  # Return BGR values

# Smoothing and Lightening Controls
st.sidebar.markdown("---")
st.sidebar.title("✨ Face Enhancements")

# Smoothing and Brightness Sliders
smoothing_intensity = st.sidebar.slider("Smooth Skin Level", 0, 100, 30)
brightness_increase = st.sidebar.slider("Increase Brightness", 0, 100, 30)


def color_swatch_with_tooltip(hex_code):
    return f"""
    <div title="{hex_code}" style="
        display: inline-block;
        width: 40px;
        height: 40px;
        background-color: {hex_code};
        margin: 5px;
        border-radius: 5px;
        position: relative;
        border: 1px solid #ddd;
    ">
    </div>
    """

FRAME_WINDOW = st.image([])

if st.session_state.webcam_enabled:
    cap = cv2.VideoCapture(0)
    skin_hex, iris_hex, lips_hex = None, None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access the webcam.")
            break

        # Convert frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Extract skin color
                skin_point = face_landmarks.landmark[SKIN_LANDMARK]
                skin_x, skin_y = int(skin_point.x * w), int(skin_point.y * h)
                skin_color = frame[skin_y, skin_x]

                # Extract iris color
                iris_point = face_landmarks.landmark[LEFT_IRIS]
                iris_x, iris_y = int(iris_point.x * w), int(iris_point.y * h)
                iris_color = frame[iris_y, iris_x]

                # Extract lips color
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
                
                blush_overlay = frame.copy()
                mask = np.zeros_like(frame, dtype=np.uint8)

                # -------- Apply Blush -------- #
                if st.session_state.blush_enabled:
                    left_cheek_points = np.array(
                        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                         for i in LEFT_CHEEK_CONTOUR], np.int32)
                    right_cheek_points = np.array(
                        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                         for i in RIGHT_CHEEK_CONTOUR], np.int32)

                    if blush_color_mode == "Match Skin Tone":
                        left_cheek_color = get_average_skin_tone(frame, left_cheek_points)
                        right_cheek_color = get_average_skin_tone(frame, right_cheek_points)
                    else:
                        left_cheek_color = hex_to_bgr(blush_color)
                        right_cheek_color = hex_to_bgr(blush_color)

                    cv2.fillPoly(blush_overlay, [left_cheek_points], left_cheek_color)
                    cv2.fillPoly(blush_overlay, [right_cheek_points], right_cheek_color)
                    blush_overlay = cv2.GaussianBlur(blush_overlay, (35, 35), 10)
                    mask_blush = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(mask_blush, [left_cheek_points], (255, 255, 255))
                    cv2.fillPoly(mask_blush, [right_cheek_points], (255, 255, 255))
                    mask_blush = cv2.GaussianBlur(mask_blush, (35, 35), 10)
                    mask_blush = mask_blush.astype(float) / 255

                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * (1 - mask_blush[:, :, c] * opacity_blush) + blush_overlay[:, :, c] * (mask_blush[:, :, c] * opacity_blush)

                # -------- Apply Lipstick -------- #
                if st.session_state.lipstick_enabled:
                    upper_lip_points = np.array(
                        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                         for i in UPPER_LIP], np.int32)
                    lower_lip_points = np.array(
                        [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                         for i in LOWER_LIP], np.int32)

                    lipstick_bgr = hex_to_bgr(lipstick_color)
                    cv2.fillPoly(mask, [upper_lip_points], lipstick_bgr)
                    cv2.fillPoly(mask, [lower_lip_points], lipstick_bgr)

                    # -------- Apply Eyeshadow -------- #
                    if st.session_state.eyeshadow_enabled:
                        eyeshadow_points = [
                            np.array(
                                [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                for i in LEFT_EYESHADOW], np.int32
                            ),
                            np.array(
                                [[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                for i in RIGHT_EYESHADOW], np.int32
                            )
                        ]

                        eye_bgr = hex_to_bgr(eyeshadow_color)
                        for eye_points in eyeshadow_points:
                            cv2.fillPoly(mask, [eye_points], eye_bgr)

                # -------- Blend Everything -------- #
                frame = cv2.addWeighted(frame, 1, mask, lipstick_opacity, 0)
                frame = cv2.addWeighted(frame, 1, mask, eyeshadow_opacity, 0)
                
                # Apply Face Smoothing using Bilateral Filter
                smoothed_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=smoothing_intensity, sigmaSpace=smoothing_intensity)

                # Lighten the Frame (Increase Brightness)
                brightness_matrix = np.ones_like(smoothed_frame, dtype="uint8") * brightness_increase
                brightened_frame = cv2.add(smoothed_frame, brightness_matrix)
                if brightened_frame is not None:
                    display_frame = cv2.cvtColor(brightened_frame, cv2.COLOR_BGR2RGB)
                else:
                    st.warning("⚠️ No valid frame captured, skipping display update.")

        # Convert frame to RGB for display
        display_frame = cv2.cvtColor(brightened_frame, cv2.COLOR_BGR2RGB)
            
        # Save the image only once after applying all makeup
        if st.session_state.capture_image:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = os.path.join(capture_directory, f"makeup_applied_{timestamp}.png")
            
            # Convert the final processed frame back to BGR format before saving
            processed_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            success = cv2.imwrite(image_filename, processed_frame_bgr)  # Save the final frame with makeup applied
            
            if success:
                st.success(f"✅ Image with makeup saved successfully as {image_filename}")
            else:
                st.error("❌ Failed to save the image. Check the directory permissions or path.")
            
            # Reset the flag after saving
            st.session_state.capture_image = False
            
        FRAME_WINDOW.image(display_frame)
        # AI Color Analysis with Gemini AI
        if generate_analysis and api_key and not st.session_state.analysis_done:
            if skin_hex and iris_hex and lips_hex:
                prompt = (
                    f"Provide an in-depth fashion, makeup, and color palette analysis for someone with "
                    f"skin tone {skin_hex}, iris color {iris_hex}, and lip color {lips_hex}. "
                    f"Suggest suitable clothing colors, makeup shades, and hair colors based on seasonal color theory. "
                    f"Ensure that the response includes HEX color codes for each recommendation. "
                    f"Provide the result in bullet points."
                )

                try:
                    with st.spinner("🔍 Analyzing colors with Gemini AI..."):
                        # Use the correct model name from API response
                        model = genai.GenerativeModel("gemini-1.5-pro")  # Change this based on API output
                        
                        response = model.generate_content(prompt)
                        analysis = response.text  # Extract AI-generated text
                        
                        st.session_state.analysis_done = True  # Prevents multiple analyses until button is clicked again
                        
                        # **Extract hex codes from the AI response using regex**
                        hex_codes = re.findall(r'#(?:[0-9a-fA-F]{3}){1,2}', analysis)

                        # **Display AI analysis results**
                        st.subheader("🌈 AI Color Analysis Result")
                        st.markdown(f"**Skin Color:** {color_swatch(skin_hex)} {skin_hex}", unsafe_allow_html=True)
                        st.markdown(f"**Iris Color:** {color_swatch(iris_hex)} {iris_hex}", unsafe_allow_html=True)
                        st.markdown(f"**Lip Color:** {color_swatch(lips_hex)} {lips_hex}", unsafe_allow_html=True)
                        st.markdown(analysis)

                        # **Display extracted AI-suggested color swatches**
                        if hex_codes:
                            st.subheader("🎨 AI Suggested Color Swatches")
                            color_swatch_html = "".join([color_swatch_with_tooltip(code) for code in hex_codes])
                            st.markdown(color_swatch_html, unsafe_allow_html=True)
                        else:
                            st.info("No HEX color codes detected in the AI analysis.")

                except Exception as e:
                    st.error(f"❌ Error during AI analysis: {str(e)}")


    cap.release()
    st.session_state.analysis_done = False  # Reset analysis state after release
else:
    st.info("👆 Enable the webcam from the sidebar to start the feed.")