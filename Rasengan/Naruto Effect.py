import cv2
import mediapipe as mp
import numpy as np

# Load the Rasengan video
rasengan_video = cv2.VideoCapture('Rasengan/Rasengan GIF.mp4')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to overlay video frame on hand
def overlay_rasengan(frame, rasengan_frame, center, angle):
    (h, w) = rasengan_frame.shape[:2]

    # Resize and rotate Rasengan
    rasengan_resized = cv2.resize(rasengan_frame, (100, 100))  # Resize to fit the palm
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_rasengan = cv2.warpAffine(rasengan_resized, M, (w, h))

    # Create mask for the Rasengan
    gray = cv2.cvtColor(rotated_rasengan, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Overlay Rasengan on the frame
    x, y = center
    top_left_x = x - w // 2
    top_left_y = y - h // 2

    # Handle out-of-frame errors
    if top_left_x < 0 or top_left_y < 0 or top_left_x + w > frame.shape[1] or top_left_y + h > frame.shape[0]:
        return frame

    roi = frame[top_left_y:top_left_y + h, top_left_x:top_left_x + w]
    masked_frame = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    added_frame = cv2.add(masked_frame, rotated_rasengan)

    frame[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = added_frame
    return frame

# Webcam Feed
cap = cv2.VideoCapture(0)
angle = 0  # Initial rotation angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not access the webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    ret_vid, rasengan_frame = rasengan_video.read()
    if not ret_vid:
        rasengan_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
        ret_vid, rasengan_frame = rasengan_video.read()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            palm_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
            palm_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)

            # Overlay spinning Rasengan
            frame = overlay_rasengan(frame, rasengan_frame, (palm_x, palm_y), angle)
            angle = (angle + 5) % 360  # Rotate continuously

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display frame
    cv2.imshow("🌀 Spinning Rasengan Effect", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
rasengan_video.release()
cv2.destroyAllWindows()
