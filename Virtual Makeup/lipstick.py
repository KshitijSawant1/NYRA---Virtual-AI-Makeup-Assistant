import cv2
from utils import read_landmarks, apply_makeup

LIP_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
LIP_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LIP_COLOR = (0, 0, 255)  # Red

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = read_landmarks(frame)

    if landmarks:
        lip_upper = [landmarks[i] for i in LIP_UPPER]
        lip_lower = [landmarks[i] for i in LIP_LOWER]
        
        frame = apply_makeup(frame, lip_upper, LIP_COLOR)
        frame = apply_makeup(frame, lip_lower, LIP_COLOR)

    cv2.imshow("Virtual Lipstick", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
