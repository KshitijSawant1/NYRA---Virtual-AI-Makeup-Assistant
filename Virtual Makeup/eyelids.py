import cv2
from utils import read_landmarks, apply_makeup

LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]

LEFT_COLOR = (255, 20, 147)  # Pink
RIGHT_COLOR = (147, 112, 219)  # Purple

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = read_landmarks(frame)

    if landmarks:
        left_eye = [landmarks[i] for i in LEFT_EYESHADOW]
        right_eye = [landmarks[i] for i in RIGHT_EYESHADOW]
        
        frame = apply_makeup(frame, left_eye, LEFT_COLOR)
        frame = apply_makeup(frame, right_eye, RIGHT_COLOR)

    cv2.imshow("Virtual Eyeshadow", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
