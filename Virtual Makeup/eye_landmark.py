import cv2
from utils import read_landmarks

LEFT_EYE = [263, 362, 373, 374, 380, 381, 382, 386, 387, 388, 390, 398, 466, 362]
RIGHT_EYE = [33, 133, 160, 161, 159, 158, 157, 154, 153, 145, 144, 163, 246, 33]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = read_landmarks(frame)

    if landmarks:
        for i in LEFT_EYE:
            x, y = landmarks[i]
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        for i in RIGHT_EYE:
            x, y = landmarks[i]
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Eye Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
