import cv2

# Load the eyewear image with transparency (PNG format)
eyewear_image = cv2.imread("Virtual Makeup/Eyewear_Collection.png", cv2.IMREAD_UNCHANGED)

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

def add_eyewear(face_frame, eyewear, x, y, w, h):
    eyewear_resized = cv2.resize(eyewear, (w, int(h / 3)))  # Resize eyewear to fit the face width
    ew_height, ew_width, _ = eyewear_resized.shape

    for i in range(ew_height):
        for j in range(ew_width):
            # Skip transparent pixels
            if eyewear_resized[i, j][3] != 0:
                face_frame[y + i, x + j] = eyewear_resized[i, j][:3]
    return face_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Add eyewear to detected faces
    for (x, y, w, h) in faces:
        frame = add_eyewear(frame, eyewear_image, x, y + int(h / 4), w, h)

    cv2.imshow("Virtual Eyewear", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
