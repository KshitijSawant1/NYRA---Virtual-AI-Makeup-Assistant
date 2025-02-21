import cv2
import numpy as np
import face_recognition
from PIL import Image

# Load Glasses Image
GLASSES = Image.open("glasses.png").convert("RGBA")

# Load Image
image = face_recognition.load_image_file("restored_output.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)

if face_landmarks_list:
    left_eye = face_landmarks_list[0]["left_eye"][0]
    right_eye = face_landmarks_list[0]["right_eye"][3]
    x1, y1 = left_eye
    x2, y2 = right_eye

    glasses_width = abs(x2 - x1) + 50
    glasses_height = glasses_width // 3

    GLASSES = GLASSES.resize((glasses_width, glasses_height), Image.ANTIALIAS)
    pil_image.paste(GLASSES, (x1 - 20, y1 - 20), GLASSES)

# Save or Show
pil_image.save("glasses_output.jpg")
pil_image.show()
