import cv2
import numpy as np
import face_recognition
from PIL import Image

# Load Hat Image
HAT = Image.open("hat.png").convert("RGBA")

# Load Image
image = face_recognition.load_image_file("restored_output.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)

if face_landmarks_list:
    top_head = face_landmarks_list[0]["left_eyebrow"][0]  # Using eyebrow as ref
    x, y = top_head
    hat_width = 200
    hat_height = 150

    HAT = HAT.resize((hat_width, hat_height), Image.ANTIALIAS)
    pil_image.paste(HAT, (x - 100, y - 120), HAT)

# Save or Show
pil_image.save("hat_output.jpg")
pil_image.show()
