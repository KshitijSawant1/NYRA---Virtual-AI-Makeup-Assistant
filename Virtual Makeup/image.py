import cv2
import argparse
from utils import *

# Features to apply makeup
face_elements = ["LIP_LOWER", "LIP_UPPER", "EYEBROW_LEFT", "EYEBROW_RIGHT"]

# Define makeup colors in BGR format
colors_map = {
    "LIP_UPPER": [0, 0, 255],  # Red Lips
    "LIP_LOWER": [0, 0, 255],
    "EYEBROW_LEFT": [19, 69, 139],  # Dark Brown Eyebrows
    "EYEBROW_RIGHT": [19, 69, 139]
}

def apply_makeup(image_path):
    face_connections = [face_points[idx] for idx in face_elements]
    colors = [colors_map[idx] for idx in face_elements]

    image = cv2.imread(image_path)
    mask = np.zeros_like(image)
    face_landmarks = read_landmarks(image)

    mask = add_mask(mask, face_landmarks, face_connections, colors)
    output = cv2.addWeighted(image, 1.0, mask, 0.3, 0)

    cv2.imshow("Virtual Makeup", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply virtual makeup on an image")
    parser.add_argument("--img", type=str, required=True, help="Path to the image")
    args = parser.parse_args()
    apply_makeup(args.img)
