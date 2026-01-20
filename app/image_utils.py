import cv2
import numpy as np

def preprocess_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid or unsupported image file")

    image = cv2.resize(image, (224, 224))
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size

    brightness = gray.mean()

    return {
        "edge_density": float(edge_density),
        "brightness": float(brightness)
    }
