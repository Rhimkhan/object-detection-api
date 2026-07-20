import cv2
import numpy as np


def preprocess_image(image_path: str, target_size: tuple = (640, 640)) -> np.ndarray:
    """Load an image, resize it to the model's expected input size, and return it."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    resized = cv2.resize(image, target_size)
    return resized
