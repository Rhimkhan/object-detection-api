import cv2

from utils.detection import ObjectDetector


def main() -> None:
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam detected")
        return

    print("Webcam started. Press q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_image(frame)
        annotated = result.get("annotated_image") or frame
        cv2.imshow("Object Detection Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# webcam fps logging planned

import time

def calculate_fps(prev_time):
    """Return current FPS and updated timestamp, given the previous frame's timestamp."""
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    return round(fps, 1), current_time
