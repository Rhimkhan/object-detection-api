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
