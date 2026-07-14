import os
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        default_path = os.path.join("models", "best.pt")
        self.model_path = model_path or default_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = []
        self.load_model()

    def load_model(self) -> bool:
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                self.model = YOLO("yolov8n.pt")
            self.class_names = self.model.names
            self.model.eval()
            return True
        except Exception as exc:
            print(f"Failed to load model: {exc}")
            return False

    def detect_image(self, image_path: str | np.ndarray, save_result: bool = False, output_path: str | None = None) -> Dict[str, Any]:
        try:
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image from {image_path}")
            else:
                image = image_path

            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, stream=False, verbose=False)
            detections: List[Dict[str, Any]] = []
            annotated_image = None

            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for box, confidence, class_id in zip(boxes, confidences, classes):
                        detections.append(
                            {
                                "class_id": int(class_id),
                                "class_name": self.class_names[int(class_id)],
                                "confidence": round(float(confidence), 3),
                                "bbox": [round(float(value), 2) for value in box.tolist()],
                                "bbox_normalized": [
                                    round(float(box[0] / image.shape[1]), 4),
                                    round(float(box[1] / image.shape[0]), 4),
                                    round(float(box[2] / image.shape[1]), 4),
                                    round(float(box[3] / image.shape[0]), 4),
                                ],
                            }
                        )

                    annotated_image = result.plot()

                    if save_result:
                        if output_path is None:
                            output_dir = os.path.join("static", "outputs")
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, "prediction.jpg")
                        else:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        success = cv2.imwrite(output_path, annotated_image)
                        if not success:
                            raise RuntimeError(f"Unable to save annotated image to {output_path}")

            return {
                "success": True,
                "detections": detections,
                "total_detections": len(detections),
                "annotated_image": annotated_image,
                "image_shape": image.shape[:2],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def detect_objects(self, image_path: str | np.ndarray, save_result: bool = False, output_path: str | None = None):
        return self.detect_image(image_path, save_result=save_result, output_path=output_path)

    def detect_video(self, video_path: str, output_path: str | None = None, show_fps: bool = False) -> bool:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if output_path is None:
                output_dir = os.path.join("static", "outputs")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "prediction.mp4")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            fps_display = fps
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect_image(frame)
                annotated_frame = result.get("annotated_image") or frame
                if show_fps:
                    cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                out.write(annotated_frame)
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

            cap.release()
            out.release()
            return True
        except Exception as exc:
            print(f"Video detection failed: {exc}")
            return False
