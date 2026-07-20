import os
from typing import Any, Dict

from ultralytics import YOLO


class YOLOTrainer:
    def __init__(self) -> None:
        self.default_model = "yolov8n.pt"

    def train_model(
        self,
        data_yaml: str,
        model: str = "yolov8n.pt",
        epochs: int = 10,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "custom_model",
        device: str = "cpu",
    ) -> Dict[str, Any]:
        try:
            model_instance = YOLO(model or self.default_model)
            train_result = model_instance.train(
                data=data_yaml,
                epochs=int(epochs),
                imgsz=int(imgsz),
                project=project,
                name=name,
                exist_ok=True,
                pretrained=True,
                device=device,
                workers=0,
                stream=False,
                verbose=False,
            )
            best_model_path = os.path.join(project, name, "weights", "best.pt")
            if os.path.exists(best_model_path):
                return {
                    "success": True,
                    "message": "Training completed successfully.",
                    "model_path": best_model_path,
                    "train_dir": os.path.join(project, name),
                    "results": str(train_result),
                }
            return {
                "success": True,
                "message": "Training completed, but no weights were written.",
                "model_path": best_model_path,
                "train_dir": os.path.join(project, name),
                "results": str(train_result),
            }
        except Exception as exc:  # pragma: no cover - defensive path
            return {"success": False, "error": str(exc)}

# added training notes

# checkpoint saving reviewed

def get_training_summary(epochs: int, batch_size: int, model_name: str) -> str:
    """Return a human-readable summary of training configuration for logging."""
    return f"Training {model_name} for {epochs} epochs with batch size {batch_size}"
