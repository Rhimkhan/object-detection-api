import os

class Config:
    \"\"\"Centralized application configuration, loaded from environment variables where possible.\"\"\"
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", os.path.join(os.getcwd(), "outputs"))
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.25))
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
