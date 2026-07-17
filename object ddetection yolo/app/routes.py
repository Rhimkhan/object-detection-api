import os
from datetime import datetime
from typing import Any, Dict

from flask import current_app, jsonify, render_template, request, Response
from werkzeug.utils import secure_filename

from config import Config
from utils.detection import ObjectDetector
from utils.training import YOLOTrainer
from utils.validators import allowed_file, validate_image_size
from utils.logger import setup_logger

# Setup logger for routes
logger = setup_logger(__name__)


def register_routes(app):
    detector = ObjectDetector()
    trainer = YOLOTrainer()

    @app.route("/", methods=["GET"])
    def index():
        logger.info("Index page loaded")
        return render_template(
            "index.html",
            detections=[],
            image_url=None,
            summary="",
            video_url=None,
            webcam_status="offline",
            deployment_info={"render": True, "huggingface": True, "docker": True},
            api_endpoints=["/api/health", "/api/info", "/api/train", "/api/webcam"],
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": detector.model_path})

    @app.route("/api/health", methods=["GET"])
    def api_health():
        return jsonify({"status": "healthy", "model_loaded": detector.model is not None})

    @app.route("/api/info", methods=["GET"])
    def api_info():
        return jsonify({"success": True, "class_count": len(detector.class_names), "classes": detector.class_names})

    @app.route("/detect", methods=["POST"])
    def detect():
        logger.info("Received detection request")
        
        if "image" not in request.files:
            logger.warning("No image part in request")
            return jsonify({"error": "No image part"}), 400

        file = request.files["image"]
        if file.filename == "":
            logger.warning("Empty filename received")
            return jsonify({"error": "No selected file"}), 400

        # Use the centralized validator
        if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File type not allowed. Allowed: png, jpg, jpeg"}), 400

        # Validate file size
        if not validate_image_size(file, Config.MAX_CONTENT_LENGTH):
            logger.warning(f"File too large: {file.filename}")
            return jsonify({"error": f"File size exceeds limit of {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        uploaded_name = f"{base_name}_{timestamp}{extension}"
        upload_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_name)
        output_path = os.path.join(Config.OUTPUT_FOLDER, f"pred_{uploaded_name}")

        file.save(upload_path)
        logger.info(f"File saved: {upload_path}")
        
        result = detector.detect_image(upload_path, save_result=True, output_path=output_path)
        if not result["success"]:
            logger.error(f"Detection failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500

        image_url = "/" + output_path.replace("\\", "/")
        summary = f"Detected {result['total_detections']} object(s)."

        logger.info(f"Detection successful: {result['total_detections']} objects found")
        return render_template(
            "index.html",
            detections=result["detections"],
            image_url=image_url,
            summary=summary,
            video_url=None,
            webcam_status="offline",
            deployment_info={"render": True, "huggingface": True, "docker": True},
            api_endpoints=["/api/health", "/api/info", "/api/train", "/api/webcam"],
        )

    @app.route("/detect/video", methods=["POST"])
    def detect_video():
        logger.info("Received video detection request")
        
        if "video" not in request.files:
            logger.warning("No video part in request")
            return jsonify({"error": "No video part"}), 400

        file = request.files["video"]
        if file.filename == "":
            logger.warning("Empty video filename")
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
            logger.warning(f"Invalid video file type: {file.filename}")
            return jsonify({"error": "File type not allowed. Allowed: png, jpg, jpeg"}), 400

        if not validate_image_size(file, Config.MAX_CONTENT_LENGTH):
            logger.warning(f"Video file too large: {file.filename}")
            return jsonify({"error": f"File size exceeds limit of {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        uploaded_name = f"{base_name}_{timestamp}{extension}"
        upload_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_name)
        output_path = os.path.join(Config.OUTPUT_FOLDER, f"pred_{uploaded_name}")

        file.save(upload_path)
        logger.info(f"Video saved: {upload_path}")
        
        success = detector.detect_video(upload_path, output_path=output_path, show_fps=True)
        if not success:
            logger.error("Video processing failed")
            return jsonify({"error": "Video processing failed"}), 500

        video_url = "/" + output_path.replace("\\", "/")
        logger.info("Video processing successful")
        return render_template(
            "index.html",
            detections=[],
            image_url=None,
            summary="Video processed successfully",
            video_url=video_url,
            webcam_status="offline",
            deployment_info={"render": True, "huggingface": True, "docker": True},
            api_endpoints=["/api/health", "/api/info", "/api/train", "/api/webcam"],
        )

    @app.route("/api/train", methods=["POST"])
    def train_api():
        data_yaml = request.form.get("data_yaml") or request.args.get("data_yaml")
        if not data_yaml:
            logger.warning("Training request missing data_yaml")
            return jsonify({"success": False, "error": "data_yaml is required"}), 400

        epochs = request.form.get("epochs", 10)
        imgsz = request.form.get("imgsz", 640)
        logger.info(f"Training started with data_yaml={data_yaml}, epochs={epochs}, imgsz={imgsz}")
        result = trainer.train_model(data_yaml=data_yaml, epochs=int(epochs), imgsz=int(imgsz))
        return jsonify(result)

    @app.route("/api/webcam")
    def webcam_stream():
        return jsonify({"status": "ready", "message": "Webcam support is available in the local runtime. Connect a camera and use the demo script."})

    @app.route("/api/demo")
    def api_demo():
        return jsonify({
            "success": True,
            "message": "Object detection API ready",
            "features": [
                "Custom training pipeline",
                "Video processing",
                "Webcam support",
                "Render/Hugging Face deployment",
                "Docker support",
                "API endpoints",
                "Professional UI",
            ],
        })
