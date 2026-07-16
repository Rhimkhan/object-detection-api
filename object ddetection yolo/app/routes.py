import os
from datetime import datetime
from typing import Any, Dict

from flask import current_app, jsonify, render_template, request, Response
from werkzeug.utils import secure_filename

from utils.detection import ObjectDetector
from utils.training import YOLOTrainer


def register_routes(app):
    detector = ObjectDetector()
    trainer = YOLOTrainer()

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]

    @app.route("/", methods=["GET"])
    def index():
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
        if "image" not in request.files:
            return jsonify({"error": "No image part"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        uploaded_name = f"{base_name}_{timestamp}{extension}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_name)
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"pred_{uploaded_name}")

        file.save(upload_path)
        result = detector.detect_image(upload_path, save_result=True, output_path=output_path)
        if not result["success"]:
            return jsonify(result), 500

        image_url = "/" + output_path.replace("\\", "/")
        summary = f"Detected {result['total_detections']} object(s)."

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
        if "video" not in request.files:
            return jsonify({"error": "No video part"}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        uploaded_name = f"{base_name}_{timestamp}{extension}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_name)
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"pred_{uploaded_name}")

        file.save(upload_path)
        success = detector.detect_video(upload_path, output_path=output_path, show_fps=True)
        if not success:
            return jsonify({"error": "Video processing failed"}), 500

        video_url = "/" + output_path.replace("\\", "/")
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
            return jsonify({"success": False, "error": "data_yaml is required"}), 400

        epochs = request.form.get("epochs", 10)
        imgsz = request.form.get("imgsz", 640)
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

# TODO: add more routes
