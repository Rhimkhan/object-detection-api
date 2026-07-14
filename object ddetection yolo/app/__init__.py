import os

from flask import Flask
from flask_cors import CORS

from app.routes import register_routes


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
    app.config["OUTPUT_FOLDER"] = os.path.join("static", "outputs")
    app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "mp4", "avi", "mov"}

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    CORS(app)
    register_routes(app)
    return app


app = create_app()
