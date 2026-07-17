import os
import threading
import webbrowser

from config import Config
from app import app
from app.error_handlers import register_error_handlers
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Apply configuration
app.config.from_object(Config)

# Register error handlers
register_error_handlers(app)

# Ensure upload folder exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

logger.info(f"Starting Object Detection API with config: MODEL_PATH={Config.MODEL_PATH}, DEBUG={Config.DEBUG}")


def open_browser() -> None:
    \"\"\"Open browser automatically after app starts.\"\"\"
    webbrowser.open("http://127.0.0.1:5001", new=0, autoraise=True)


if __name__ == "__main__":
    if not Config.DEBUG:
        threading.Timer(1.5, open_browser).start()
    else:
        logger.info("Running in debug mode - browser will not open automatically")
    
    app.run(
        debug=Config.DEBUG,
        host="0.0.0.0",
        port=5001,
        use_reloader=False
    )
