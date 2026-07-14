# Professional Object Detection Platform

This project provides a Flask-based object detection application with:
- image detection using YOLO
- video processing
- custom training support
- webcam-ready local runtime guidance
- REST API endpoints
- deployment files for Render and Docker

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Docker

```bash
docker build -t object-detection-platform .
docker run -p 5001:5001 object-detection-platform
```

## Deployment

- Render: use the included render.yaml
- Hugging Face Spaces: deploy the app as a Python app with requirements.txt


Updated on 15 July 2026
