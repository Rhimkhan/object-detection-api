from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint used by load balancers and uptime monitors."""
    return jsonify({"status": "ok"}), 200
