from flask import jsonify


def register_error_handlers(app):
    # Attach JSON error handlers to the Flask app for consistent API responses

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": "The requested resource does not exist."}), 404

    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({"error": "Payload too large", "message": "Uploaded file exceeds the size limit."}), 413

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "message": "Something went wrong on our end."}), 500
