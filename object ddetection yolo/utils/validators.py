import os
from typing import Iterable


def allowed_file(filename: str, allowed_extensions: Iterable[str]) -> bool:
    """Check whether the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def validate_image_size(file, max_size_bytes: int) -> bool:
    """Ensure the uploaded file does not exceed the configured size limit."""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size <= max_size_bytes
