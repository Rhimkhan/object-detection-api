import logging
import sys


def setup_logger(name: str = "object_detection_api") -> logging.Logger:
    """Create and return a configured logger that writes to stdout.

    Prevents duplicate handlers if the logger is fetched multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
