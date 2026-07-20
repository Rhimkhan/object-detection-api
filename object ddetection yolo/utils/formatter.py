def format_detections(results) -> list:
    """Convert raw YOLO results into a clean list of dicts for JSON API responses."""
    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
        })
    return detections
