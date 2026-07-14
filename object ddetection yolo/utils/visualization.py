from typing import List, Dict, Any


def summarize_detections(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return "No objects detected."

    names = [item["class"] for item in detections]
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{label}: {count}" for label, count in counts.items())
