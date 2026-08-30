from __future__ import annotations

import base64
from pathlib import Path


def create_image_content(image_path: str | Path, question: str) -> list[dict]:
    path = Path(image_path)
    mime = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": question},
        {"type": "image", "base64": encoded, "mime_type": mime},
    ]
