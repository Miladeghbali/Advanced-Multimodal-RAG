from __future__ import annotations

import re
from pathlib import Path

from config.settings import settings


_SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all|any|the)?\s*(previous|prior|system)\s+instructions?",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?system\s+prompt",
    r"developer\s+message",
    r"forget\s+(all|your)\s+(previous|prior)\s+instructions?",
    r"دستور(?:ها|های)?[\s‌]*(?:قبلی|پیشین)[\s‌]*را[\s‌]*(?:نادیده|فراموش)[\s‌]*(?:بگیر|کن)",
    r"(?:پرامپت|پیام)[\s‌]*(?:سیستم|توسعه[\s‌]*دهنده)[\s‌]*را[\s‌]*(?:نشان|نمایش)[\s‌]*(?:بده|ده)",
]


def validate_upload(name: str, size_bytes: int, allowed_extensions: set[str]) -> None:
    suffix = Path(name).suffix.lower()
    if suffix not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {suffix}")
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(
            f"File '{name}' exceeds the {settings.max_upload_mb} MB upload limit."
        )


def detect_prompt_injection(text: str) -> list[str]:
    """Flag common injection phrases for visibility; do not treat as foolproof security."""
    findings: list[str] = []
    lowered = text.lower()
    for pattern in _SUSPICIOUS_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            findings.append(pattern)
    return findings
