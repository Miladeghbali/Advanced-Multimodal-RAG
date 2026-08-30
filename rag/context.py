from __future__ import annotations

from langchain_core.documents import Document

from config.settings import settings


def build_context(documents: list[Document], max_chars: int | None = None) -> str:
    """Build a bounded context with stable source labels for citation."""
    if not documents:
        return "No relevant context was retrieved."

    budget = max_chars or settings.max_context_chars
    parts: list[str] = []
    used = 0

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        header = f"[S{i}] source={source}; page={page}; chunk_id={chunk_id}\n"
        text = doc.page_content.strip()
        remaining = budget - used - len(header) - 2
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "…"
        block = header + text
        parts.append(block)
        used += len(block) + 2
        if used >= budget:
            break

    return "\n\n".join(parts)
