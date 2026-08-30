from __future__ import annotations

import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


def _stable_chunk_id(source: str, page: int | str, text: str) -> str:
    raw = f"{source}|{page}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", "")
        chunk.metadata["chunk_id"] = _stable_chunk_id(source, page, chunk.page_content)
    return chunks
