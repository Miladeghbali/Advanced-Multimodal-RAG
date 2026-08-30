from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.settings import settings


class CorpusStore:
    """Persist the lexical corpus, index metadata, and optional multimodal assets."""

    def __init__(self, state_dir: str | Path | None = None) -> None:
        self.state_dir = Path(state_dir or settings.rag_state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.state_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.state_dir / "corpus.jsonl"
        self.index_meta_path = self.state_dir / "index_meta.json"

    @staticmethod
    def index_signature() -> str:
        raw = "|".join(
            [
                settings.embedding_model,
                str(settings.chunk_size),
                str(settings.chunk_overlap),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): CorpusStore._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [CorpusStore._json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)

    def save_asset(self, filename: str, data: bytes) -> Path:
        safe_name = Path(filename).name
        digest = hashlib.sha256(data).hexdigest()[:12]
        target = self.assets_dir / f"{digest}_{safe_name}"
        target.write_bytes(data)
        return target

    def save(self, documents: list[Document]) -> None:
        corpus_lines: list[str] = []
        for doc in documents:
            payload = {
                "page_content": doc.page_content,
                "metadata": self._json_safe(dict(doc.metadata)),
            }
            corpus_lines.append(json.dumps(payload, ensure_ascii=False))
        corpus_text = "\n".join(corpus_lines) + ("\n" if corpus_lines else "")
        self._atomic_write(self.corpus_path, corpus_text)

        meta_text = json.dumps(
            {
                "index_signature": self.index_signature(),
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "chunk_count": len(documents),
            },
            ensure_ascii=False,
            indent=2,
        )
        self._atomic_write(self.index_meta_path, meta_text)

    def load(self) -> list[Document]:
        if not self.corpus_path.exists():
            return []
        documents: list[Document] = []
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                Document(
                    page_content=payload["page_content"],
                    metadata=payload.get("metadata", {}),
                )
            )
        return documents

    def is_compatible(self) -> bool:
        if not self.index_meta_path.exists():
            return not self.corpus_path.exists()
        try:
            meta = json.loads(self.index_meta_path.read_text(encoding="utf-8"))
            return meta.get("index_signature") == self.index_signature()
        except Exception:
            return False

    def clear(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
