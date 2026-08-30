from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from core.embeddings import create_embeddings


class VectorStoreManager:
    def __init__(
        self, persist_dir: str | Path | None = None, collection_name: str | None = None
    ) -> None:
        self.persist_path = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = create_embeddings()
        self._embedding_cache: dict[str, list[float]] = {}
        self._create_store()

    def _create_store(self) -> None:
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        self.store.add_documents(documents=documents, ids=ids)
        self._embedding_cache.clear()

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        try:
            return self.store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            return [(doc, 0.0) for doc in self.similarity_search(query, k=k)]

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(documents)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        missing_keys: list[str] = []

        for i, doc in enumerate(documents):
            key = str(doc.metadata.get("chunk_id") or doc.page_content)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                missing_texts.append(doc.page_content)
                missing_indices.append(i)
                missing_keys.append(key)

        if missing_texts:
            vectors = self.embeddings.embed_documents(missing_texts)
            for i, key, vector in zip(missing_indices, missing_keys, vectors):
                vector_list = list(vector)
                self._embedding_cache[key] = vector_list
                results[i] = vector_list

        return [list(vector) for vector in results if vector is not None]

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def count(self) -> int:
        try:
            return int(self.store._collection.count())
        except Exception:
            return 0

    def clear(self) -> None:
        # Delete only this configured collection. Do not remove the whole Chroma
        # persistence directory because it may contain other collections and may
        # also be locked by SQLite on Windows.
        try:
            self.store.delete_collection()
        except Exception:
            pass
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._embedding_cache.clear()
        self._create_store()
