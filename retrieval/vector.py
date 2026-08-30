from __future__ import annotations

from langchain_core.documents import Document

from core.vector_store import VectorStoreManager


class VectorRetriever:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        scored = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        documents: list[Document] = []
        for rank, (raw_doc, score) in enumerate(scored, start=1):
            doc = Document(page_content=raw_doc.page_content, metadata=dict(raw_doc.metadata))
            doc.metadata["vector_rank"] = rank
            doc.metadata["vector_score"] = float(score)
            documents.append(doc)
        return documents
