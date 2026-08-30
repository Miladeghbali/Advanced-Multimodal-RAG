from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from core.vector_store import VectorStoreManager


class MMRRetriever:
    """Applies Maximum Marginal Relevance over hybrid-search candidates."""

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int = 6,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        if not documents:
            return []
        k = min(k, len(documents))

        doc_embeddings = np.asarray(self.vector_store.embed_documents(documents), dtype=float)
        query_embedding = np.asarray(self.vector_store.embed_query(query), dtype=float)

        selected: list[int] = []
        remaining = set(range(len(documents)))

        while remaining and len(selected) < k:
            best_index: int | None = None
            best_score = -float("inf")

            for index in remaining:
                relevance = self.cosine(query_embedding, doc_embeddings[index])
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self.cosine(doc_embeddings[index], doc_embeddings[j]) for j in selected
                    )
                score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index is None:
                break
            selected.append(best_index)
            remaining.remove(best_index)

        result = [documents[i] for i in selected]
        for rank, doc in enumerate(result, start=1):
            doc.metadata["mmr_rank"] = rank
        return result
