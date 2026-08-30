from __future__ import annotations

from langchain_core.documents import Document

from config.settings import settings
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever


def document_key(document: Document) -> str:
    return str(
        document.metadata.get("chunk_id")
        or f"{document.metadata.get('source','')}|{document.metadata.get('page','')}|{document.page_content[:120]}"
    )


class HybridRetriever:
    """Dense + BM25 retrieval combined with weighted Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_constant: int | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_constant = rrf_constant or settings.rrf_constant
        self.vector_weight = settings.vector_weight if vector_weight is None else vector_weight
        self.bm25_weight = settings.bm25_weight if bm25_weight is None else bm25_weight

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        vector_docs = self.vector_retriever.retrieve(query, k=k)
        bm25_docs = self.bm25_retriever.retrieve(query, k=k)

        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}

        channels = (
            ("vector", vector_docs, self.vector_weight),
            ("bm25", bm25_docs, self.bm25_weight),
        )
        for channel, ranked_docs, weight in channels:
            for rank, incoming in enumerate(ranked_docs, start=1):
                key = document_key(incoming)
                if key not in docs:
                    docs[key] = Document(
                        page_content=incoming.page_content,
                        metadata=dict(incoming.metadata),
                    )
                else:
                    # Preserve diagnostics from every retrieval channel when the
                    # same chunk is returned by both dense and lexical search.
                    docs[key].metadata.update(incoming.metadata)
                docs[key].metadata[f"{channel}_rank"] = rank
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_constant + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        output: list[Document] = []
        for hybrid_rank, (key, score) in enumerate(ranked, start=1):
            doc = docs[key]
            doc.metadata["hybrid_score"] = float(score)
            doc.metadata["hybrid_rank"] = hybrid_rank
            output.append(doc)
        return output
