from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import settings


class Reranker:
    """Second-stage Cross-Encoder re-ranker over a small candidate set."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(
            self.model_name,
            device=settings.reranker_device,
            max_length=512,
        )

    def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=settings.reranker_batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)

        output: list[Document] = []
        for rank, (doc, score) in enumerate(ranked[:top_k], start=1):
            doc.metadata["reranker_model"] = self.model_name
            doc.metadata["reranker_score"] = float(score)
            doc.metadata["reranker_rank"] = rank
            output.append(doc)
        return output
