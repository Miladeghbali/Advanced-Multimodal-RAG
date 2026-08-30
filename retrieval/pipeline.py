from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from langchain_core.documents import Document

from config.settings import settings
from core.vector_store import VectorStoreManager
from evaluation.langsmith_eval import optional_traceable
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.mmr import MMRRetriever
from retrieval.reranker import Reranker
from retrieval.vector import VectorRetriever


@dataclass
class RetrievalResult:
    query: str
    hybrid: list[Document]
    mmr: list[Document]
    reranked: list[Document]
    timings_ms: dict[str, float] = field(default_factory=dict)


class AdvancedRetrievalPipeline:
    """Required retrieval stack: Dense + BM25 -> weighted RRF -> MMR -> Cross-Encoder."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        corpus: list[Document],
        reranker: Reranker,
    ) -> None:
        self.vector_store = vector_store
        self.vector = VectorRetriever(vector_store)
        self.bm25 = BM25Retriever(corpus)
        self.hybrid = HybridRetriever(self.vector, self.bm25)
        self.mmr = MMRRetriever(vector_store)
        self.reranker = reranker

    @optional_traceable("retrieval.advanced_pipeline")
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mmr_k: int | None = None,
        mmr_lambda: float | None = None,
        rerank_top_k: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.top_k
        mmr_k = mmr_k or settings.mmr_k
        mmr_lambda = settings.mmr_lambda if mmr_lambda is None else mmr_lambda
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        timings: dict[str, float] = {}
        total_start = perf_counter()

        start = perf_counter()
        hybrid_docs = self.hybrid.retrieve(query, k=top_k)
        timings["hybrid_rrf"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        mmr_docs = self.mmr.retrieve(
            query,
            hybrid_docs,
            k=min(mmr_k, len(hybrid_docs)),
            lambda_mult=mmr_lambda,
        )
        timings["mmr"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        reranked_docs = self.reranker.rerank(
            query,
            mmr_docs,
            top_k=min(rerank_top_k, len(mmr_docs)),
        )
        timings["reranker"] = (perf_counter() - start) * 1000.0
        timings["total_retrieval"] = (perf_counter() - total_start) * 1000.0

        return RetrievalResult(
            query=query,
            hybrid=hybrid_docs,
            mmr=mmr_docs,
            reranked=reranked_docs,
            timings_ms=timings,
        )
