from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM provider (OpenAI-compatible, OpenRouter by default)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_data_collection: str = os.getenv("OPENROUTER_DATA_COLLECTION", "deny")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    llm_timeout_ms: int = int(os.getenv("LLM_TIMEOUT_MS", "60000"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1200"))

    # Local retrieval models. The embedding model explicitly includes fa/Persian support.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

    # Vector store / persistent RAG state
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "advanced_rag")
    rag_state_dir: str = os.getenv("RAG_STATE_DIR", "./rag_state")

    # Chunking and retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "12"))
    mmr_k: int = int(os.getenv("MMR_K", "7"))
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.55"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))
    rrf_constant: int = int(os.getenv("RRF_CONSTANT", "60"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "1.0"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "1.0"))

    # Generation / conversation
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
    enable_query_rewrite: bool = _env_bool("ENABLE_QUERY_REWRITE", True)

    # Basic operational guardrails
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    max_files_per_batch: int = int(os.getenv("MAX_FILES_PER_BATCH", "20"))

    # LangSmith
    langsmith_tracing: bool = _env_bool("LANGSMITH_TRACING", False)

    def __post_init__(self) -> None:
        if self.llm_timeout_ms <= 0 or self.llm_max_tokens <= 0:
            raise ValueError("LLM_TIMEOUT_MS and LLM_MAX_TOKENS must be > 0")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be >= 0")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must satisfy 0 <= overlap < chunk_size")
        if self.top_k <= 0 or self.mmr_k <= 0 or self.rerank_top_k <= 0:
            raise ValueError("TOP_K, MMR_K and RERANK_TOP_K must be > 0")
        if self.reranker_batch_size <= 0:
            raise ValueError("RERANKER_BATCH_SIZE must be > 0")
        if self.max_context_chars <= 0 or self.max_history_messages <= 0:
            raise ValueError("MAX_CONTEXT_CHARS and MAX_HISTORY_MESSAGES must be > 0")
        if self.max_upload_mb <= 0 or self.max_files_per_batch <= 0:
            raise ValueError("Upload limits must be > 0")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("MMR_LAMBDA must be between 0 and 1")
        if self.vector_weight < 0 or self.bm25_weight < 0:
            raise ValueError("VECTOR_WEIGHT and BM25_WEIGHT must be >= 0")
        if self.vector_weight == 0 and self.bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be > 0")
        Path(self.rag_state_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
