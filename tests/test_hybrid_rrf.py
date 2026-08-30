from langchain_core.documents import Document

from retrieval.hybrid import HybridRetriever, document_key


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, k: int = 10):
        return self.docs[:k]


def test_document_key_prefers_chunk_id():
    doc = Document(page_content="hello", metadata={"chunk_id": "abc123"})
    assert document_key(doc) == "abc123"


def test_rrf_rewards_document_found_by_both_channels():
    a = Document(page_content="A", metadata={"chunk_id": "a"})
    b = Document(page_content="B", metadata={"chunk_id": "b"})
    c = Document(page_content="C", metadata={"chunk_id": "c"})

    hybrid = HybridRetriever(
        FakeRetriever([a, b]),
        FakeRetriever([a, c]),
        rrf_constant=60,
        vector_weight=1.0,
        bm25_weight=1.0,
    )
    result = hybrid.retrieve("query", k=3)
    assert result[0].metadata["chunk_id"] == "a"
    assert result[0].metadata["hybrid_rank"] == 1


def test_rrf_preserves_diagnostics_from_both_channels():
    vector_doc = Document(
        page_content="same",
        metadata={"chunk_id": "x", "vector_score": 0.9},
    )
    bm25_doc = Document(
        page_content="same",
        metadata={"chunk_id": "x", "bm25_score": 3.2},
    )
    hybrid = HybridRetriever(
        FakeRetriever([vector_doc]),
        FakeRetriever([bm25_doc]),
        rrf_constant=60,
    )
    result = hybrid.retrieve("query", k=1)[0]
    assert result.metadata["vector_score"] == 0.9
    assert result.metadata["bm25_score"] == 3.2
    assert result.metadata["vector_rank"] == 1
    assert result.metadata["bm25_rank"] == 1
