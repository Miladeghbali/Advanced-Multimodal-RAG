from langchain_core.documents import Document
from rag.context import build_context


def test_context_has_source_labels_and_budget():
    docs = [Document(page_content="hello world", metadata={"source": "a.txt", "page": 1, "chunk_id": "x"})]
    context = build_context(docs, max_chars=200)
    assert "[S1]" in context
    assert "a.txt" in context
    assert len(context) <= 200
