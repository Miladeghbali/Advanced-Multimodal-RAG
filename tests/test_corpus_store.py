from langchain_core.documents import Document
from core.corpus_store import CorpusStore


def test_corpus_roundtrip(tmp_path):
    store = CorpusStore(tmp_path)
    docs = [Document(page_content="سلام", metadata={"chunk_id": "1", "source": "x.txt"})]
    store.save(docs)
    loaded = store.load()
    assert loaded[0].page_content == "سلام"
    assert loaded[0].metadata["chunk_id"] == "1"
