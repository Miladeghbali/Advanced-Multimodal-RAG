from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Persian/Arabic glyph normalization for more reliable lexical matching.
    return (
        text.replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("ة", "ه")
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def _clone(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata))


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        # Keep the persisted corpus immutable during retrieval. Query-specific
        # ranks/scores are written only to cloned result Documents.
        self.documents = documents
        corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        if not self.documents or self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(self.documents)), key=lambda i: float(scores[i]), reverse=True
        )[: min(k, len(self.documents))]
        output: list[Document] = []
        for rank, index in enumerate(ranked_indices, start=1):
            doc = _clone(self.documents[index])
            doc.metadata["bm25_rank"] = rank
            doc.metadata["bm25_score"] = float(scores[index])
            output.append(doc)
        return output
