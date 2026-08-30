from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.chunker import split_documents
from core.loaders import load_file
from core.vector_store import VectorStoreManager
from evaluation.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from retrieval.pipeline import AdvancedRetrievalPipeline
from retrieval.reranker import Reranker


def chunk_key(doc) -> str:
    return str(doc.metadata.get("chunk_id", ""))


def score_stage(retrieved_docs, relevant: set[str]) -> dict[str, float]:
    retrieved = [chunk_key(doc) for doc in retrieved_docs]
    k = max(len(retrieved), 1)
    return {
        "precision": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "documents" / "sample_knowledge.txt"
    dataset_path = project_root / "evaluation" / "golden_dataset.json"

    docs = split_documents(load_file(sample_path))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        store = VectorStoreManager(persist_dir=Path(tmp) / "chroma", collection_name="eval")
        store.add_documents(docs)
        pipeline = AdvancedRetrievalPipeline(store, docs, Reranker())

        stage_scores = {"hybrid": [], "mmr": [], "reranked": []}

        for example in dataset:
            keywords = [k.lower() for k in example["expected_keywords"]]
            relevant = {
                chunk_key(doc)
                for doc in docs
                if any(keyword in doc.page_content.lower() for keyword in keywords)
            }
            result = pipeline.retrieve(example["question"])
            stage_scores["hybrid"].append(score_stage(result.hybrid, relevant))
            stage_scores["mmr"].append(score_stage(result.mmr, relevant))
            stage_scores["reranked"].append(score_stage(result.reranked, relevant))

        print("Local retrieval evaluation (does not touch the app knowledge base)")
        for stage, rows in stage_scores.items():
            print(f"\n[{stage}]")
            for metric in ["precision", "recall", "mrr", "ndcg"]:
                print(f"{metric:10s}: {mean(row[metric] for row in rows):.3f}")


if __name__ == "__main__":
    main()
