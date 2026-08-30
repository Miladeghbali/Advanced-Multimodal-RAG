from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage

from config.settings import settings
from core.corpus_store import CorpusStore
from core.vector_store import VectorStoreManager
from evaluation.langsmith_eval import langsmith_status
from rag.chain import RAGChain
from rag.llm_factory import create_chat_model
from retrieval.pipeline import AdvancedRetrievalPipeline
from retrieval.reranker import Reranker


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def citation_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    return {"key": "has_citation", "score": "[S" in outputs.get("answer", "")}


def answer_keyword_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    keywords = (reference_outputs or {}).get("expected_keywords", [])
    return {
        "key": "answer_keyword_match",
        "score": _contains_any(outputs.get("answer", ""), keywords) if keywords else True,
    }


def retrieval_keyword_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    keywords = (reference_outputs or {}).get("expected_keywords", [])
    docs = "\n".join(outputs.get("documents", []))
    return {
        "key": "retrieval_keyword_match",
        "score": _contains_any(docs, keywords) if keywords else True,
    }


def _judge_llm():
    return create_chat_model()


def _pass_fail(llm, prompt: str) -> bool:
    text = str(llm.invoke([HumanMessage(content=prompt)]).content).strip().upper()
    return text.startswith("PASS")


def make_llm_judges() -> list:
    """Reference-based/reference-free RAG judges aligned with LangSmith RAG evaluation patterns."""
    llm = _judge_llm()

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        reference = (reference_outputs or {}).get("expected_answer", "")
        prompt = f"""Grade whether the STUDENT ANSWER is factually consistent with the REFERENCE ANSWER.
Question: {inputs.get('question','')}
Reference answer: {reference}
Student answer: {outputs.get('answer','')}
Return only PASS or FAIL."""
        return {"key": "correctness", "score": _pass_fail(llm, prompt)}

    def answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        prompt = f"""Grade whether the ANSWER directly and helpfully addresses the QUESTION.
Question: {inputs.get('question','')}
Answer: {outputs.get('answer','')}
Return only PASS or FAIL."""
        return {"key": "answer_relevance", "score": _pass_fail(llm, prompt)}

    def groundedness(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        facts = "\n\n".join(outputs.get("documents", []))
        prompt = f"""Grade whether every factual claim in the ANSWER is supported by the FACTS. Fail if the answer hallucinates material facts.
Facts:\n{facts}
Answer:\n{outputs.get('answer','')}
Return only PASS or FAIL."""
        return {"key": "groundedness", "score": _pass_fail(llm, prompt)}

    def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        facts = "\n\n".join(outputs.get("documents", []))
        prompt = f"""Grade whether the RETRIEVED FACTS contain information relevant to answering the QUESTION.
Question: {inputs.get('question','')}
Retrieved facts:\n{facts}
Return only PASS or FAIL."""
        return {"key": "retrieval_relevance", "score": _pass_fail(llm, prompt)}

    return [correctness, answer_relevance, groundedness, retrieval_relevance]


def main() -> None:
    if not langsmith_status()["enabled"]:
        raise SystemExit(
            "Enable LangSmith first: LANGSMITH_TRACING=true and LANGSMITH_API_KEY=..."
        )

    from langsmith import Client

    root = Path(__file__).resolve().parents[1]
    examples_raw = json.loads(
        (root / "evaluation" / "golden_dataset.json").read_text(encoding="utf-8")
    )
    dataset_name = "Advanced-Multimodal-RAG-Golden"
    client = Client()

    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": item["question"]},
                    "outputs": {
                        "expected_answer": item["expected_answer"],
                        "expected_keywords": item["expected_keywords"],
                    },
                }
                for item in examples_raw
            ],
        )

    corpus = CorpusStore().load()
    if not corpus:
        raise SystemExit("No persisted corpus. Index the sample or your documents in the app first.")

    vector_store = VectorStoreManager()
    pipeline = AdvancedRetrievalPipeline(vector_store, corpus, Reranker())
    chain = RAGChain()

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        result = pipeline.retrieve(question)
        answer = chain.answer(question, result.reranked)
        return {
            "answer": answer,
            "documents": [doc.page_content for doc in result.reranked],
        }

    evaluators = [citation_evaluator, answer_keyword_evaluator, retrieval_keyword_evaluator]
    if os.getenv("LANGSMITH_LLM_JUDGE", "false").lower() == "true":
        evaluators.extend(make_llm_judges())

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="advanced-rag",
        metadata={"pipeline": "Dense+BM25->RRF->MMR->CrossEncoder->LLM"},
    )
    print(results)


if __name__ == "__main__":
    main()
