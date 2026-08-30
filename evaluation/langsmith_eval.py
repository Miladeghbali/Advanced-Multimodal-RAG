from __future__ import annotations

import os
from typing import Callable


def langsmith_status() -> dict[str, str | bool]:
    enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )
    return {
        "enabled": enabled,
        "project": os.getenv("LANGSMITH_PROJECT", "Advanced-Multimodal-RAG"),
    }


def optional_traceable(name: str) -> Callable:
    """Return LangSmith @traceable when installed, otherwise a no-op decorator."""
    try:
        from langsmith import traceable

        return traceable(name=name)
    except Exception:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator


def run_langsmith_evaluation(target: Callable, dataset_name: str, evaluators: list[Callable]):
    """Run a real LangSmith experiment when credentials/dataset are configured."""
    if not langsmith_status()["enabled"]:
        raise RuntimeError("LangSmith is not enabled. Configure LANGSMITH_API_KEY and LANGSMITH_TRACING=true.")
    from langsmith import Client

    client = Client()
    return client.evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=os.getenv("LANGSMITH_PROJECT", "Advanced-Multimodal-RAG"),
    )
