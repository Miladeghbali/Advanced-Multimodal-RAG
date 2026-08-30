from __future__ import annotations

from langchain_openrouter import ChatOpenRouter

from config.settings import settings


def create_chat_model() -> ChatOpenRouter:
    provider_options = {}
    data_collection = getattr(settings, "openrouter_data_collection", "deny")
    if data_collection:
        provider_options["data_collection"] = data_collection

    return ChatOpenRouter(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        temperature=0,
        timeout=settings.llm_timeout_ms,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
        openrouter_provider=provider_options or None,
    )
