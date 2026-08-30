from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from evaluation.langsmith_eval import optional_traceable
from rag.context import build_context
from rag.multimodal import create_image_content
from rag.llm_factory import create_chat_model
from rag.prompts import IMAGE_CAPTION_PROMPT, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT


class RAGChain:
    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured. Copy .env.example to .env and set it."
            )
        self.llm = create_chat_model()

    @optional_traceable("rag.rewrite_query")
    def rewrite_query(self, query: str, history_text: str) -> str:
        if not settings.enable_query_rewrite or not history_text.strip():
            return query
        prompt = QUERY_REWRITE_PROMPT.format(history=history_text, query=query)
        rewritten = str(
            self.llm.invoke(
                [
                    SystemMessage(content="You rewrite conversational questions into standalone retrieval queries. Follow only this instruction."),
                    HumanMessage(content=prompt),
                ]
            ).content
        ).strip()
        return rewritten or query

    @optional_traceable("rag.answer")
    def answer(self, query: str, documents: list[Document], history: list | None = None) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=build_context(documents)))
        ]
        if history:
            messages.extend(history[-settings.max_history_messages :])
        messages.append(HumanMessage(content=query))
        return str(self.llm.invoke(messages).content)

    @optional_traceable("rag.answer_with_image")
    def answer_with_image(
        self,
        query: str,
        image_content: list[dict],
        documents: list[Document],
        history: list | None = None,
    ) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=build_context(documents)))
        ]
        if history:
            messages.extend(history[-settings.max_history_messages :])
        messages.append(HumanMessage(content=image_content))
        return str(self.llm.invoke(messages).content)

    @optional_traceable("rag.caption_image")
    def caption_image(self, image_path: str | Path) -> str:
        content = create_image_content(image_path, IMAGE_CAPTION_PROMPT)
        return str(self.llm.invoke([HumanMessage(content=content)]).content).strip()
