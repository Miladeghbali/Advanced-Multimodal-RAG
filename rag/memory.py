from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings


class ConversationMemory:
    def __init__(self, max_messages: int | None = None) -> None:
        self.max_messages = max_messages or settings.max_history_messages
        self.messages: list = []

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get(self) -> list:
        return list(self.messages)

    def as_text(self) -> str:
        lines: list[str] = []
        for message in self.messages:
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.messages.clear()
