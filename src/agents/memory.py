"""Per-session conversation memory for the agentic pipeline."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    max_turns: int = 10
    _turns: deque = field(default_factory=lambda: deque(maxlen=10))

    def __post_init__(self):
        self._turns = deque(maxlen=self.max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append({"question": question, "answer": answer})

    def get_history_text(self) -> str:
        if not self._turns:
            return ""
        parts = []
        for i, turn in enumerate(self._turns, 1):
            parts.append(f"[Turn {i}]\nUser: {turn['question']}\nAssistant: {turn['answer']}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


_memory_store: dict[str, ConversationMemory] = {}


def get_memory(session_id: str, max_turns: int = 10) -> ConversationMemory:
    """Return the ConversationMemory for this session, creating it if needed."""
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationMemory(max_turns=max_turns)
    return _memory_store[session_id]


def clear_memory(session_id: str) -> None:
    """Remove all history for a session."""
    _memory_store.pop(session_id, None)
