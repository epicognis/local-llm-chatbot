"""Adapter from chat-style history to library Turns.

Converts the generic ``{role, content, ...}`` dicts the chatbot stores into
:class:`Turn` objects. Extra keys are kept in the metadata bag so a future
retrieval level can reach the raw fields even after the rendered text has been
folded into a summary.
"""

from __future__ import annotations

from typing import Any

from .types import Role, Turn


def from_chat_dict(entry: dict[str, Any]) -> Turn:
    """Convert a generic chat dict {role, content, ...} to a Turn.

    Extra keys beyond role/content are preserved in metadata.
    """
    role_str = entry.get("role", "user")
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.USER

    content = entry.get("content", "")
    metadata = {k: v for k, v in entry.items() if k not in ("role", "content")}

    return Turn(role=role, text=content, metadata=metadata)


def from_chat_history(history: list[dict[str, Any]]) -> list[Turn]:
    """Convert a full chat-style history list into ordered Turns."""
    return [from_chat_dict(entry) for entry in history]
