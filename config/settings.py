"""Application configuration (pydantic-settings).

Values come from environment variables / an optional `.env` file. Field names map
to UPPER_SNAKE env vars automatically (e.g. ``llm_backend`` <- ``LLM_BACKEND``).
See ``.env.example`` for the full list. Note that a `.env` file overrides these
code defaults — the startup log line ``app_started`` shows the resolved values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend selection: "ollama" | "mock" (mock = no GPU / no network)
    llm_backend: Literal["ollama", "mock"] = "ollama"
    default_model: str = "Gemma 4 12B"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    # How long Ollama keeps a model resident after a turn when the UI's
    # "Keep model loaded" toggle is off. Per-request keep_alive overrides it.
    ollama_keep_alive: str = "30m"

    # Generation defaults (the "external" sampling/decoding knobs the UI exposes).
    # top_p / top_k defaults mirror Ollama's own defaults.
    default_temperature: float = Field(0.7, ge=0, le=2)
    default_max_tokens: int = Field(1024, gt=0)
    default_top_p: float = Field(0.9, ge=0, le=1)
    default_top_k: int = Field(40, ge=0)
    context_safety_margin: int = Field(768, ge=0)

    # Context management: 0 = raw passthrough, 1 = sliding window,
    # 2 = recursive summarization. Validated so a bad value fails at startup.
    context_level: int = Field(2, ge=0, le=2)

    # Level 2 — recursive summarization (see recursive_summarization/)
    summary_keep_recent_turns: int = Field(4, ge=0)
    summary_target_budget_tokens: int = Field(0, ge=0)   # 0 = derive from the model window
    summary_input_budget_tokens: int = Field(6000, gt=0)
    summary_recompact_cadence: int = Field(20, ge=0)     # full recompact every N folds; 0 = never

    # Persona
    system_prompt: str = (
        "You are a helpful, knowledgeable AI assistant. "
        "Answer clearly and accurately. "
        "Be concise unless depth is needed."
    )

    # Session storage
    session_db_url: str = "sqlite:///./sessions.db"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so the whole app shares one config instance."""
    return Settings()
