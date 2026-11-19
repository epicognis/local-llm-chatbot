from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_BACKEND: Literal["ollama", "mock", "anthropic"] = "ollama"
    DEFAULT_MODEL: str = "Gemma 4 12B"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_KEEP_ALIVE: str = "30m"

    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = Field(1024, gt=0)
    CONTEXT_SAFETY_MARGIN: int = Field(768, ge=0)

    # Validated here so a bad value fails at startup, not on the first chat turn.
    CONTEXT_LEVEL: int = Field(0, ge=0, le=2)
    SUMMARY_TRIGGER_TOKENS: int = 0
    SLIDING_WINDOW_TURNS: int = Field(12, gt=0)

    SESSION_DB_URL: str = "sqlite:///./sessions.db"

    ANTHROPIC_API_KEY: str = Field("", repr=False)  # never log credentials


settings = Settings()
