from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_BACKEND: str = "ollama"
    DEFAULT_MODEL: str = "Gemma 4 12B"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1024
    CONTEXT_SAFETY_MARGIN: int = 768

    CONTEXT_LEVEL: int = 0
    SUMMARY_TRIGGER_TOKENS: int = 0
    SLIDING_WINDOW_TURNS: int = 12

    SESSION_DB_URL: str = "sqlite:///./sessions.db"

    ANTHROPIC_API_KEY: str = ""


settings = Settings()
