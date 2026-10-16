"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the local search assistant."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    database_path: str = Field(default="data/conversations.db")
    model_timeout_seconds: float = Field(default=60.0, ge=1.0)
    search_timeout_seconds: float = Field(default=15.0, ge=1.0)
    agent_timeout_seconds: float = Field(default=90.0, ge=0.01)
    ollama_health_timeout_seconds: float = Field(default=3.0, ge=1.0)
    conversation_history_limit: int = Field(default=12, ge=2, le=40)
    max_message_length: int = Field(default=4000, ge=64, le=20000)
    max_search_results: int = Field(default=5, ge=1, le=10)
    max_search_calls_per_run: int = Field(default=3, ge=1, le=8)
    agent_recursion_limit: int = Field(default=12, ge=4, le=40)
    model_call_limit: int = Field(default=6, ge=2, le=20)

    def resolved_database_path(self) -> Path:
        path = Path(self.database_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
