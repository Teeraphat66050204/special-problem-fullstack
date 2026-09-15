"""Environment-backed backend configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://app:app@localhost:5433/app"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field(default="qwen2.5:7b-instruct", min_length=1)
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_temperature: float = Field(default=0.2, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; tests may clear the cache."""

    return Settings()
