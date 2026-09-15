"""Environment-backed backend configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://app:app@localhost:5433/app"


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; tests may clear the cache."""

    return Settings()
