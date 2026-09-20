"""Environment-backed backend configuration."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://app:app@localhost:5433/app"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field(default="qwen2.5:7b-instruct", min_length=1)
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_temperature: float = Field(default=0.2, ge=0)
    ocr_provider: str = "disabled"
    typhoon_api_key: SecretStr | None = None
    typhoon_base_url: str = "https://api.opentyphoon.ai/v1"
    typhoon_ocr_model: str = "typhoon-ocr"
    ocr_timeout_seconds: float = Field(default=120.0, gt=0)
    ocr_max_pages: int = Field(
        default=6,
        ge=1,
        le=6,
        validation_alias=AliasChoices("OCR_MAX_PAGES", "OCR_FRONT_MATTER_PAGE_LIMIT"),
    )

    # Retained for compatibility with external provider factories configured before
    # the built-in Typhoon adapter was introduced.
    ocr_endpoint: str | None = None
    ocr_api_key: SecretStr | None = None

    @property
    def ocr_front_matter_page_limit(self) -> int:
        """Compatibility name for callers using the original OCR abstraction."""

        return self.ocr_max_pages


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; tests may clear the cache."""

    return Settings()
