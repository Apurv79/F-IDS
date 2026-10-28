
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the FL-IDS ML service."""

    # Application
    app_name: str = "FL-IDS ML Service"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    # API security
    internal_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # AI providers
    huggingface_api_token: str = ""
    huggingface_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    # API key for hosted Ollama inference.
    # Store the actual secret in .env, not in source code.
    ollama_api_key: str = ""

    # AWS S3 — optional during local development
    aws_region: str = "ap-south-1"
    s3_bucket_name: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
