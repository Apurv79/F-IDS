
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
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    huggingface_api_token: str = ""
    huggingface_model: str = ""
    ollama_base_url: str = "https://ollama.com"
    ollama_model: str = ""
    ollama_api_key: str = ""

    # AWS S3 & Cloud Storage
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"
    aws_s3_bucket_name: str = ""
    s3_bucket_name: str = ""
    aws_s3_prefix: str = "fids ai/"
    s3_signed_url_expiry: int = 300

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
