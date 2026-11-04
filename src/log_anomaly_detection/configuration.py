"""Typed service settings loaded from the environment (prefix ``LAD_``) or a ``.env`` file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LAD_", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    seed: int = 42
    host: str = "127.0.0.1"
    port: int = 8000
    max_body_bytes: int = 4 * 1024 * 1024


settings = Settings()
