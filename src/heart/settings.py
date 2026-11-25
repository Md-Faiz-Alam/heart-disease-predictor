"""Runtime settings, overridable with HEART_* environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from heart.config import METADATA_PATH, MODEL_PATH, ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HEART_", env_file=".env", extra="ignore")

    model_path: Path = MODEL_PATH
    metadata_path: Path = METADATA_PATH

    log_level: str = "INFO"
    log_json: bool = True

    # Prediction audit log (JSON lines). Contains clinical inputs and outputs only;
    # never names, IDs or other identifiers.
    log_predictions: bool = True
    prediction_log_path: Path = ROOT / "logs" / "predictions.jsonl"

    # When set, /predict and /predict/batch require this value in an X-API-Key header.
    api_key: SecretStr | None = None

    max_batch_size: int = 1000
    top_factors: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
