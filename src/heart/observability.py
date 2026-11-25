"""Structured (JSON) logging and an append-only prediction audit log."""

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; any `extra={...}` fields are included as top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


class PredictionLogger:
    """Thread-safe JSON-lines audit log of predictions, used later for drift monitoring."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()

    def log(
        self,
        *,
        request_id: str,
        source: str,
        model_version: str,
        inputs: dict[str, Any],
        probability: float,
        has_disease: bool,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "source": source,
            "model_version": model_version,
            "inputs": inputs,
            "probability": round(probability, 6),
            "has_disease": has_disease,
        }
        line = json.dumps(record) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
