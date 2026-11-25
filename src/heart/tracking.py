"""Optional MLflow experiment tracking (local SQLite store under ./mlruns)."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

from heart.config import MLFLOW_DIR, MLFLOW_EXPERIMENT

log = logging.getLogger(__name__)


@contextmanager
def mlflow_run(enabled: bool, run_name: str) -> Iterator[ModuleType | None]:
    """Yield the mlflow module inside an active run, or None if tracking is off/unavailable."""
    if not enabled:
        yield None
        return
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping experiment tracking (pip install -e .[train])")
        yield None
        return

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        yield mlflow
