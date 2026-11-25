"""Container entrypoint: start the API or the UI, chosen by environment variables.

    SERVICE=api   (default)  ->  uvicorn heart.api:app
    SERVICE=ui               ->  streamlit run app.py

PORT is honoured when the platform injects it (Render, Railway, Cloud Run, ...).

Usage:
    python -m heart.serve                 # start the selected service
    python -m heart.serve --healthcheck   # exit 0 if the running service is healthy
"""

import os
import sys
import urllib.request
from collections.abc import Mapping

from heart.config import ROOT

DEFAULT_PORTS = {"api": "8000", "ui": "8501"}
HEALTH_PATHS = {"api": "/health", "ui": "/_stcore/health"}


def _service(env: Mapping[str, str]) -> str:
    service = env.get("SERVICE", "api").strip().lower()
    if service not in DEFAULT_PORTS:
        raise SystemExit(f"Unknown SERVICE={service!r}; expected 'api' or 'ui'")
    return service


def _port(service: str, env: Mapping[str, str]) -> str:
    return env.get("PORT") or DEFAULT_PORTS[service]


def build_command(env: Mapping[str, str]) -> list[str]:
    service = _service(env)
    port = _port(service, env)
    if service == "ui":
        return [
            "streamlit", "run", str(ROOT / "app.py"),
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]  # fmt: skip
    return [
        "uvicorn", "heart.api:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", env.get("WEB_CONCURRENCY", "1"),
        "--no-access-log",  # requests are already logged as JSON by the API middleware
        "--proxy-headers",
        "--forwarded-allow-ips", "*",
    ]  # fmt: skip


def healthcheck(env: Mapping[str, str]) -> int:
    service = _service(env)
    url = f"http://127.0.0.1:{_port(service, env)}{HEALTH_PATHS[service]}"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return 0 if response.status == 200 else 1
    except OSError:
        return 1


def main() -> None:
    env = dict(os.environ)
    if "--healthcheck" in sys.argv[1:]:
        sys.exit(healthcheck(env))
    cmd = build_command(env)
    os.execvp(cmd[0], cmd)  # replace this process so signals reach the server directly


if __name__ == "__main__":
    main()
