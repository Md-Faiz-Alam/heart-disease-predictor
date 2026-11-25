# syntax=docker/dockerfile:1
# One image, two services. Pick one at runtime:
#   docker run -e SERVICE=api -p 8000:8000 heart-disease     (default)
#   docker run -e SERVICE=ui  -p 8501:8501 heart-disease
# PORT is honoured if the platform injects it (e.g. Render).

ARG PYTHON_VERSION=3.13

# ---------- builder: locked dependencies in an isolated venv ----------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Locked deps first: this layer is cached until requirements.lock changes.
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# ---------- runtime: no build tools, non-root ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
# libgomp1: OpenMP runtime required by LightGBM / XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 heart
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEART_HOME=/app \
    HEART_LOG_JSON=true \
    MPLCONFIGDIR=/tmp/matplotlib \
    SERVICE=api
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY models ./models
COPY reports ./reports
COPY .streamlit ./.streamlit
COPY app.py .
RUN mkdir -p /app/logs && chown heart:heart /app/logs
USER heart

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "heart.serve", "--healthcheck"]
CMD ["python", "-m", "heart.serve"]
