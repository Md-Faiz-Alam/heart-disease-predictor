"""REST API for the heart disease model.

Run:   uvicorn heart.api:app --reload
Docs:  http://localhost:8000/docs
"""

import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, Security
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader

from heart import __version__
from heart.observability import PredictionLogger, configure_logging
from heart.predict import HeartModel
from heart.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    Contribution,
    HealthResponse,
    ModelInfo,
    PatientInput,
    PredictionResponse,
)
from heart.settings import Settings, get_settings

log = logging.getLogger("heart.api")


def create_app(
    settings: Settings | None = None,
    model: HeartModel | None = None,
    metadata: dict | None = None,
) -> FastAPI:
    """App factory. Tests inject a model; production loads it from settings at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        app.state.model = model or HeartModel.load(settings.model_path)
        app.state.metadata = metadata
        if app.state.metadata is None and settings.metadata_path.exists():
            app.state.metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        app.state.prediction_log = PredictionLogger(
            settings.prediction_log_path, settings.log_predictions
        )
        log.info(
            "model loaded",
            extra={
                "model_name": app.state.model.model_name,
                "model_version": app.state.model.version,
            },
        )
        yield

    app = FastAPI(
        title="Heart Disease Risk API",
        version=__version__,
        description=(
            "Predicts the probability of coronary artery disease from 13 clinical measurements "
            "and explains each prediction with SHAP.\n\n"
            "**For education and research only. Not a medical device.**"
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        """Enforced only when HEART_API_KEY is set; open otherwise (local dev, demos)."""
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (key and secrets.compare_digest(key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

    def get_model(request: Request) -> HeartModel:
        model_: HeartModel | None = getattr(request.app.state, "model", None)
        if model_ is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return model_

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        model_ = getattr(request.app.state, "model", None)
        return HealthResponse(
            status="ok" if model_ else "degraded",
            model_loaded=model_ is not None,
            model_version=model_.version if model_ else None,
        )

    @app.get("/model-info", response_model=ModelInfo, tags=["model"])
    def model_info(request: Request, model_: HeartModel = Depends(get_model)) -> ModelInfo:
        meta = request.app.state.metadata or {}
        metrics = {}
        if meta:
            cv = meta["cv"]["results"][meta["model_name"]]
            metrics = {
                "cv_roc_auc_mean": cv["roc_auc"]["mean"],
                "cv_roc_auc_std": cv["roc_auc"]["std"],
                "test": meta.get("test_metrics", {}),
                "test_ci_95": meta.get("test_ci_95", {}),
            }
        return ModelInfo(
            model_name=model_.model_name,
            model_version=model_.version,
            threshold=model_.threshold,
            feature_set=model_.feature_set,
            features=model_.model_features,
            calibration=model_.calibration,
            metrics=metrics,
            input_schema=PatientInput.model_json_schema(),
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["predict"],
        dependencies=[Depends(require_api_key)],
    )
    def predict(
        patient: PatientInput,
        request: Request,
        explain: bool = Query(True, description="Include SHAP explanation"),
        top_k: int = Query(settings.top_factors, ge=1, le=13, description="Factors to return"),
        model_: HeartModel = Depends(get_model),
    ) -> PredictionResponse:
        result = model_.predict(patient, explain=explain)
        request_id = request.state.request_id
        request.app.state.prediction_log.log(
            request_id=request_id,
            source="api",
            model_version=model_.version,
            inputs=patient.model_dump(),
            probability=result.probability,
            has_disease=result.has_disease,
        )
        return PredictionResponse(
            request_id=request_id,
            probability=result.probability,
            has_disease=result.has_disease,
            risk_band=result.risk_band,
            threshold=result.threshold,
            base_probability=result.base_probability if explain else None,
            top_factors=[Contribution(**vars(c)) for c in result.contributions[:top_k]],
            model_name=result.model_name,
            model_version=result.model_version,
        )

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        tags=["predict"],
        dependencies=[Depends(require_api_key)],
    )
    def predict_batch(
        batch: BatchRequest, request: Request, model_: HeartModel = Depends(get_model)
    ) -> BatchResponse:
        if len(batch.patients) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: {len(batch.patients)} > {settings.max_batch_size}",
            )
        X = pd.concat([p.to_frame() for p in batch.patients], ignore_index=True)
        scored = model_.predict_batch(X)
        request_id = request.state.request_id
        for patient, row in zip(batch.patients, scored.itertuples(), strict=True):
            request.app.state.prediction_log.log(
                request_id=request_id,
                source="api-batch",
                model_version=model_.version,
                inputs=patient.model_dump(),
                probability=float(row.probability),
                has_disease=bool(row.has_disease),
            )
        return BatchResponse(
            request_id=request_id,
            model_version=model_.version,
            threshold=model_.threshold,
            n=len(scored),
            n_positive=int(scored["has_disease"].sum()),
            results=[
                BatchItem(
                    probability=float(r.probability),
                    has_disease=bool(r.has_disease),
                    risk_band=r.risk_band,
                )
                for r in scored.itertuples()
            ],
        )

    return app


app = create_app()
