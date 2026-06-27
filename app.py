from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train_model import METADATA_PATH, MODEL_PATH, train_model


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
ARTIFACT_DIR = BASE_DIR / "artifacts"

app = FastAPI(title="Trader Sentiment Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_CACHE: dict[str, Any] = {"artifact": None}


class PredictRequest(BaseModel):
    total_trades: int = Field(..., ge=0)
    avg_pnl: float
    total_pnl: float
    win_rate: float = Field(..., ge=0, le=1)
    avg_trade_size: float = Field(..., ge=0)
    total_trade_size: float = Field(..., ge=0)
    buy_ratio: float = Field(..., ge=0, le=1)
    sell_ratio: float = Field(..., ge=0, le=1)
    pnl_per_trade: float


class ReloadResponse(BaseModel):
    message: str
    metadata: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI template not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    artifact = load_model_artifact()
    metadata = load_metadata()
    return {
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "loaded": artifact is not None,
        "classes": metadata.get("classes", []),
        "features": metadata.get("features", []),
        "accuracy": metadata.get("accuracy"),
    }


@app.post("/api/reload", response_model=ReloadResponse)
def reload_model() -> ReloadResponse:
    metadata = train_model()
    MODEL_CACHE["artifact"] = None
    load_model_artifact(force_reload=True)
    return ReloadResponse(message="Model retrained and reloaded", metadata=metadata)


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    encoder = artifact["label_encoder"]
    feature_columns = artifact["feature_columns"]

    features = pd.DataFrame([payload.model_dump()], columns=feature_columns)
    probabilities = pipeline.predict_proba(features)[0]
    prediction_index = int(probabilities.argmax())
    prediction = encoder.inverse_transform([prediction_index])[0]

    probability_map = {
        str(label): float(score)
        for label, score in zip(encoder.classes_, probabilities, strict=False)
    }
    return PredictResponse(
        prediction=str(prediction),
        confidence=float(probabilities[prediction_index]),
        probabilities=probability_map,
    )


@app.get("/api/sample-input")
def sample_input() -> dict[str, Any]:
    metadata = load_metadata()
    return {
        "features": metadata.get("features", []),
        "example": {
            "total_trades": 20,
            "avg_pnl": 120.5,
            "total_pnl": 2410.0,
            "win_rate": 0.55,
            "avg_trade_size": 725.0,
            "total_trade_size": 14500.0,
            "buy_ratio": 0.6,
            "sell_ratio": 0.4,
            "pnl_per_trade": 120.5,
        },
    }


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def load_model_artifact(force_reload: bool = False) -> dict[str, Any]:
    cached = MODEL_CACHE.get("artifact")
    if cached is not None and not force_reload:
        return cached

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Run train_model.py first.")

    artifact = joblib.load(MODEL_PATH)
    MODEL_CACHE["artifact"] = artifact
    return artifact


@app.on_event("startup")
def startup_event() -> None:
    if MODEL_PATH.exists():
        load_model_artifact(force_reload=True)
