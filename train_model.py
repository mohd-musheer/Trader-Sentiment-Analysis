from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


@dataclass(frozen=True)
class DataPaths:
    sentiment: Path
    trades: Path


class DataFileError(FileNotFoundError):
    pass


def resolve_data_paths() -> DataPaths:
    sentiment_candidates = [
        BASE_DIR / "sentiment.csv",
        BASE_DIR / "Sentiment.csv",
        BASE_DIR / "fear_greed_index.csv",
    ]
    trade_candidates = [
        BASE_DIR / "trader_data.csv",
        BASE_DIR / "historical_data.csv",
    ]

    sentiment_path = next((path for path in sentiment_candidates if path.exists()), None)
    trade_path = next((path for path in trade_candidates if path.exists()), None)

    missing = []
    if sentiment_path is None:
        missing.append("sentiment.csv / Sentiment.csv / fear_greed_index.csv")
    if trade_path is None:
        missing.append("trader_data.csv / historical_data.csv")
    if missing:
        raise DataFileError(f"Missing required file(s): {', '.join(missing)}")

    return DataPaths(sentiment=sentiment_path, trades=trade_path)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_data_paths()
    sentiment = pd.read_csv(paths.sentiment)
    trades = pd.read_csv(paths.trades)
    sentiment.columns = sentiment.columns.str.strip()
    trades.columns = trades.columns.str.strip()
    return sentiment, trades


def prepare_daily_training_frame() -> pd.DataFrame:
    sentiment, trades = load_datasets()

    sentiment = sentiment.copy()
    trades = trades.copy()

    if "date" not in sentiment.columns and "Date" in sentiment.columns:
        sentiment["date"] = sentiment["Date"]
    if "Timestamp IST" in trades.columns:
        trades["trade_datetime"] = trades["Timestamp IST"]
    elif "Timestamp" in trades.columns:
        trades["trade_datetime"] = trades["Timestamp"]
    else:
        raise DataFileError("Missing required trade timestamp column")

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date
    trades["trade_datetime"] = pd.to_datetime(trades["trade_datetime"], errors="coerce", dayfirst=True)
    trades["date"] = trades["trade_datetime"].dt.date

    trades = trades.dropna(subset=["date"])
    sentiment = sentiment.dropna(subset=["date"])
    trades = trades.drop_duplicates()

    merged = trades.merge(sentiment[["date", "classification"]], on="date", how="inner")
    merged = merged.dropna(subset=["classification"])

    merged["is_profit"] = pd.to_numeric(merged["Closed PnL"], errors="coerce").fillna(0) > 0
    merged["side_is_buy"] = merged["Side"].astype(str).str.upper().eq("BUY").astype(int)

    daily = merged.groupby(["date", "classification"], as_index=False).agg(
        total_trades=("classification", "size"),
        avg_pnl=("Closed PnL", "mean"),
        total_pnl=("Closed PnL", "sum"),
        win_rate=("is_profit", "mean"),
        avg_trade_size=("Size USD", "mean"),
        total_trade_size=("Size USD", "sum"),
        buy_trades=("side_is_buy", "sum"),
    )

    daily["buy_ratio"] = daily["buy_trades"] / daily["total_trades"]
    daily["sell_ratio"] = 1 - daily["buy_ratio"]
    daily["pnl_per_trade"] = daily["total_pnl"] / daily["total_trades"]

    return daily.drop(columns=["buy_trades"])


def build_training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    feature_columns = [
        "total_trades",
        "avg_pnl",
        "total_pnl",
        "win_rate",
        "avg_trade_size",
        "total_trade_size",
        "buy_ratio",
        "sell_ratio",
        "pnl_per_trade",
    ]
    X = frame[feature_columns].fillna(0)
    y = frame["classification"].astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, pd.Series(y_encoded, index=frame.index), encoder


def train_model() -> dict[str, object]:
    frame = prepare_daily_training_frame()
    if frame.empty:
        raise RuntimeError("No training rows were created from the CSV files.")

    X, y, encoder = build_training_matrix(frame)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0)

    artifact = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifact, MODEL_PATH)

    metadata = {
        "rows": int(frame.shape[0]),
        "features": X.columns.tolist(),
        "classes": encoder.classes_.tolist(),
        "accuracy": float(accuracy),
        "report": report,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))
