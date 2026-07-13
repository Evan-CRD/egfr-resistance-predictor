from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "egfr_rich_regressor.joblib"
SCHEMA_PATH = PROJECT_ROOT / "models" / "feature_schema.json"


def load_assets() -> tuple[Any, dict]:
    """Load the fitted model and feature schema."""
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    return model, schema


def validate_input(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Validate and order input columns for prediction."""
    required = schema["numeric_features"] + schema["categorical_features"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Input is missing required columns: " + ", ".join(missing)
        )
    return frame[required].copy()


def predict_auc_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    """Predict AUC ratio versus WT and assign an exploratory class."""
    model, schema = load_assets()
    ordered = validate_input(frame, schema)
    predictions = model.predict(ordered)

    output = frame.copy()
    output["predicted_auc_ratio_vs_wt"] = predictions
    threshold = float(schema["resistance_threshold"])
    output["predicted_resistant"] = (
        output["predicted_auc_ratio_vs_wt"] >= threshold
    ).astype(int)
    return output
