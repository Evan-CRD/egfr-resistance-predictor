from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "strong_model_outputs"


def load_assets():
    metadata = json.loads((MODEL_DIR / "best_model_metadata.json").read_text())
    model = CatBoostRegressor()
    model.load_model(str(MODEL_DIR / "best_egfr_catboost_model.cbm"))
    data = pd.read_csv(MODEL_DIR / "nature_egfr_long_format.csv")
    return model, metadata, data


def predict_profile(structure_group: str) -> pd.DataFrame:
    model, metadata, _ = load_assets()
    frame = pd.DataFrame({"drug": metadata["known_drugs"], "structure_group": structure_group})
    frame["predicted_log_mutant_wt_ic50"] = model.predict(frame[metadata["feature_columns"]])
    frame["relative_fold_change"] = 2 ** frame["predicted_log_mutant_wt_ic50"]
    frame["interpretation"] = frame["predicted_log_mutant_wt_ic50"].map(
        lambda x: "More resistant than WT" if x > 0 else "More sensitive than WT"
    )
    return frame.sort_values("predicted_log_mutant_wt_ic50", ascending=False)
