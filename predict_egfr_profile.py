#!/usr/bin/env python3
"""Predict the relative response profile across all drugs saved with the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_strong_egfr_model import (
    CATEGORICAL_COLUMNS,
    derive_mutation_features,
    normalize_category,
    prepare_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("strong_model_outputs"))
    parser.add_argument("--mutation", required=True, help="Example: G719S")
    parser.add_argument(
        "--structure-group",
        required=True,
        choices=["Classical-Like", "PACC", "T790M-like-3S", "T790M-like-3R", "Ex20ins-L"],
    )
    parser.add_argument("--exon1", default="Missing")
    parser.add_argument("--exon2", default="Missing")
    parser.add_argument("--exon3", default="Missing")
    parser.add_argument("--output", type=Path, default=Path("predicted_drug_profile.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.model_dir / "best_model_metadata.json"
    model_path = args.model_dir / "best_egfr_catboost_model.cbm"
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError("Train the model first; model or metadata file is missing.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    mutation_descriptors = derive_mutation_features(args.mutation)
    rows = []
    for drug in metadata["known_drugs"]:
        row = {
            "mutation": args.mutation,
            "drug": drug,
            "structure_group": args.structure_group,
            "exon1": normalize_category(args.exon1),
            "exon2": normalize_category(args.exon2),
            "exon3": normalize_category(args.exon3),
            **mutation_descriptors,
        }
        rows.append(row)

    prediction_data = pd.DataFrame(rows)
    x = prepare_features(prediction_data, metadata["feature_columns"])
    prediction_data["predicted_log_mutant_wt_ic50"] = model.predict(x)
    prediction_data["relative_interpretation"] = np.where(
        prediction_data["predicted_log_mutant_wt_ic50"] > 0,
        "more resistant than WT",
        "more sensitive than WT",
    )
    prediction_data = prediction_data.sort_values("predicted_log_mutant_wt_ic50")
    prediction_data.to_csv(args.output, index=False)

    print("\nPredicted relative drug-response profile")
    print(
        prediction_data[
            ["drug", "predicted_log_mutant_wt_ic50", "relative_interpretation"]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.output.resolve()}")
    print("These are experimental relative-response predictions, not clinical advice.")


if __name__ == "__main__":
    main()
