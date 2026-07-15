#!/usr/bin/env python3
"""Train a mutation-held-out EGFR drug-response model from Nature Figure 2 data.

The script:
1. Reads Panel A from the Nature source-data workbook.
2. Converts triplicate drug measurements to one median response per mutation-drug pair.
3. Compares progressively richer feature sets using GroupKFold by mutation.
4. Tunes only the best feature set.
5. Fits and saves the final CatBoost model on all available data.

Target
------
Median log(mutant IC50 / wild-type IC50).
Positive values indicate relative resistance; negative values indicate relative sensitivity.
This is an experimental relative-response target, not a clinical treatment recommendation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 20260714
N_SPLITS = 5

# Approximate residue properties used only for simple mutation descriptors.
AA_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"hydropathy": 1.8, "volume": 88.6, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "R": {"hydropathy": -4.5, "volume": 173.4, "charge": 1.0, "aromatic": 0.0, "polar": 1.0},
    "N": {"hydropathy": -3.5, "volume": 114.1, "charge": 0.0, "aromatic": 0.0, "polar": 1.0},
    "D": {"hydropathy": -3.5, "volume": 111.1, "charge": -1.0, "aromatic": 0.0, "polar": 1.0},
    "C": {"hydropathy": 2.5, "volume": 108.5, "charge": 0.0, "aromatic": 0.0, "polar": 1.0},
    "Q": {"hydropathy": -3.5, "volume": 143.8, "charge": 0.0, "aromatic": 0.0, "polar": 1.0},
    "E": {"hydropathy": -3.5, "volume": 138.4, "charge": -1.0, "aromatic": 0.0, "polar": 1.0},
    "G": {"hydropathy": -0.4, "volume": 60.1, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "H": {"hydropathy": -3.2, "volume": 153.2, "charge": 0.1, "aromatic": 1.0, "polar": 1.0},
    "I": {"hydropathy": 4.5, "volume": 166.7, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "L": {"hydropathy": 3.8, "volume": 166.7, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "K": {"hydropathy": -3.9, "volume": 168.6, "charge": 1.0, "aromatic": 0.0, "polar": 1.0},
    "M": {"hydropathy": 1.9, "volume": 162.9, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "F": {"hydropathy": 2.8, "volume": 189.9, "charge": 0.0, "aromatic": 1.0, "polar": 0.0},
    "P": {"hydropathy": -1.6, "volume": 112.7, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
    "S": {"hydropathy": -0.8, "volume": 89.0, "charge": 0.0, "aromatic": 0.0, "polar": 1.0},
    "T": {"hydropathy": -0.7, "volume": 116.1, "charge": 0.0, "aromatic": 0.0, "polar": 1.0},
    "W": {"hydropathy": -0.9, "volume": 227.8, "charge": 0.0, "aromatic": 1.0, "polar": 0.0},
    "Y": {"hydropathy": -1.3, "volume": 193.6, "charge": 0.0, "aromatic": 1.0, "polar": 1.0},
    "V": {"hydropathy": 4.2, "volume": 140.0, "charge": 0.0, "aromatic": 0.0, "polar": 0.0},
}

CATEGORICAL_COLUMNS = ["drug", "structure_group", "exon1", "exon2", "exon3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("2021.8.12 Figure 2 Source Data.xlsx"),
        help="Path to the Nature Figure 2 source-data workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("strong_model_outputs"),
        help="Directory for cleaned data, metrics, predictions, and model files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer boosting iterations and skip the second tuning stage.",
    )
    return parser.parse_args()


def normalize_category(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Missing"
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def load_nature_panel_a(workbook: Path) -> pd.DataFrame:
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook.resolve()}")

    raw = pd.read_excel(workbook, sheet_name="Panel A", header=None)
    if raw.shape[1] < 8:
        raise ValueError("Panel A does not have the expected Figure 2 layout.")

    headers = [str(value).strip() for value in raw.iloc[0].tolist()]
    drug_names = list(dict.fromkeys(headers[5:]))
    rows: list[dict[str, Any]] = []

    for row_index in range(1, len(raw)):
        mutation = normalize_category(raw.iloc[row_index, 4])
        structure_group = normalize_category(raw.iloc[row_index, 3])
        exons = [normalize_category(raw.iloc[row_index, i]) for i in range(3)]

        for drug in drug_names:
            drug_columns = [i for i in range(5, len(headers)) if headers[i] == drug]
            replicate_values = pd.to_numeric(
                raw.iloc[row_index, drug_columns], errors="coerce"
            ).dropna()
            if replicate_values.empty:
                continue

            rows.append(
                {
                    "mutation": mutation,
                    "structure_group": structure_group,
                    "exon1": exons[0],
                    "exon2": exons[1],
                    "exon3": exons[2],
                    "drug": drug,
                    "response": float(replicate_values.median()),
                    "replicate_mean": float(replicate_values.mean()),
                    "replicate_sd": float(replicate_values.std(ddof=1))
                    if len(replicate_values) > 1
                    else 0.0,
                    "n_replicates": int(len(replicate_values)),
                }
            )

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No mutation-drug response rows could be read from Panel A.")
    return data


def derive_mutation_features(mutation: str) -> dict[str, float]:
    text = str(mutation)
    positions = [int(x) for x in re.findall(r"(?<!\d)(\d{3})(?!\d)", text)]
    substitutions = re.findall(r"([A-Z])(\d{3})([A-Z])", text)

    result: dict[str, float] = {
        "mutation_component_count": float(
            len([x for x in re.split(r"[/\s]+", text.strip()) if x])
        ),
        "n_detected_positions": float(len(positions)),
        "position_min": float(min(positions)) if positions else np.nan,
        "position_max": float(max(positions)) if positions else np.nan,
        "position_mean": float(np.mean(positions)) if positions else np.nan,
        "n_simple_substitutions": float(len(substitutions)),
        "has_insertion_or_duplication": float(
            "ins" in text.lower() or "dup" in text.lower()
        ),
        "has_deletion": float("del" in text.lower()),
        "has_T790M": float("T790M" in text),
        "has_C797S": float("C797S" in text),
        "has_L858R": float("L858R" in text),
        "has_G719": float("G719" in text),
        "has_Ex19del": float("Ex19del" in text),
    }

    property_deltas: dict[str, list[float]] = {
        name: [] for name in ["hydropathy", "volume", "charge", "aromatic", "polar"]
    }
    for wild_type, _position, mutant in substitutions:
        if wild_type not in AA_PROPERTIES or mutant not in AA_PROPERTIES:
            continue
        for property_name in property_deltas:
            property_deltas[property_name].append(
                AA_PROPERTIES[mutant][property_name]
                - AA_PROPERTIES[wild_type][property_name]
            )

    for property_name, deltas in property_deltas.items():
        result[f"{property_name}_delta_mean"] = (
            float(np.mean(deltas)) if deltas else 0.0
        )
        result[f"{property_name}_delta_abs_sum"] = (
            float(np.abs(deltas).sum()) if deltas else 0.0
        )
    return result


def add_mutation_features(data: pd.DataFrame) -> pd.DataFrame:
    descriptors = pd.DataFrame(
        [derive_mutation_features(name) for name in data["mutation"]],
        index=data.index,
    )
    return pd.concat([data, descriptors], axis=1)


def prepare_features(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = data[feature_columns].copy()
    for column in feature_columns:
        if column in CATEGORICAL_COLUMNS:
            features[column] = features[column].map(normalize_category)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            median = features[column].median()
            features[column] = features[column].fillna(0.0 if pd.isna(median) else median)
    return features


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_pred)) < 2:
        return np.nan
    return float(spearmanr(y_true, y_pred, nan_policy="omit").statistic)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
    }


def make_model(params: dict[str, Any], seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
        max_ctr_complexity=1,
        one_hot_max_size=10,
        **params,
    )


def cross_validate(
    data: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    x = prepare_features(data, feature_columns)
    y = data["response"].to_numpy(dtype=float)
    groups = data["mutation"].to_numpy()
    categorical = [column for column in feature_columns if column in CATEGORICAL_COLUMNS]

    splitter = GroupKFold(n_splits=N_SPLITS)
    out_of_fold = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups=groups), start=1
    ):
        model = make_model(params, RANDOM_SEED + fold)
        model.fit(
            x.iloc[train_index],
            y[train_index],
            cat_features=categorical,
        )
        predictions = model.predict(x.iloc[test_index])
        out_of_fold[test_index] = predictions

        fold_metric = metrics(y[test_index], predictions)
        fold_rows.append(
            {
                "fold": fold,
                "n_train_rows": len(train_index),
                "n_test_rows": len(test_index),
                "n_test_mutations": data.iloc[test_index]["mutation"].nunique(),
                **fold_metric,
            }
        )

    overall = metrics(y, out_of_fold)

    # Per-drug metrics prevent pooled correlation from being dominated by
    # differences in overall response levels between drugs.
    prediction_table = data[
        ["mutation", "drug", "structure_group", "exon1", "response"]
    ].copy()
    prediction_table["prediction"] = out_of_fold
    per_drug_rows: list[dict[str, Any]] = []
    for drug, subset in prediction_table.groupby("drug", sort=False):
        drug_metrics = metrics(
            subset["response"].to_numpy(), subset["prediction"].to_numpy()
        )
        per_drug_rows.append({"drug": drug, "n_rows": len(subset), **drug_metrics})
    per_drug = pd.DataFrame(per_drug_rows)

    overall["mean_per_drug_MAE"] = float(per_drug["MAE"].mean())
    overall["mean_per_drug_RMSE"] = float(per_drug["RMSE"].mean())
    overall["mean_per_drug_R2"] = float(per_drug["R2"].mean())
    overall["mean_per_drug_Spearman"] = float(per_drug["Spearman"].mean())

    return overall, pd.DataFrame(fold_rows), prediction_table


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = add_mutation_features(load_nature_panel_a(args.data))
    data.to_csv(args.output / "nature_egfr_long_format.csv", index=False)

    descriptor_columns = [
        column
        for column in data.columns
        if column
        not in {
            "mutation",
            "structure_group",
            "exon1",
            "exon2",
            "exon3",
            "drug",
            "response",
            "replicate_mean",
            "replicate_sd",
            "n_replicates",
        }
    ]

    feature_sets: dict[str, list[str]] = {
        "drug_only": ["drug"],
        "drug_plus_exon": ["drug", "exon1", "exon2", "exon3"],
        "drug_plus_structure": ["drug", "structure_group"],
        "drug_structure_exon": [
            "drug",
            "structure_group",
            "exon1",
            "exon2",
            "exon3",
        ],
        "mechanism_enhanced": [
            "drug",
            "structure_group",
            "exon1",
            "exon2",
            "exon3",
            *descriptor_columns,
        ],
    }

    if args.quick:
        feature_sets = {
            name: feature_sets[name]
            for name in ["drug_only", "drug_plus_structure", "mechanism_enhanced"]
        }

    base_params = {
        "iterations": 50 if args.quick else 250,
        "depth": 5,
        "learning_rate": 0.06 if args.quick else 0.04,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.5,
    }

    comparison_rows: list[dict[str, Any]] = []
    cv_cache: dict[str, tuple[dict[str, float], pd.DataFrame, pd.DataFrame]] = {}

    print(f"Loaded {data['mutation'].nunique()} mutations, "
          f"{data['drug'].nunique()} drugs, and {len(data)} mutation-drug rows.")
    print("\nComparing feature sets with mutation-held-out cross-validation...")

    for name, columns in feature_sets.items():
        score, fold_table, predictions = cross_validate(data, columns, base_params)
        cv_cache[name] = (score, fold_table, predictions)
        comparison_rows.append(
            {
                "model": name,
                "n_features": len(columns),
                **score,
                "parameters": json.dumps(base_params, sort_keys=True),
            }
        )
        print(
            f"  {name:24s} MAE={score['MAE']:.3f}  "
            f"R2={score['R2']:.3f}  Spearman={score['Spearman']:.3f}  "
            f"mean drug Spearman={score['mean_per_drug_Spearman']:.3f}"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("MAE").reset_index(drop=True)
    best_feature_set_name = str(comparison.iloc[0]["model"])
    best_columns = feature_sets[best_feature_set_name]
    best_params = dict(base_params)
    best_score = float(comparison.iloc[0]["MAE"])

    # A compact second-stage search. All choices are evaluated on held-out mutations.
    if not args.quick:
        candidates = [
            {**base_params, "depth": 4, "l2_leaf_reg": 8.0},
            {**base_params, "depth": 5, "l2_leaf_reg": 15.0},
            {**base_params, "depth": 6, "l2_leaf_reg": 15.0, "random_strength": 1.0},
            {**base_params, "depth": 7, "l2_leaf_reg": 20.0, "random_strength": 1.0},
        ]
        print(f"\nTuning the best feature set: {best_feature_set_name}")
        tuning_rows: list[dict[str, Any]] = []
        for candidate_number, candidate in enumerate(candidates, start=1):
            score, folds, predictions = cross_validate(data, best_columns, candidate)
            tuning_rows.append(
                {
                    "candidate": candidate_number,
                    **score,
                    "parameters": json.dumps(candidate, sort_keys=True),
                }
            )
            print(
                f"  candidate {candidate_number}: MAE={score['MAE']:.3f}, "
                f"R2={score['R2']:.3f}, Spearman={score['Spearman']:.3f}"
            )
            if score["MAE"] < best_score:
                best_score = score["MAE"]
                best_params = candidate
                cv_cache[best_feature_set_name] = (score, folds, predictions)
        pd.DataFrame(tuning_rows).sort_values("MAE").to_csv(
            args.output / "catboost_tuning_results.csv", index=False
        )

    final_cv_score, final_fold_table, final_predictions = cv_cache[best_feature_set_name]
    # Re-evaluate if tuning selected different parameters.
    if json.dumps(best_params, sort_keys=True) != json.dumps(base_params, sort_keys=True):
        final_cv_score, final_fold_table, final_predictions = cross_validate(
            data, best_columns, best_params
        )

    comparison.to_csv(args.output / "feature_set_comparison.csv", index=False)
    final_fold_table.to_csv(args.output / "best_model_fold_metrics.csv", index=False)
    final_predictions.to_csv(args.output / "best_model_oof_predictions.csv", index=False)

    per_drug_rows = []
    for drug, subset in final_predictions.groupby("drug", sort=False):
        per_drug_rows.append(
            {
                "drug": drug,
                "n_rows": len(subset),
                **metrics(
                    subset["response"].to_numpy(),
                    subset["prediction"].to_numpy(),
                ),
            }
        )
    pd.DataFrame(per_drug_rows).sort_values("MAE").to_csv(
        args.output / "best_model_per_drug_metrics.csv", index=False
    )

    x_all = prepare_features(data, best_columns)
    y_all = data["response"].to_numpy(dtype=float)
    categorical = [column for column in best_columns if column in CATEGORICAL_COLUMNS]
    final_model = make_model(best_params, RANDOM_SEED)
    final_model.fit(x_all, y_all, cat_features=categorical)
    final_model.save_model(str(args.output / "best_egfr_catboost_model.cbm"))

    importance = pd.DataFrame(
        {
            "feature": best_columns,
            "importance": final_model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output / "best_model_feature_importance.csv", index=False)

    metadata = {
        "workbook": str(args.data),
        "target": "median log(mutant IC50 / wild-type IC50)",
        "target_interpretation": {
            "positive": "relative resistance versus WT",
            "negative": "relative sensitivity versus WT",
            "zero": "similar IC50 to WT",
        },
        "n_mutations": int(data["mutation"].nunique()),
        "n_drugs": int(data["drug"].nunique()),
        "n_rows": int(len(data)),
        "best_feature_set": best_feature_set_name,
        "feature_columns": best_columns,
        "categorical_columns": categorical,
        "parameters": best_params,
        "cross_validated_metrics": final_cv_score,
        "known_drugs": sorted(data["drug"].unique().tolist()),
        "known_structure_groups": sorted(data["structure_group"].unique().tolist()),
        "random_seed": RANDOM_SEED,
        "validation": "5-fold GroupKFold with mutation as the grouping variable",
        "limitation": (
            "The model predicts responses for drugs represented in the workbook and "
            "requires a structure-function group. It does not infer resistance directly "
            "from a raw PDB structure and is not a clinical treatment recommendation."
        ),
    }
    with open(args.output / "best_model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nBest validated feature set:", best_feature_set_name)
    print("Best parameters:", best_params)
    print("Cross-validated metrics:")
    for name, value in final_cv_score.items():
        print(f"  {name}: {value:.4f}")
    print(f"\nSaved outputs to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
