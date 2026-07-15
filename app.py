from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "strong_model_outputs"
GITHUB_URL = "https://github.com/Evan-CRD/egfr-resistance-predictor"

st.set_page_config(
    page_title="EGFR TKI Response Model",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
<style>
:root{--cyan:#42d8e8;--teal:#34d5a7;--line:#25313d}
.stApp{background:radial-gradient(circle at 12% 0%,rgba(66,216,232,.08),transparent 28%),linear-gradient(180deg,#030405,#070a0e)}
.block-container{max-width:1260px;padding-top:1.4rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#06090d;border-right:1px solid var(--line)}
.hero{padding:2.2rem 2.4rem;border-radius:24px;border:1px solid rgba(66,216,232,.28);background:linear-gradient(135deg,#101923,#071015);box-shadow:0 22px 60px rgba(0,0,0,.45);margin-bottom:1.3rem}
.hero .k{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.hero h1{color:white;font-size:2.65rem;margin:.55rem 0}
.hero p{color:#bdc8d2;font-size:1.08rem;line-height:1.6;max-width:900px}
.result{border-left:4px solid var(--teal);padding:1rem 1.1rem;background:rgba(52,213,167,.08);border-radius:0 14px 14px 0;margin:.7rem 0}
.note{border:1px solid rgba(66,216,232,.24);padding:1rem 1.1rem;background:rgba(66,216,232,.05);border-radius:14px;margin:.7rem 0}
div[data-testid="stMetric"]{border:1px solid rgba(105,125,145,.28);border-radius:16px;padding:.9rem 1rem;background:linear-gradient(180deg,#101720,#080c11)}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,var(--cyan),var(--teal));color:#021012;border:none;border-radius:12px;font-weight:800}
h1,h2,h3,h4{color:white}
p,li,label{color:#d7dfe7}
a{color:var(--cyan)!important}
.footer{text-align:center;color:#778490;font-size:.82rem;padding-top:2rem}
</style>
""",
    unsafe_allow_html=True,
)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        st.error(
            f"{label} is missing: `{path.name}`. "
            "Upload the complete project folder, including `strong_model_outputs`."
        )
        st.stop()


@st.cache_resource
def load_model() -> CatBoostRegressor:
    path = OUT / "best_egfr_catboost_model.cbm"
    require_file(path, "Trained CatBoost model")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


@st.cache_data
def load_all():
    files = {
        "metadata": OUT / "best_model_metadata.json",
        "data": OUT / "nature_egfr_long_format.csv",
        "comparison": OUT / "feature_set_comparison.csv",
        "oof": OUT / "best_model_oof_predictions.csv",
        "per_drug": OUT / "best_model_per_drug_metrics.csv",
        "importance": OUT / "best_model_feature_importance.csv",
    }
    for label, path in files.items():
        require_file(path, label)

    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    data = pd.read_csv(files["data"])
    comparison = pd.read_csv(files["comparison"])
    oof = pd.read_csv(files["oof"])
    per_drug = pd.read_csv(files["per_drug"])
    importance = pd.read_csv(files["importance"])
    return metadata, data, comparison, oof, per_drug, importance


model = load_model()
metadata, data, comparison, oof, per_drug, importance = load_all()

st.markdown(
    """
<div class="hero">
<div class="k">Machine learning · EGFR · TKI response</div>
<h1>EGFR Structure–Function Drug Response Model</h1>
<p>Predict relative inhibitor response from drug identity and published EGFR structure–function groups, and test whether a mechanistic structural representation outperforms the conventional exon-based representation.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🧬 Navigation")
page = st.sidebar.radio(
    "",
    [
        "Prediction",
        "Model Comparison",
        "Validation Results",
        "How It Works",
        "Pipeline",
        "Limitations",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"[View source on GitHub]({GITHUB_URL})")
st.sidebar.caption("Version 5.0 · Rebuilt deployment")


if page == "Prediction":
    st.header("Predict a mutation's relative drug-response profile")

    mapping = (
        data[["mutation", "structure_group", "exon1", "exon2", "exon3"]]
        .drop_duplicates()
        .sort_values("mutation")
    )

    group_counts = mapping.groupby("mutation")["structure_group"].nunique()
    ambiguous = group_counts[group_counts > 1].index.tolist()
    if ambiguous:
        st.error(
            "The source data assign multiple structural groups to: "
            + ", ".join(ambiguous)
        )
        st.stop()

    mutation = st.selectbox("EGFR mutation", mapping["mutation"].unique().tolist())
    selected = mapping.loc[mapping["mutation"] == mutation].iloc[0]
    group = str(selected["structure_group"])

    c1, c2 = st.columns(2)
    c1.text_input(
        "Published structure–function group",
        value=group,
        disabled=True,
    )

    exons = [
        str(value).replace(".0", "")
        for value in [selected["exon1"], selected["exon2"], selected["exon3"]]
        if pd.notna(value) and str(value) not in {"Missing", "nan"}
    ]
    c2.text_input(
        "Published exon assignment",
        value=", ".join(exons) if exons else "Not specified",
        disabled=True,
    )

    st.markdown(
        f"""
<div class="note">
<strong>{mutation}</strong> is automatically linked to the published
<strong>{group}</strong> structure–function class. The classification cannot
be changed independently, preventing unsupported mutation–group combinations.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Predict response across all 18 TKIs",
        type="primary",
        use_container_width=True,
    ):
        prediction_frame = pd.DataFrame(
            {
                "drug": metadata["known_drugs"],
                "structure_group": group,
            }
        )

        expected = metadata["feature_columns"]
        missing = [name for name in expected if name not in prediction_frame.columns]
        if missing:
            st.error(
                "The saved model expects unsupported page inputs: "
                + ", ".join(missing)
                + ". Retrain or redeploy the matching model outputs."
            )
            st.stop()

        prediction_frame["predicted_log2_ratio"] = model.predict(
            prediction_frame[expected]
        )
        prediction_frame["predicted_IC50_fold_vs_WT"] = np.power(
            2.0, prediction_frame["predicted_log2_ratio"]
        )
        prediction_frame["interpretation"] = np.select(
            [
                prediction_frame["predicted_log2_ratio"] > 0.05,
                prediction_frame["predicted_log2_ratio"] < -0.05,
            ],
            [
                "More resistant than WT",
                "More sensitive than WT",
            ],
            default="Approximately WT-like",
        )
        prediction_frame = prediction_frame.sort_values(
            "predicted_log2_ratio",
            ascending=False,
        )

        most_resistant = prediction_frame.iloc[0]
        most_sensitive = prediction_frame.iloc[-1]
        cols = st.columns(3)
        cols[0].metric("Structure class", group)
        cols[1].metric(
            "Highest predicted resistance",
            most_resistant["drug"],
            f"{most_resistant['predicted_IC50_fold_vs_WT']:.2f}× WT IC50",
        )
        cols[2].metric(
            "Highest predicted sensitivity",
            most_sensitive["drug"],
            f"{most_sensitive['predicted_IC50_fold_vs_WT']:.2f}× WT IC50",
        )

        st.markdown(
            """
<div class="result">
<strong>How to read the graph:</strong> Values above zero indicate predicted
relative resistance; values below zero indicate predicted relative sensitivity.
</div>
""",
            unsafe_allow_html=True,
        )

        st.bar_chart(
            prediction_frame.set_index("drug")[["predicted_log2_ratio"]]
        )

        display = prediction_frame.rename(
            columns={
                "drug": "Drug",
                "predicted_log2_ratio": "Predicted log2(mutant/WT IC50)",
                "predicted_IC50_fold_vs_WT": "Predicted IC50 fold vs WT",
                "interpretation": "Interpretation",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted log2(mutant/WT IC50)": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Predicted IC50 fold vs WT": st.column_config.NumberColumn(
                    format="%.2f"
                ),
            },
        )

        st.download_button(
            "Download predicted profile",
            prediction_frame.to_csv(index=False),
            f"{mutation}_predicted_TKI_profile.csv",
            "text/csv",
        )

        st.caption(
            "Because the selected final model uses drug identity and structural "
            "group, mutations in the same group receive the same predicted profile."
        )


elif page == "Model Comparison":
    st.header("Does structure–function grouping outperform exon location?")

    table = comparison.set_index("model")
    exon = table.loc["drug_plus_exon"]
    structure = table.loc["drug_plus_structure"]

    cols = st.columns(4)
    cols[0].metric("Exon model R²", f"{exon.R2:.3f}")
    cols[1].metric(
        "Structure-group R²",
        f"{structure.R2:.3f}",
        delta=f"{structure.R2 - exon.R2:+.3f}",
    )
    cols[2].metric("Exon Spearman", f"{exon.Spearman:.3f}")
    cols[3].metric(
        "Structure-group Spearman",
        f"{structure.Spearman:.3f}",
        delta=f"{structure.Spearman - exon.Spearman:+.3f}",
    )

    st.success(
        "Drug identity plus the published structure–function group was the "
        "best validated representation."
    )
    st.bar_chart(comparison.set_index("model")[["R2", "Spearman"]])
    st.dataframe(
        comparison[
            ["model", "n_features", "MAE", "RMSE", "R2", "Spearman"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Final-model feature importance")
    st.bar_chart(importance.set_index("feature")[["importance"]])
    st.caption(
        "Feature importance shows predictive use by CatBoost; it does not prove causality."
    )


elif page == "Validation Results":
    st.header("Mutation-held-out validation")
    metrics = metadata["cross_validated_metrics"]

    cols = st.columns(4)
    cols[0].metric("R²", f"{metrics['R2']:.3f}")
    cols[1].metric("Spearman", f"{metrics['Spearman']:.3f}")
    cols[2].metric("MAE", f"{metrics['MAE']:.3f}")
    cols[3].metric("RMSE", f"{metrics['RMSE']:.3f}")

    st.info(
        "All measurements for each mutation were kept in the same fold. "
        "Every out-of-fold prediction was therefore generated when that "
        "mutation was absent from training."
    )

    scatter = oof[["response", "prediction"]].rename(
        columns={"response": "Experimental response", "prediction": "Prediction"}
    )
    st.scatter_chart(
        scatter,
        x="Experimental response",
        y="Prediction",
    )

    st.subheader("Performance by drug")
    st.dataframe(
        per_drug.sort_values("Spearman", ascending=False),
        hide_index=True,
        use_container_width=True,
    )


elif page == "How It Works":
    st.header("What the model learns")
    st.markdown(
        """
### Prediction target
The target is the median **log2(mutant IC50 / wild-type IC50)** for each
mutation–drug pair.

- **0:** mutant and WT have similar IC50 values.
- **+1:** mutant IC50 is approximately twofold higher than WT.
- **−1:** mutant IC50 is approximately half that of WT.

### Model inputs
The selected CatBoost model uses:

1. **Drug identity**, one of the 18 TKIs in the source workbook.
2. **Published EGFR structure–function group**, one of five mechanistic classes.

### Why CatBoost?
CatBoost builds boosted decision trees sequentially and handles categorical
variables directly. Each new tree attempts to reduce errors left by previous
trees.

### Why the structural group is locked
The structural group is a published classification associated with the
mutation. Allowing users to choose an unrelated group would create combinations
that were not represented biologically in the source data.
"""
    )


elif page == "Pipeline":
    st.header("End-to-end machine-learning pipeline")
    steps = [
        (
            "1 · Load the source data",
            "Read mutation, structure group, exon, drug, and triplicate relative IC50 measurements from Nature Figure 2 Panel A.",
        ),
        (
            "2 · Aggregate replicates",
            "Use the median of the available replicate measurements for each mutation–drug pair.",
        ),
        (
            "3 · Define the target",
            "Predict log2(mutant IC50 / WT IC50).",
        ),
        (
            "4 · Compare biological representations",
            "Evaluate drug-only, exon-based, structure-group, combined, and mutation-descriptor feature sets.",
        ),
        (
            "5 · Prevent mutation leakage",
            "Use five-fold GroupKFold so that all rows belonging to one mutation remain together.",
        ),
        (
            "6 · Tune CatBoost",
            "Tune depth and regularization only for the best-performing representation.",
        ),
        (
            "7 · Fit the final model",
            "Retrain the selected model on all 1,380 mutation–drug rows.",
        ),
        (
            "8 · Deploy safely",
            "Map each selected mutation automatically to its published structural group and predict its 18-drug profile.",
        ),
    ]

    for title, description in steps:
        st.markdown(
            f'<div class="result"><strong>{title}</strong><br>{description}</div>',
            unsafe_allow_html=True,
        )


elif page == "Limitations":
    st.header("Limitations and responsible interpretation")
    st.error(
        "This is an experimental relative-response model, not a clinical treatment recommendation."
    )
    st.markdown(
        """
- The model uses expert-defined structural labels; it does not analyze a raw PDB structure.
- The final selected model uses structural group rather than mutation identity directly.
- Therefore, mutations in the same structural group receive the same predicted drug profile.
- The cross-validated R² is approximately 0.43, which is scientifically useful but not clinically sufficient.
- Predictions are restricted to the 18 drugs and five structural groups in the training workbook.
- External experimental validation is still required.
- Feature importance is predictive, not causal.
"""
    )

st.markdown(
    '<div class="footer">EGFR Structure–Function Drug Response Model · Experimental research tool · Not clinical advice</div>',
    unsafe_allow_html=True,
)
