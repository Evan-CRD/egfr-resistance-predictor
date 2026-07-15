from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "strong_model_outputs"
SOURCE_WORKBOOK = ROOT / "2021.8.12 Figure 2 Source Data.xlsx"

GITHUB_URL = "https://github.com/Evan-CRD/egfr-resistance-predictor"
PAPER_URL = "https://www.nature.com/articles/s41586-021-03898-1"
PAPER_CODE_URL = (
    "https://github.com/MD-Anderson-Bioinformatics/"
    "EGFR-Structure-Function-Nature-Manuscript"
)

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
.step{border:1px solid rgba(105,125,145,.28);padding:1.05rem 1.2rem;background:linear-gradient(180deg,#101720,#080c11);border-radius:16px;margin:.7rem 0}
.step-number{color:var(--cyan);font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
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
            "Upload the complete project, including `strong_model_outputs`."
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
        "folds": OUT / "best_model_fold_metrics.csv",
        "tuning": OUT / "catboost_tuning_results.csv",
    }
    for label, path in files.items():
        require_file(path, label)

    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    data = pd.read_csv(files["data"])
    comparison = pd.read_csv(files["comparison"])
    oof = pd.read_csv(files["oof"])
    per_drug = pd.read_csv(files["per_drug"])
    importance = pd.read_csv(files["importance"])
    folds = pd.read_csv(files["folds"])
    tuning = pd.read_csv(files["tuning"])

    return (
        metadata,
        data,
        comparison,
        oof,
        per_drug,
        importance,
        folds,
        tuning,
    )


(
    metadata,
    data,
    comparison,
    oof,
    per_drug,
    importance,
    folds,
    tuning,
) = load_all()
model = load_model()

st.markdown(
    """
<div class="hero">
<div class="k">Machine learning · EGFR · TKI response</div>
<h1>EGFR Structure–Function Drug Response Model</h1>
<p>Predict relative inhibitor response from drug identity and published EGFR
structure–function groups, and test whether a mechanistic structural
representation outperforms the conventional exon-based representation.</p>
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
        "Dataset Source",
        "Detailed Pipeline",
        "How It Works",
        "Limitations",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"[View project source on GitHub]({GITHUB_URL})")
st.sidebar.caption("Version 6.0 · Experimental values + source methodology")


if page == "Prediction":
    st.header("Experimental and predicted drug-response profile")

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

    mutation = st.selectbox(
        "EGFR mutation",
        mapping["mutation"].unique().tolist(),
    )
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
<strong>{group}</strong> structure–function class. The group cannot be changed
independently, preventing unsupported mutation–group combinations.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Show experimental and predicted values",
        type="primary",
        use_container_width=True,
    ):
        # Final deployed prediction trained on all rows.
        deployed = pd.DataFrame(
            {
                "drug": metadata["known_drugs"],
                "structure_group": group,
            }
        )
        expected = metadata["feature_columns"]
        missing = [name for name in expected if name not in deployed.columns]
        if missing:
            st.error(
                "The saved model expects inputs that are not supplied here: "
                + ", ".join(missing)
            )
            st.stop()

        deployed["deployed_prediction"] = model.predict(deployed[expected])

        # Experimental values and honest mutation-held-out predictions.
        held_out = (
            oof.loc[oof["mutation"] == mutation, ["drug", "response", "prediction"]]
            .rename(
                columns={
                    "response": "experimental_value",
                    "prediction": "held_out_prediction",
                }
            )
        )

        profile = deployed.merge(held_out, on="drug", how="left")
        profile["experimental_fold_vs_WT"] = np.power(
            2.0, profile["experimental_value"]
        )
        profile["held_out_fold_vs_WT"] = np.power(
            2.0, profile["held_out_prediction"]
        )
        profile["deployed_fold_vs_WT"] = np.power(
            2.0, profile["deployed_prediction"]
        )
        profile["absolute_error"] = (
            profile["experimental_value"] - profile["held_out_prediction"]
        ).abs()
        profile = profile.sort_values("experimental_value", ascending=False)

        available = profile["experimental_value"].notna().sum()
        if available == 0:
            st.warning(
                "No experimental rows were found for this mutation in the "
                "out-of-fold prediction file."
            )
        else:
            mae_for_mutation = profile["absolute_error"].mean()
            spearman_for_mutation = profile[
                ["experimental_value", "held_out_prediction"]
            ].corr(method="spearman").iloc[0, 1]

            cols = st.columns(4)
            cols[0].metric("Structure class", group)
            cols[1].metric("Drugs with experimental data", f"{available}")
            cols[2].metric("Mutation-specific MAE", f"{mae_for_mutation:.3f}")
            cols[3].metric(
                "Mutation-specific Spearman",
                f"{spearman_for_mutation:.3f}",
            )

        st.markdown(
            """
<div class="result">
<strong>What is being compared?</strong><br>
<strong>Experimental value</strong> is the median measured
log₂(mutant IC50 / WT IC50). <strong>Held-out prediction</strong> was generated
when this entire mutation was absent from model training. The held-out value is
therefore the fairest comparison with the experiment.
</div>
""",
            unsafe_allow_html=True,
        )

        chart = (
            profile.set_index("drug")[
                ["experimental_value", "held_out_prediction"]
            ]
            .rename(
                columns={
                    "experimental_value": "Experimental",
                    "held_out_prediction": "Held-out prediction",
                }
            )
        )
        st.bar_chart(chart)

        display = profile[
            [
                "drug",
                "experimental_value",
                "held_out_prediction",
                "absolute_error",
                "experimental_fold_vs_WT",
                "held_out_fold_vs_WT",
            ]
        ].rename(
            columns={
                "drug": "Drug",
                "experimental_value": "Experimental log₂ ratio",
                "held_out_prediction": "Held-out predicted log₂ ratio",
                "absolute_error": "Absolute error",
                "experimental_fold_vs_WT": "Experimental fold vs WT",
                "held_out_fold_vs_WT": "Predicted fold vs WT",
            }
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Experimental log₂ ratio": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Held-out predicted log₂ ratio": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Absolute error": st.column_config.NumberColumn(format="%.3f"),
                "Experimental fold vs WT": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Predicted fold vs WT": st.column_config.NumberColumn(
                    format="%.2f"
                ),
            },
        )

        st.download_button(
            "Download this mutation's experimental and predicted profile",
            profile.to_csv(index=False),
            f"{mutation}_experimental_and_predicted_profile.csv",
            "text/csv",
        )

        with st.expander("What is the deployed prediction?"):
            st.write(
                "The deployed prediction is produced by the final CatBoost "
                "model after it was retrained on all available data. It is "
                "included in the downloaded table, but the graph emphasizes "
                "the held-out prediction because that prediction did not use "
                "this mutation during training."
            )


elif page == "Model Comparison":
    st.header("Which biological representation predicts drug response best?")

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
        "best mutation-held-out feature set."
    )

    st.markdown(
        """
- **Drug only** is a baseline: can the drug name alone predict response?
- **Drug + exon** identifies where the mutation lies in the gene.
- **Drug + structure** identifies how the mutation is predicted to alter the kinase.
- **Drug + structure + exon** tests whether exon adds information after structure is known.
- **Mechanism enhanced** additionally includes simple mutation-name and amino-acid descriptors.
"""
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
        "Every out-of-fold prediction was generated when that entire mutation "
        "was absent from training."
    )

    scatter = oof[["response", "prediction"]].rename(
        columns={
            "response": "Experimental response",
            "prediction": "Held-out prediction",
        }
    )
    st.scatter_chart(
        scatter,
        x="Experimental response",
        y="Held-out prediction",
    )

    st.subheader("Results for each validation fold")
    st.dataframe(folds, hide_index=True, use_container_width=True)

    st.subheader("Performance by drug")
    st.dataframe(
        per_drug.sort_values("Spearman", ascending=False),
        hide_index=True,
        use_container_width=True,
    )


elif page == "Dataset Source":
    st.header("Dataset source")

    st.markdown(
        f"""
This project uses the Figure 2 source data from:

**Robichaux, J. P. et al. _Structure-based classification predicts drug
response in EGFR-mutant NSCLC._ Nature 597, 732–737 (2021).**

[Open the Nature paper]({PAPER_URL})  
[Open the authors' analysis repository]({PAPER_CODE_URL})
"""
    )

    cols = st.columns(4)
    cols[0].metric("Mutation labels", f"{metadata['n_mutations']}")
    cols[1].metric("TKIs", f"{metadata['n_drugs']}")
    cols[2].metric("Mutation–drug rows", f"{metadata['n_rows']:,}")
    cols[3].metric("Response used", "Median replicate")

    st.markdown(
        """
The published dataset provides the mutation, exon assignment,
structure–function group, drug identity, and relative drug-response
measurements used in this project. The published structure–function labels
were used as provided by the study; they were not created by our machine
learning model.
"""
    )

    st.subheader("Dataset preview")
    preview_columns = [
        "mutation",
        "structure_group",
        "exon1",
        "exon2",
        "exon3",
        "drug",
        "response",
        "n_replicates",
    ]
    st.dataframe(
        data[preview_columns].head(100),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Showing the first 100 cleaned mutation–drug rows.")

    if SOURCE_WORKBOOK.exists():
        st.download_button(
            "Download the original Figure 2 source-data workbook",
            SOURCE_WORKBOOK.read_bytes(),
            SOURCE_WORKBOOK.name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.download_button(
        "Download the cleaned long-format dataset",
        data.to_csv(index=False),
        "nature_egfr_long_format.csv",
        "text/csv",
    )


elif page == "Detailed Pipeline":
    st.header("Machine-learning pipeline")

    st.markdown(
        """
The project follows four main stages. Open each section for a little more
detail.
"""
    )

    st.graphviz_chart(
        """
digraph Pipeline {
    rankdir=LR;
    graph [bgcolor="transparent", pad="0.25", nodesep="0.45", ranksep="0.7"];
    node [shape=box, style="rounded,filled", fillcolor="#101720",
          color="#42d8e8", fontcolor="white", fontname="Arial",
          margin="0.18,0.12"];
    edge [color="#34d5a7", penwidth=2, arrowsize=0.8];

    A [label="1. Dataset"];
    B [label="2. Median replicate\nresponse"];
    C [label="3. Five candidate\nfeature sets"];
    D [label="4. CatBoost\nregression"];
    E [label="5. Mutation-held-out\nvalidation"];

    A -> B -> C -> D -> E;
}
""",
        use_container_width=True,
    )

    with st.expander(
        "1 · Dataset: mutation, structural group, exons, drug, and relative drug response",
        expanded=True,
    ):
        st.markdown(
            """
The source spreadsheet is converted into **one row per mutation–drug pair**.
Each row includes:

- the EGFR mutation;
- its published structure–function group;
- up to three exon fields;
- the TKI used;
- the measured relative drug response.

The response is the published log ratio comparing mutant and wild-type IC50.
Positive values indicate relative resistance, while negative values indicate
relative sensitivity.
"""
        )

        sample = data[
            ["mutation", "structure_group", "exon1", "drug", "response"]
        ].head(12)
        st.dataframe(sample, hide_index=True, use_container_width=True)

    with st.expander(
        "2 · Median of the available replicate values",
        expanded=False,
    ):
        st.markdown(
            """
Several replicate measurements may be available for the same mutation–drug
pair. The pipeline uses their **median** as the response given to the model.

The median reduces the influence of one unusually high or low replicate. The
mean, standard deviation, and number of replicates are also saved for quality
checking.
"""
        )

        replicate_example = (
            data[["mutation", "drug", "replicate_mean", "replicate_sd", "response"]]
            .dropna()
            .head(20)
            .rename(
                columns={
                    "replicate_mean": "Replicate mean",
                    "replicate_sd": "Replicate SD",
                    "response": "Median used as target",
                }
            )
        )
        st.dataframe(
            replicate_example,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(
        "3 · Generate five candidate feature sets",
        expanded=False,
    ):
        st.markdown(
            """
Five versions of the input information were tested:

1. **Drug only** — baseline model.
2. **Drug + exon** — mutation location in the gene.
3. **Drug + structure group** — published structural mechanism.
4. **Drug + structure group + exon** — tests whether exon adds information
   after structure is known.
5. **Mechanism enhanced** — adds simple mutation-name and amino-acid
   descriptors.

Every candidate was evaluated using the same validation procedure so the
comparison was fair.
"""
        )

        feature_chart = comparison[
            ["model", "MAE", "R2", "Spearman"]
        ].set_index("model")
        st.bar_chart(feature_chart[["R2", "Spearman"]])

        st.dataframe(
            comparison[
                ["model", "n_features", "MAE", "RMSE", "R2", "Spearman"]
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(
        "4 · CatBoost regression",
        expanded=False,
    ):
        st.markdown(
            """
CatBoost is a boosted decision-tree regression model. It builds trees
sequentially, and each new tree tries to correct errors left by the earlier
trees.

It is useful here because the most important inputs—drug, exon, and structural
group—are categorical labels. CatBoost can use these labels directly and can
learn nonlinear interactions such as a particular drug behaving differently
for different structural groups.
"""
        )

        st.json(
            {
                "selected_feature_set": metadata["best_feature_set"],
                "selected_features": metadata["feature_columns"],
                "parameters": metadata["parameters"],
            }
        )

    with st.expander(
        "5 · Mutation-held-out validation",
        expanded=False,
    ):
        st.markdown(
            """
All rows belonging to the same mutation are kept together. When a mutation is
used for testing, none of its drug-response measurements are present in the
training data.

This tests whether the model can generalize to a mutation it has not seen,
rather than simply remembering other drug measurements for the same mutation.
The process is repeated across five folds, producing an out-of-fold prediction
for every row.
"""
        )

        metrics = metadata["cross_validated_metrics"]
        cols = st.columns(4)
        cols[0].metric("R²", f"{metrics['R2']:.3f}")
        cols[1].metric("Spearman", f"{metrics['Spearman']:.3f}")
        cols[2].metric("MAE", f"{metrics['MAE']:.3f}")
        cols[3].metric("RMSE", f"{metrics['RMSE']:.3f}")

        validation_plot = oof[["response", "prediction"]].rename(
            columns={
                "response": "Experimental response",
                "prediction": "Held-out prediction",
            }
        )
        st.scatter_chart(
            validation_plot,
            x="Experimental response",
            y="Held-out prediction",
        )

    st.markdown(
        """
<div class="result">
<strong>Final outcome:</strong> Drug identity plus the published
structure–function group was the best-performing feature representation and
was retrained on the full dataset for deployment.
</div>
""",
        unsafe_allow_html=True,
    )


elif page == "How It Works":
    st.header("What the model learns")
    st.markdown(
        """
### Prediction target

The target is the median **log₂(mutant IC50 / wild-type IC50)** for each
mutation–drug pair.

- **0:** mutant and WT have similar IC50 values.
- **+1:** mutant IC50 is approximately twofold higher than WT.
- **−1:** mutant IC50 is approximately half that of WT.

### Selected inputs

The best held-out CatBoost model uses:

1. **Drug identity**, one of the 18 TKIs in the source workbook.
2. **Published EGFR structure–function group**, one of five displayed classes.

### Why CatBoost?

CatBoost builds boosted decision trees sequentially and handles categorical
variables directly. Each new tree reduces prediction errors left by earlier
trees.

### Why the structure group is locked

The group is a published classification associated with the mutation.
Allowing an unrelated group would create an input combination unsupported by
the source study.
"""
    )


elif page == "Limitations":
    st.header("Limitations and responsible interpretation")
    st.error(
        "This is an experimental relative-response model, not a clinical treatment recommendation."
    )
    st.markdown(
        """
- The model uses published expert-defined structural labels; it does not infer
  a structural group from a raw PDB structure.
- The final selected model uses structural group rather than mutation identity
  directly, so mutations in the same group receive the same final deployed
  profile.
- The experimental dataset comes from engineered Ba/F3 cell lines and cannot
  capture every biological feature of a human tumor.
- R² around 0.43 is scientifically informative but not clinically sufficient.
- Predictions are limited to the 18 drugs and five displayed structural groups
  represented in the training data.
- Out-of-fold predictions estimate generalization within this dataset; an
  independent external dataset would be a stronger validation.
- Feature importance is predictive, not causal.
"""
    )

st.markdown(
    '<div class="footer">EGFR Structure–Function Drug Response Model · Experimental research tool · Not clinical advice</div>',
    unsafe_allow_html=True,
)
