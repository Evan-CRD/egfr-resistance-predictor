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
    st.header("Dataset source and experimental design")

    st.markdown(
        f"""
### Primary source

**Robichaux, J. P. et al. _Structure-based classification predicts drug
response in EGFR-mutant NSCLC._ Nature 597, 732–737 (2021).**

[Open the Nature paper]({PAPER_URL})  
[Open the authors' analysis repository]({PAPER_CODE_URL})
"""
    )

    cols = st.columns(4)
    cols[0].metric("EGFR mutations/cell lines", "76–77")
    cols[1].metric("TKIs screened", "18")
    cols[2].metric("Rows used here", f"{metadata['n_rows']:,}")
    cols[3].metric("Replicates summarized", "Median of 3")

    st.markdown(
        """
### How the experimental response data were generated

1. **Create mutant EGFR cell lines.** The authors introduced different mutant
   EGFR genes into Ba/F3 cells and selected cells expressing EGFR.

2. **Treat each cell line with EGFR inhibitors.** Cells were plated in
   384-well plates and exposed to seven concentrations of each TKI or a DMSO
   control.

3. **Measure cell viability after 72 hours.** CellTiter-Glo luminescence was
   used to measure how many cells remained viable.

4. **Fit dose–response curves.** A nonlinear dose–response curve was fit for
   each mutation–drug pair, and the IC50—the concentration required for 50%
   inhibition—was estimated.

5. **Normalize each mutant to wild-type EGFR.** The mutant IC50 was divided by
   the WT IC50 for the same drug. The published heat map used the logarithm of
   this mutant/WT ratio.

6. **Repeat the measurements.** Drug screens were performed in technical
   triplicate and in duplicate or triplicate biological experiments. Our
   pipeline uses the median response supplied in the Figure 2 source data.
"""
    )

    st.markdown(
        """
### How the structure–function groups were assigned

The groups were **not invented by our CatBoost model**. The paper's authors
combined structural mapping with drug-sensitivity patterns:

- They mapped mutations onto experimentally determined EGFR structures and
  homology models.
- They asked whether each mutation was distant from the binding pocket, in the
  hydrophobic core, in an exon 20 loop, or likely to move the P-loop and/or
  αC-helix.
- They used established mutations—such as L858R, T790M, and known exon 20
  insertions—as structural reference cases.
- They then checked whether mutations assigned to the same group clustered
  together based on their measured responses to the 18 TKIs.
- Drug-sensitivity data further separated subgroups such as
  **T790M-like-3S** (third-generation sensitive) and
  **T790M-like-3R** (third-generation resistant).

The four major biological classes were Classical-Like, T790M-like,
Ex20ins-L, and PACC. This website retains the two T790M-like sensitivity
subclasses as separate labels, giving five displayed categories.
"""
    )

    st.subheader("Dataset used by this project")
    st.dataframe(
        data[
            [
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
            ]
        ].head(100),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("The table preview shows the first 100 rows.")

    if SOURCE_WORKBOOK.exists():
        st.download_button(
            "Download the Nature Figure 2 source-data workbook",
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
    st.header("Detailed end-to-end machine-learning pipeline")

    pipeline_steps = [
        (
            "1 · Read Panel A of the Nature workbook",
            """
The training script opens the Excel worksheet named **Panel A** without
assuming a conventional header row. It reads mutation names, published
structure–function groups, up to three exon fields, drug names, and the
replicate response measurements. It verifies that the workbook has the
expected layout and stops if the sheet cannot be interpreted.
""",
        ),
        (
            "2 · Convert the wide spreadsheet into long format",
            """
The source worksheet stores multiple drugs across columns. The script converts
it into one row per **mutation–drug pair**. Each row contains the mutation,
drug, structural group, exon fields, replicate summary statistics, and the
response target. This produces 1,380 usable mutation–drug rows representing
77 mutation labels and 18 drugs.
""",
        ),
        (
            "3 · Aggregate experimental replicates",
            """
For each mutation–drug pair, the available replicate measurements are
converted to numbers. The **median** becomes the training target because it is
less sensitive to one unusually high or low replicate. The script also saves
the replicate mean, standard deviation, and number of replicates for quality
checking.
""",
        ),
        (
            "4 · Define the prediction target",
            """
The response is the published logarithm of the mutant-to-WT IC50 ratio. A
positive value means the mutant requires more drug than WT and is relatively
resistant. A negative value means the mutant requires less drug and is
relatively sensitive. Zero means mutant and WT have similar IC50 values.
""",
        ),
        (
            "5 · Derive optional mutation descriptors",
            """
The script parses mutation names to extract the number of mutation components,
residue positions, whether the variant contains an insertion, duplication, or
deletion, and indicator variables for common mutations such as T790M, C797S,
L858R, and G719X. For simple substitutions, it calculates approximate changes
in amino-acid hydropathy, volume, charge, aromaticity, and polarity.
""",
        ),
        (
            "6 · Construct five competing feature sets",
            """
The model does not assume in advance that the richest feature set is best. It
compares: **drug only**, **drug + exon**, **drug + structure group**,
**drug + structure + exon**, and a **mechanism-enhanced** set containing all
of those fields plus the mutation descriptors. This directly tests whether
structural grouping adds more predictive information than exon location.
""",
        ),
        (
            "7 · Prepare categorical and numerical features",
            """
Drug, structural group, and exon fields are treated as categorical variables.
CatBoost receives those categories directly rather than requiring manual
one-hot encoding. Numerical features are converted safely, and missing values
are filled using the median of the available values or zero when no median is
available.
""",
        ),
        (
            "8 · Perform five-fold GroupKFold validation by mutation",
            """
The grouping variable is the mutation name. All 18 drug measurements for a
mutation remain in the same fold. During each validation round, the model is
trained on four groups of mutations and tested on the remaining group. This
prevents rows from the same mutation from appearing in both training and test
data and provides a realistic test of performance on unseen mutations.
""",
        ),
        (
            "9 · Train CatBoost regressors",
            """
Each candidate is a CatBoost regression model optimized for RMSE while
reporting MAE. CatBoost builds decision trees sequentially, with later trees
correcting errors made by earlier trees. The base model uses controlled tree
depth, learning rate, L2 regularization, random strength, bagging temperature,
and a fixed random seed for reproducibility.
""",
        ),
        (
            "10 · Compare models using four validation metrics",
            """
For every feature set, the script calculates **MAE**, **RMSE**, **R²**, and
**Spearman correlation** across all out-of-fold predictions. It also calculates
the same metrics separately for each drug so that pooled performance is not
dominated only by differences among drugs.
""",
        ),
        (
            "11 · Select and tune the best feature representation",
            """
Feature sets are ranked primarily by held-out MAE. Only the winning
representation is taken into a second tuning stage, where alternative tree
depths, regularization values, and random-strength settings are evaluated
using the same mutation-held-out design. This avoids tuning every model
excessively on a small dataset.
""",
        ),
        (
            "12 · Save honest out-of-fold predictions",
            """
The file **best_model_oof_predictions.csv** stores, for every mutation–drug
pair, the experimental response and the prediction generated while that
mutation was excluded from training. These values power the first website page
and the validation scatter plot.
""",
        ),
        (
            "13 · Retrain the final deployed model",
            """
After validation and tuning are complete, the selected CatBoost model is
trained once more using all available rows. This final model is saved as
**best_egfr_catboost_model.cbm** and is used for deployment. Its selected
inputs are drug identity and structure–function group.
""",
        ),
        (
            "14 · Save outputs needed for reproducibility",
            """
The pipeline exports the cleaned dataset, feature-set comparison, fold
metrics, per-drug metrics, tuning results, feature importance, out-of-fold
predictions, trained model, and a JSON metadata file describing the target,
features, parameters, known drugs, structural groups, random seed, validation
design, and limitations.
""",
        ),
        (
            "15 · Deploy with biologically valid inputs",
            """
The website reads the mutation-to-group mapping from the cleaned published
data. Selecting a mutation automatically assigns its structural group and exon
information. The group is locked so users cannot create biologically
unsupported mutation–group combinations. The website displays both measured
experimental values and honest held-out predictions.
""",
        ),
    ]

    for index, (title, description) in enumerate(pipeline_steps, start=1):
        st.markdown(
            f"""
<div class="step">
<div class="step-number">Pipeline stage {index}</div>
<strong>{title}</strong><br><br>
{description}
</div>
""",
            unsafe_allow_html=True,
        )

    st.subheader("Final selected model")
    st.json(
        {
            "best_feature_set": metadata["best_feature_set"],
            "feature_columns": metadata["feature_columns"],
            "parameters": metadata["parameters"],
            "validation": metadata["validation"],
            "cross_validated_metrics": metadata["cross_validated_metrics"],
        }
    )

    st.subheader("Hyperparameter candidates")
    st.dataframe(
        tuning.sort_values("MAE"),
        hide_index=True,
        use_container_width=True,
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
