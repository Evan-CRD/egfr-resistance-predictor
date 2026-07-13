from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.predict import predict_auc_ratio


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "egfr_structural_features_rich.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.json"
GITHUB_URL = "https://github.com/Evan-CRD/egfr-resistance-predictor"
RESISTANCE_THRESHOLD = 1.5


st.set_page_config(
    page_title="EGFR TKI Resistance Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(75, 117, 255, 0.10), transparent 30%),
            radial-gradient(circle at 95% 10%, rgba(48, 205, 164, 0.08), transparent 28%);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem 2.3rem;
        border: 1px solid rgba(120, 130, 170, 0.22);
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(42, 61, 112, 0.96),
            rgba(25, 106, 112, 0.92)
        );
        box-shadow: 0 18px 48px rgba(20, 30, 60, 0.18);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        line-height: 1.12;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.08rem;
        max-width: 850px;
        margin: 0.8rem 0 0 0;
    }

    .eyebrow {
        color: #9fe6d5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .info-card {
        border: 1px solid rgba(120, 130, 170, 0.22);
        border-radius: 16px;
        padding: 1.15rem 1.2rem;
        background: rgba(255, 255, 255, 0.03);
        min-height: 145px;
    }

    .info-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.02rem;
    }

    .info-card p {
        margin: 0;
        opacity: 0.82;
        line-height: 1.5;
        font-size: 0.93rem;
    }

    .result-panel {
        border: 1px solid rgba(83, 166, 145, 0.34);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        background: rgba(53, 151, 130, 0.07);
        margin: 0.8rem 0 1.25rem 0;
    }

    .small-note {
        font-size: 0.86rem;
        opacity: 0.78;
        line-height: 1.48;
    }

    .pipeline-step {
        border-left: 4px solid rgba(71, 149, 224, 0.78);
        padding: 0.25rem 0 0.25rem 0.8rem;
        margin: 0.55rem 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 130, 170, 0.20);
        border-radius: 15px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.025);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .footer {
        text-align: center;
        opacity: 0.65;
        font-size: 0.82rem;
        padding-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    return frame


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def mean_feature_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows into one prediction row."""
    output = frame.iloc[[0]].copy()
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    output.loc[:, numeric_columns] = frame[numeric_columns].mean().values
    return output


def classify_auc(value: float) -> tuple[str, str]:
    if value >= RESISTANCE_THRESHOLD:
        return "Resistant-like", "Above the exploratory 1.50 threshold"
    return "Sensitive-like", "Below the exploratory 1.50 threshold"


def normalized_feature_profile(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Create a robust relative profile using median and IQR."""
    available = [column for column in columns if column in selected.columns]
    rows = []

    labels = {
        "delta_sidechain_volume_A3": "Side-chain volume change",
        "delta_local_packing_atoms_6A": "Local packing change",
        "delta_steric_clashes_2A": "Steric-clash change",
        "delta_protein_drug_atom_contacts_4p5A": "Protein–drug contact change",
        "distance_to_ATP_pocket_centroid_A": "Distance to ATP pocket",
        "mutation_site_DFI": "Mutation-site DFI",
        "auxiliary_predicted_mutation_effect": "Auxiliary mutation-effect score",
    }

    for column in available:
        values = pd.to_numeric(reference[column], errors="coerce")
        selected_value = float(pd.to_numeric(selected[column], errors="coerce").mean())

        median = float(values.median())
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr == 0:
            relative = 0.0
        else:
            relative = (selected_value - median) / iqr

        rows.append(
            {
                "Feature": labels.get(column, column),
                "Relative to dataset median": float(np.clip(relative, -3, 3)),
                "Raw value": selected_value,
            }
        )

    return pd.DataFrame(rows)


data = load_data()
metrics = load_metrics()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Structural bioinformatics · Machine learning · EGFR</div>
        <h1>EGFR TKI Resistance Explorer</h1>
        <p>
            An end-to-end research prototype that connects experimental inhibitor-response
            measurements with mutant structure generation, kinase-aware feature engineering,
            drug descriptors, DFI, and an auxiliary mutation-effect model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_middle, top_right = st.columns([1.15, 1, 1])

with top_left:
    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Research question</h4>
            <p>Which structural changes in mutant EGFR are associated with altered response to tyrosine kinase inhibitors?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_middle:
    st.markdown(
        f"""
        <div class="info-card">
            <h4>📊 Current dataset</h4>
            <p>{len(data)} modeled mutation–drug rows, {data["mutation"].nunique()} distinct kinase-domain mutations, and {data["drug"].nunique()} represented inhibitors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
        <div class="info-card">
            <h4>⚠️ Appropriate use</h4>
            <p>Educational and exploratory only. The current model is not clinically validated and should not guide treatment decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("## 🧬 Project navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Prediction explorer",
        "Model results",
        "Pipeline",
        "Batch prediction",
        "Limitations",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"[View the source code on GitHub]({GITHUB_URL})"
)
st.sidebar.caption(
    "Version 1.0 · exploratory kinase-domain model"
)


if page == "Prediction explorer":
    st.header("Prediction explorer")
    st.write(
        "Choose a mutation and inhibitor represented in the precomputed structural dataset."
    )

    control_col, context_col = st.columns([0.9, 1.35], gap="large")

    with control_col:
        mutation = st.selectbox(
            "EGFR mutation",
            sorted(data["mutation"].unique()),
            help="Only kinase-domain substitutions supported by the available structures are shown.",
        )

        available_drugs = sorted(
            data.loc[data["mutation"] == mutation, "drug"].unique()
        )
        drug = st.selectbox(
            "Tyrosine kinase inhibitor",
            available_drugs,
        )

        run_prediction = st.button(
            "Run exploratory prediction",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "The app uses precomputed structure-derived features for the selected pair."
        )

    selected = data[
        (data["mutation"] == mutation) &
        (data["drug"] == drug)
    ].copy()

    with context_col:
        st.subheader(f"{mutation} with {drug}")
        st.write(
            f"This selection contains **{len(selected)} experimental row(s)**. "
            "Replicate feature values are averaged before prediction."
        )

        context_columns = [
            "position",
            "distance_to_T790_A",
            "distance_to_C797_A",
            "distance_to_ATP_pocket_centroid_A",
        ]
        available_context = [
            column for column in context_columns if column in selected.columns
        ]
        context = selected[available_context].mean(numeric_only=True)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Residue position",
            f'{int(context.get("position", 0))}',
        )
        c2.metric(
            "Distance to T790",
            f'{context.get("distance_to_T790_A", np.nan):.1f} Å',
        )
        c3.metric(
            "Distance to C797",
            f'{context.get("distance_to_C797_A", np.nan):.1f} Å',
        )

    if run_prediction:
        prediction_input = mean_feature_row(selected)
        prediction = predict_auc_ratio(prediction_input)

        predicted_auc = float(
            prediction["predicted_auc_ratio_vs_wt"].iloc[0]
        )
        observed_auc = float(selected["auc_ratio_vs_wt"].mean())
        label, threshold_note = classify_auc(predicted_auc)

        st.markdown(
            """
            <div class="result-panel">
                <strong>Exploratory result</strong><br>
                The model output is shown alongside the experimental mean for this
                represented pair. Because this pair comes from the modeling dataset,
                the prediction should not be interpreted as external validation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_1, result_2, result_3, result_4 = st.columns(4)
        result_1.metric(
            "Predicted AUC ratio vs WT",
            f"{predicted_auc:.3f}",
        )
        result_2.metric(
            "Observed mean AUC ratio",
            f"{observed_auc:.3f}",
            delta=f"{predicted_auc - observed_auc:+.3f}",
            delta_color="off",
        )
        result_3.metric(
            "Interpretation",
            label,
        )
        result_4.metric(
            "Exploratory threshold",
            f"{RESISTANCE_THRESHOLD:.2f}",
        )

        st.caption(threshold_note)

        st.subheader("Structural feature profile")
        profile_columns = [
            "delta_sidechain_volume_A3",
            "delta_local_packing_atoms_6A",
            "delta_steric_clashes_2A",
            "delta_protein_drug_atom_contacts_4p5A",
            "distance_to_ATP_pocket_centroid_A",
            "mutation_site_DFI",
            "auxiliary_predicted_mutation_effect",
        ]
        profile = normalized_feature_profile(
            selected,
            data,
            profile_columns,
        )

        chart_col, table_col = st.columns([1.35, 1], gap="large")
        with chart_col:
            chart_frame = profile.set_index("Feature")[
                ["Relative to dataset median"]
            ]
            st.bar_chart(chart_frame, horizontal=True)
            st.caption(
                "Values are scaled relative to the median and interquartile range "
                "of the current 50-row dataset."
            )

        with table_col:
            st.dataframe(
                profile[["Feature", "Raw value"]],
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("How should I interpret this prediction?"):
            st.markdown(
                """
                - A higher predicted AUC ratio indicates greater modeled cell survival
                  relative to the corresponding WT reference under inhibitor treatment.
                - The **1.50 threshold** is an exploratory project definition, not a
                  clinical standard.
                - The current grouped cross-validation performance is weak for unseen
                  mutations, so this output is best treated as a visualization of the
                  pipeline rather than a reliable clinical prediction.
                """
            )


elif page == "Model results":
    st.header("Model results")
    st.write(
        "The evaluation grouped all measurements from the same mutation together, "
        "reducing mutation-level leakage between training and testing."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric(
        "Grouped CV MAE",
        f'{metrics.get("rich_model_grouped_cv_MAE", 0.362):.3f}',
    )
    metric_2.metric(
        "Grouped CV RMSE",
        f'{metrics.get("rich_model_grouped_cv_RMSE", 0.483):.3f}',
    )
    metric_3.metric(
        "Grouped CV R²",
        f'{metrics.get("rich_model_grouped_cv_R2", -0.259):.3f}',
    )
    metric_4.metric(
        "Grouped CV Spearman",
        f'{metrics.get("rich_model_grouped_cv_Spearman", -0.093):.3f}',
    )

    st.subheader("What the metrics mean")
    st.markdown(
        """
        - **MAE** is the average absolute difference between observed and predicted AUC ratios.
        - **RMSE** gives larger errors more weight.
        - **R² below zero** means the model did not outperform a training-fold mean predictor
          on unseen mutation groups.
        - **Spearman near zero** means the model did not reliably rank unseen mutation–drug
          responses.
        """
    )

    st.subheader("Model development sequence")
    comparison = pd.DataFrame(
        {
            "Representation": [
                "Original structural model",
                "Kinase-wide expansion",
                "Rich interaction model",
                "Rich + auxiliary mutation score",
            ],
            "Added information": [
                "Local geometry and chemistry",
                "Kinase landmarks and contact-network context",
                "Drug chemistry, clashes, contacts, packing, and DFI",
                "Transfer of general mutation-effect information",
            ],
            "Conclusion": [
                "Working baseline pipeline",
                "Location alone was insufficient",
                "Lower error but limited generalization",
                "Small incremental benefit",
            ],
        }
    )
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "The main scientific finding is not that the model is clinically predictive. "
        "It is that a rigorous pipeline can quantify how progressively richer structural "
        "representations affect generalization under a small-data constraint."
    )


elif page == "Pipeline":
    st.header("End-to-end pipeline")
    st.write(
        "The project is organized as a reproducible sequence of data, structure, "
        "feature, and modeling stages."
    )

    steps = [
        (
            "1 · Experimental outcome extraction",
            "Normalized dose-response measurements were parsed from the Hayes workbooks "
            "and converted into mutation–drug AUC ratios relative to WT.",
        ),
        (
            "2 · Drug-bound structure selection",
            "Experimentally determined EGFR kinase–inhibitor structures were used for "
            "erlotinib, afatinib, dacomitinib, and osimertinib.",
        ),
        (
            "3 · Automated mutant generation",
            "Supported kinase-domain substitutions were introduced into their "
            "corresponding drug-bound structures.",
        ),
        (
            "4 · Structural feature engineering",
            "Local chemistry, packing, contacts, kinase landmarks, graph descriptors, "
            "DFI, and drug physicochemical properties were calculated.",
        ),
        (
            "5 · Auxiliary mutation-effect learning",
            "A separate model learned general EGFR mutation effects from 4,802 variants "
            "and supplied an auxiliary score to the resistance model.",
        ),
        (
            "6 · Grouped machine-learning evaluation",
            "Mutation-grouped cross-validation prevented the same mutation from appearing "
            "in both training and testing folds.",
        ),
        (
            "7 · Interactive deployment",
            "The fitted pipeline and precomputed feature table were packaged into this "
            "Streamlit research interface.",
        ),
    ]

    for title, description in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <strong>{title}</strong><br>
                <span class="small-note">{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Repository contents")
    st.code(
        """egfr-resistance-predictor/
├── app.py
├── src/predict.py
├── models/
├── data/
├── results/
├── notebooks/
├── requirements.txt
└── README.md""",
        language="text",
    )


elif page == "Batch prediction":
    st.header("Batch prediction")
    st.write(
        "Upload rows produced by the same rich-feature pipeline. The model requires "
        "the feature schema stored in `models/feature_schema.json`."
    )

    uploaded = st.file_uploader(
        "Upload a rich-feature CSV",
        type=["csv"],
    )

    example_path = PROJECT_ROOT / "data" / "example_prediction_rows.csv"
    if example_path.exists():
        st.download_button(
            "Download example input",
            data=example_path.read_bytes(),
            file_name="example_prediction_rows.csv",
            mime="text/csv",
        )

    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        st.write(f"Uploaded **{len(frame)} row(s)**.")

        try:
            output = predict_auc_ratio(frame)
            st.success("Prediction completed.")
            st.dataframe(
                output[
                    [
                        column
                        for column in [
                            "mutation",
                            "drug",
                            "predicted_auc_ratio_vs_wt",
                            "predicted_resistant",
                        ]
                        if column in output.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="egfr_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))


elif page == "Limitations":
    st.header("Limitations and responsible use")

    st.error(
        "This application is not a medical device and must not be used to select "
        "cancer treatment."
    )

    limitation_data = pd.DataFrame(
        {
            "Limitation": [
                "Small resistance dataset",
                "Restricted structural coverage",
                "Static mutant models",
                "Proxy interaction features",
                "Weak unseen-mutation performance",
                "Training-pair exploration",
            ],
            "Current implication": [
                "Only 50 supported mutation–drug rows were available.",
                "Only kinase-domain substitutions present in the chosen PDB structures were modeled.",
                "Mutants were not subjected to validated ligand-aware relaxation.",
                "Hydrogen bonds and clashes are geometry-based rather than energetic calculations.",
                "Grouped cross-validated R² remained negative.",
                "Predictions shown for supported pairs are not independent external tests.",
            ],
            "Planned improvement": [
                "Integrate independent EGFR response datasets.",
                "Add validated structures or carefully modeled missing regions.",
                "Introduce validated minimization and stability workflows.",
                "Add pocket volume, stability, and interaction-energy descriptors.",
                "Retrain with more independent mutation–drug pairs.",
                "Add external validation and calibrated uncertainty.",
            ],
        }
    )

    st.dataframe(
        limitation_data,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("What this project demonstrates")
    st.markdown(
        """
        - Reproducible processing of real biological assay data
        - Automated protein mutation modeling
        - Physics-informed and structure-informed feature engineering
        - Transfer of information from a larger auxiliary mutation dataset
        - Leakage-aware machine-learning validation
        - Deployment of a modular research application
        """
    )


st.markdown(
    """
    <div class="footer">
        Built as an educational computational biology project ·
        Structural predictions are exploratory and not clinically validated
    </div>
    """,
    unsafe_allow_html=True,
)
