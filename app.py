from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.predict import predict_auc_ratio


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "egfr_all_supported_mutation_drug_features.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.json"
UNSUPPORTED_PATH = PROJECT_ROOT / "data" / "unsupported_mutation_drug_pairs.csv"
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
    :root {
        --bg: #050607;
        --panel: #0D1117;
        --panel-2: #111821;
        --border: #24303D;
        --border-soft: rgba(95, 115, 135, 0.28);
        --text: #F4F7FA;
        --muted: #9AA7B4;
        --cyan: #44D7E8;
        --teal: #33D6A6;
        --purple: #A78BFA;
        --warning: #F3C969;
        --danger: #FF6B7A;
    }

    html, body, [class*="css"] {
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 14% 0%, rgba(68, 215, 232, 0.08), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(167, 139, 250, 0.06), transparent 24%),
            linear-gradient(180deg, #050607 0%, #080B0F 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1260px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07090C 0%, #0A0D11 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 2.3rem 2.4rem;
        border: 1px solid rgba(68, 215, 232, 0.24);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15, 24, 33, 0.98), rgba(9, 18, 24, 0.98));
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.42);
        margin-bottom: 1.35rem;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 2.65rem;
        font-weight: 760;
        margin: 0;
        line-height: 1.08;
        letter-spacing: -0.025em;
    }

    .hero p {
        color: #B9C5D1;
        font-size: 1.08rem;
        max-width: 900px;
        margin: 0.85rem 0 0 0;
        line-height: 1.62;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .info-card, .status-card {
        border: 1px solid var(--border-soft);
        border-radius: 17px;
        padding: 1.2rem 1.25rem;
        background: linear-gradient(180deg, rgba(17, 24, 33, 0.96), rgba(11, 16, 22, 0.96));
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
    }

    .info-card { min-height: 150px; }

    .info-card h4, .status-card h4 {
        color: #FFFFFF;
        margin: 0 0 0.48rem 0;
    }

    .info-card p, .status-card p {
        color: #AAB6C3;
        margin: 0;
        line-height: 1.52;
        font-size: 0.93rem;
    }

    .observed-card {
        border: 1px solid rgba(51, 214, 166, 0.45);
        background: linear-gradient(180deg, rgba(10, 35, 30, 0.88), rgba(8, 22, 20, 0.92));
    }

    .extrapolation-card {
        border: 1px solid rgba(243, 201, 105, 0.45);
        background: linear-gradient(180deg, rgba(40, 31, 10, 0.82), rgba(24, 19, 8, 0.92));
    }

    .pipeline-step {
        border-left: 4px solid var(--cyan);
        padding: 0.38rem 0 0.38rem 0.9rem;
        margin: 0.62rem 0;
        background: linear-gradient(90deg, rgba(68, 215, 232, 0.055), transparent);
        border-radius: 0 10px 10px 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(17, 24, 33, 0.98), rgba(10, 14, 19, 0.98));
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stMetricLabel"] {
        color: #AAB5C1;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #44D7E8, #33D6A6);
        color: #031011;
        border: 0;
        border-radius: 12px;
        font-weight: 750;
        padding: 0.65rem 1rem;
        box-shadow: 0 8px 22px rgba(51, 214, 166, 0.18);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        color: #020708;
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--panel);
        border-color: var(--border);
        color: var(--text);
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        overflow: hidden;
    }

    h1, h2, h3, h4 { color: #FFFFFF; }
    p, li, label, .stMarkdown { color: #D7DFE7; }
    a { color: var(--cyan) !important; }

    .footer {
        text-align: center;
        color: #758292;
        font-size: 0.82rem;
        padding-top: 2.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_data
def load_unsupported() -> pd.DataFrame:
    if UNSUPPORTED_PATH.exists():
        return pd.read_csv(UNSUPPORTED_PATH)
    return pd.DataFrame()


def mean_feature_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows safely without pandas dtype-assignment errors."""
    output = frame.iloc[[0]].copy()
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        output[column] = pd.to_numeric(frame[column], errors="coerce").mean()
    return output


def classify_auc(value: float) -> tuple[str, str]:
    if value >= RESISTANCE_THRESHOLD:
        return "Resistant-like", "Above the exploratory 1.50 threshold"
    return "Sensitive-like", "Below the exploratory 1.50 threshold"


def feature_profile(selected: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    feature_map = {
        "delta_sidechain_volume_A3": "Side-chain volume change",
        "delta_local_packing_atoms_6A": "Local packing change",
        "delta_steric_clashes_2A": "Steric-clash change",
        "delta_protein_drug_atom_contacts_4p5A": "Protein–drug contact change",
        "pocket_residues_lost_4p5A": "Pocket residues lost",
        "putative_polar_contacts_lost": "Polar contacts lost",
        "distance_to_ATP_pocket_centroid_A": "Distance to ATP pocket",
        "mutation_site_DFI": "Mutation-site DFI",
        "auxiliary_predicted_mutation_effect": "Auxiliary mutation-effect score",
    }

    rows = []
    for column, label in feature_map.items():
        if column not in selected.columns:
            continue

        values = pd.to_numeric(reference[column], errors="coerce")
        selected_value = float(pd.to_numeric(selected[column], errors="coerce").mean())
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))

        relative = 0.0 if not np.isfinite(iqr) or iqr == 0 else (selected_value - median) / iqr

        rows.append(
            {
                "Feature": label,
                "Relative to matrix median": float(np.clip(relative, -3, 3)),
                "Raw value": selected_value,
            }
        )

    return pd.DataFrame(rows)


data = load_data()
metrics = load_metrics()
unsupported = load_unsupported()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Structure-generated features · Machine learning · EGFR</div>
        <h1>EGFR TKI Resistance Explorer</h1>
        <p>
            Select a supported EGFR kinase mutation and inhibitor. The app retrieves a
            structurally generated feature vector for that pair and asks the trained model
            to predict its response. Stored experimental outcomes are shown only when they
            genuinely exist.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <h4>🧬 Supported mutations</h4>
            <p>{data["mutation"].nunique()} kinase-domain substitutions with successfully generated structural feature vectors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="info-card">
            <h4>💊 Generated combinations</h4>
            <p>{len(data)} mutation–drug feature rows across {data["drug"].nunique()} represented inhibitors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="info-card">
            <h4>⚠️ Research prototype</h4>
            <p>Predictions are exploratory, not clinically validated, and must not be used for treatment decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("## 🧬 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Prediction explorer",
        "Combination coverage",
        "Model results",
        "Pipeline",
        "Batch prediction",
        "Limitations",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"[View source on GitHub]({GITHUB_URL})")
st.sidebar.caption("Version 2.1 · generated structural feature matrix")


if page == "Prediction explorer":
    st.header("Prediction explorer")
    st.write(
        "Every listed choice has a successfully generated structural feature vector."
    )

    left, right = st.columns([0.9, 1.3], gap="large")

    with left:
        mutation_options = sorted(data["mutation"].unique())
        default_mutation = mutation_options.index("T790M") if "T790M" in mutation_options else 0
        mutation = st.selectbox(
            "EGFR mutation",
            mutation_options,
            index=default_mutation,
        )

        available_drugs = sorted(
            data.loc[data["mutation"] == mutation, "drug"].unique()
        )
        default_drug = available_drugs.index("Dacomitinib") if "Dacomitinib" in available_drugs else 0
        drug = st.selectbox(
            "Tyrosine kinase inhibitor",
            available_drugs,
            index=default_drug,
        )

        run = st.button(
            "Run model prediction",
            type="primary",
            use_container_width=True,
        )

    selected = data[
        (data["mutation"] == mutation) &
        (data["drug"] == drug)
    ].copy()

    has_observed = selected["auc_ratio_vs_wt"].notna().any()

    with right:
        if has_observed:
            st.markdown(
                """
                <div class="status-card observed-card">
                    <h4>✓ Observed + predicted pair</h4>
                    <p>This combination has a structurally generated feature row and an experimental AUC-ratio measurement in the project dataset.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-card extrapolation-card">
                    <h4>⚠ Model-only extrapolation</h4>
                    <p>The structural features were generated for this combination, but no experimental AUC-ratio label exists in the current resistance dataset.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        context = selected[
            [
                column
                for column in [
                    "position",
                    "distance_to_T790_A",
                    "distance_to_C797_A",
                    "distance_to_ATP_pocket_centroid_A",
                ]
                if column in selected.columns
            ]
        ].mean(numeric_only=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Position", f'{int(context.get("position", 0))}')
        m2.metric("Distance to T790", f'{context.get("distance_to_T790_A", np.nan):.1f} Å')
        m3.metric("Distance to C797", f'{context.get("distance_to_C797_A", np.nan):.1f} Å')

    if run:
        prediction_input = mean_feature_row(selected)
        prediction = predict_auc_ratio(prediction_input)
        predicted_auc = float(prediction["predicted_auc_ratio_vs_wt"].iloc[0])
        label, note = classify_auc(predicted_auc)

        result_cols = st.columns(4)
        result_cols[0].metric("Predicted AUC ratio vs WT", f"{predicted_auc:.3f}")
        result_cols[1].metric("Exploratory interpretation", label)
        result_cols[2].metric("Decision threshold", f"{RESISTANCE_THRESHOLD:.2f}")

        if has_observed:
            observed_auc = float(selected["auc_ratio_vs_wt"].mean())
            result_cols[3].metric(
                "Observed mean AUC ratio",
                f"{observed_auc:.3f}",
                delta=f"{predicted_auc - observed_auc:+.3f}",
                delta_color="off",
            )
        else:
            result_cols[3].metric("Experimental outcome", "Unavailable")

        st.caption(note)

        st.subheader("Structural interpretation profile")
        profile = feature_profile(selected, data)

        chart_col, table_col = st.columns([1.35, 1], gap="large")
        with chart_col:
            st.bar_chart(
                profile.set_index("Feature")[["Relative to matrix median"]],
                horizontal=True,
            )
            st.caption(
                "Positive values are above the median of the generated mutation–drug matrix; negative values are below it."
            )

        with table_col:
            st.dataframe(
                profile[["Feature", "Raw value"]],
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("What exactly happened when I clicked Predict?"):
            st.markdown(
                """
                1. The app selected the feature vector generated from the corresponding drug-bound EGFR mutant structure.
                2. It passed that vector through the fitted Random Forest pipeline.
                3. The displayed prediction was calculated by the model at click time.
                4. An experimental value is shown separately only when one exists.
                """
            )


elif page == "Combination coverage":
    st.header("Mutation–drug coverage")

    coverage = (
        data.assign(generated="✓")
        .pivot_table(
            index="mutation",
            columns="drug",
            values="generated",
            aggfunc="first",
            fill_value="—",
        )
        .reset_index()
    )
    st.dataframe(coverage, hide_index=True, use_container_width=True)

    st.subheader("Generated combinations by inhibitor")
    st.bar_chart(data.groupby("drug").size().rename("generated pairs"))

    if not unsupported.empty:
        with st.expander("View unsupported or failed combinations"):
            columns = [
                c for c in [
                    "mutation", "drug", "generation_status", "reason"
                ] if c in unsupported.columns
            ]
            st.dataframe(
                unsupported[columns],
                hide_index=True,
                use_container_width=True,
            )


elif page == "Model results":
    st.header("Model results")

    cols = st.columns(4)
    cols[0].metric("Grouped CV MAE", f'{metrics.get("rich_model_grouped_cv_MAE", 0.362):.3f}')
    cols[1].metric("Grouped CV RMSE", f'{metrics.get("rich_model_grouped_cv_RMSE", 0.483):.3f}')
    cols[2].metric("Grouped CV R²", f'{metrics.get("rich_model_grouped_cv_R2", -0.259):.3f}')
    cols[3].metric("Grouped CV Spearman", f'{metrics.get("rich_model_grouped_cv_Spearman", -0.093):.3f}')

    st.warning(
        "The grouped cross-validated R² is negative. Predictions for unseen mutations are therefore exploratory and currently unreliable."
    )

    matrix_predictions = predict_auc_ratio(data.copy())
    predicted_resistant_count = int(matrix_predictions["predicted_resistant"].sum())
    st.write(
        f"Across the current generated matrix, **{predicted_resistant_count} of {len(matrix_predictions)}** "
        "pairs are above the exploratory 1.50 resistant-like threshold. "
        "Examples include T790M–Dacomitinib and T790M–Afatinib."
    )

    st.markdown(
        """
        **What is still valuable here?**

        The project demonstrates an end-to-end structural ML workflow and tests whether richer biological representations improve generalization under severe data limitations. The final app does not hide that the current dataset remains the bottleneck.
        """
    )


elif page == "Pipeline":
    st.header("End-to-end pipeline")

    steps = [
        ("1 · Experimental assay extraction", "Parse mutation–drug dose-response outcomes from the Hayes workbooks."),
        ("2 · Drug-bound structure selection", "Use an experimentally determined EGFR–inhibitor structure for each represented drug."),
        ("3 · Mutant generation", "Introduce each supported mutation into each drug-bound structure."),
        ("4 · Structural feature calculation", "Calculate local geometry, contacts, packing, kinase landmarks, graph descriptors, DFI, and drug descriptors."),
        ("5 · Auxiliary learning", "Learn general EGFR mutation effects from 4,802 variants."),
        ("6 · Resistance modeling", "Predict AUC ratio with mutation-grouped cross-validation."),
        ("7 · Precompute all supported combinations", "Generate the full deployable mutation–drug feature matrix."),
        ("8 · Streamlit inference", "Load a generated feature row and calculate a model prediction at click time."),
    ]

    for title, description in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <strong>{title}</strong><br>
                <span style="color:#9AA7B4">{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Batch prediction":
    st.header("Batch prediction")
    st.write("Upload rows produced by the same structural feature pipeline.")

    example_path = PROJECT_ROOT / "data" / "example_prediction_rows.csv"
    if example_path.exists():
        st.download_button(
            "Download example input",
            data=example_path.read_bytes(),
            file_name="example_prediction_rows.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload rich-feature CSV", type=["csv"])

    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        try:
            output = predict_auc_ratio(frame)
            st.dataframe(output, use_container_width=True)
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
    st.error("Do not use this application to guide cancer treatment.")

    limitations = pd.DataFrame(
        {
            "Limitation": [
                "Small resistance dataset",
                "Limited inhibitor coverage",
                "Static mutant structures",
                "Geometry-based interaction proxies",
                "Weak unseen-mutation performance",
                "Precomputed structural matrix",
            ],
            "Implication": [
                "Only a small number of labeled mutation–drug rows train the resistance model.",
                "Only inhibitors with successful reference-structure processing are shown.",
                "Mutants were not subjected to validated ligand-aware relaxation.",
                "Contacts, clashes, and polar interactions are not binding free energies.",
                "Grouped cross-validated R² remains negative.",
                "Structural calculations occur offline once; the website performs fast model inference.",
            ],
        }
    )
    st.dataframe(limitations, hide_index=True, use_container_width=True)


st.markdown(
    """
    <div class="footer">
        Educational computational biology project · Structurally generated features · Exploratory machine-learning predictions
    </div>
    """,
    unsafe_allow_html=True,
)
