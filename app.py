from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_auc_ratio


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "egfr_structural_features_rich.csv"

st.set_page_config(
    page_title="EGFR TKI Resistance Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("EGFR TKI Resistance Explorer")
st.caption(
    "An exploratory structural machine-learning application for "
    "EGFR kinase-domain mutations."
)

st.warning(
    "Research and educational use only. This model was trained on a small "
    "dataset and is not clinically validated."
)

data = pd.read_csv(DATA_PATH)

tab_known, tab_upload, tab_about = st.tabs(
    ["Explore supported pairs", "Upload feature rows", "About the model"]
)

with tab_known:
    st.subheader("Explore a modeled mutation–drug pair")

    mutation = st.selectbox(
        "Mutation",
        sorted(data["mutation"].unique()),
    )

    available_drugs = sorted(
        data.loc[data["mutation"] == mutation, "drug"].unique()
    )
    drug = st.selectbox("Drug", available_drugs)

    selected = data[
        (data["mutation"] == mutation) &
        (data["drug"] == drug)
    ].copy()

    if st.button("Predict response", type="primary"):
        prediction = predict_auc_ratio(selected)

        predicted_auc = float(
            prediction["predicted_auc_ratio_vs_wt"].mean()
        )
        observed_auc = float(selected["auc_ratio_vs_wt"].mean())
        resistance_label = (
            "Resistant-like" if predicted_auc >= 1.5 else "Sensitive-like"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted AUC ratio vs WT", f"{predicted_auc:.3f}")
        col2.metric("Observed mean AUC ratio", f"{observed_auc:.3f}")
        col3.metric("Exploratory interpretation", resistance_label)

        st.markdown("#### Selected structural context")
        display_columns = [
            "distance_to_T790_A",
            "distance_to_C797_A",
            "distance_to_ATP_pocket_centroid_A",
            "delta_local_packing_atoms_6A",
            "delta_steric_clashes_2A",
            "auxiliary_predicted_mutation_effect",
            "mutation_site_DFI",
        ]
        available = [c for c in display_columns if c in selected.columns]
        st.dataframe(
            selected[available].mean(numeric_only=True).to_frame("value"),
            use_container_width=True,
        )

        st.caption(
            "The prediction is generated from precomputed structural features. "
            "Repeated experimental rows are averaged for display."
        )

with tab_upload:
    st.subheader("Predict from precomputed rich features")
    st.write(
        "Upload a CSV containing the same feature columns used by the model. "
        "This is intended for rows produced by the structural pipeline."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        try:
            output = predict_auc_ratio(frame)
            st.success("Prediction completed.")
            st.dataframe(output, use_container_width=True)

            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="egfr_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

with tab_about:
    st.subheader("Pipeline")
    st.markdown(
        """
        1. Extract normalized EGFR inhibitor-response assays.
        2. Generate supported kinase-domain mutant structures.
        3. Calculate local, kinase-wide, and drug-aware descriptors.
        4. Train an auxiliary mutation-effect model on 4,802 variants.
        5. Train the drug-response model with mutation-grouped validation.
        """
    )

    st.subheader("Current limitations")
    st.markdown(
        """
        - Only 50 kinase-domain mutation–drug rows were available.
        - The grouped cross-validated R² is negative, so predictions for unseen
          mutations are not yet reliable.
        - Mutant structures are unrelaxed starting models.
        - Interaction features are geometric proxies, not binding free energies.
        - The current deployed lookup includes afatinib, dacomitinib, and
          osimertinib pairs represented in the final supported structure table.
        """
    )
