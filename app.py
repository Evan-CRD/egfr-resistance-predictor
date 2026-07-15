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

st.set_page_config(page_title="EGFR TKI Response Model", page_icon="🧬", layout="wide")
st.markdown("""
<style>
:root{--cyan:#42d8e8;--teal:#34d5a7;--line:#25313d}.stApp{background:radial-gradient(circle at 12% 0%,rgba(66,216,232,.08),transparent 28%),linear-gradient(180deg,#030405,#070a0e)}.block-container{max-width:1260px;padding-top:1.4rem;padding-bottom:3rem}section[data-testid="stSidebar"]{background:#06090d;border-right:1px solid var(--line)}.hero{padding:2.2rem 2.4rem;border-radius:24px;border:1px solid rgba(66,216,232,.28);background:linear-gradient(135deg,#101923,#071015);box-shadow:0 22px 60px rgba(0,0,0,.45);margin-bottom:1.3rem}.hero .k{color:var(--cyan);font-size:.76rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hero h1{color:white;font-size:2.65rem;margin:.55rem 0}.hero p{color:#bdc8d2;font-size:1.08rem;line-height:1.6;max-width:900px}.result{border-left:4px solid var(--teal);padding:1rem 1.1rem;background:rgba(52,213,167,.08);border-radius:0 14px 14px 0;margin:.7rem 0}div[data-testid="stMetric"]{border:1px solid rgba(105,125,145,.28);border-radius:16px;padding:.9rem 1rem;background:linear-gradient(180deg,#101720,#080c11)}.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,var(--cyan),var(--teal));color:#021012;border:none;border-radius:12px;font-weight:800}h1,h2,h3,h4{color:white}p,li,label{color:#d7dfe7}a{color:var(--cyan)!important}.footer{text-align:center;color:#778490;font-size:.82rem;padding-top:2rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(OUT / "best_egfr_catboost_model.cbm"))
    return model

@st.cache_data
def load_all():
    metadata = json.loads((OUT / "best_model_metadata.json").read_text())
    data = pd.read_csv(OUT / "nature_egfr_long_format.csv")
    comparison = pd.read_csv(OUT / "feature_set_comparison.csv")
    oof = pd.read_csv(OUT / "best_model_oof_predictions.csv")
    per_drug = pd.read_csv(OUT / "best_model_per_drug_metrics.csv")
    importance = pd.read_csv(OUT / "best_model_feature_importance.csv")
    return metadata, data, comparison, oof, per_drug, importance

model = load_model()
metadata, data, comparison, oof, per_drug, importance = load_all()

st.markdown("""
<div class="hero"><div class="k">Machine learning · EGFR · TKI response</div>
<h1>EGFR Structure–Function Drug Response Model</h1>
<p>Predict relative inhibitor response from drug identity and published EGFR structure–function groups, and test whether a mechanistic structural representation outperforms the conventional exon-based representation.</p></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🧬 Navigation")
page = st.sidebar.radio("", ["Prediction", "Model Comparison", "Validation Results", "How It Works", "Pipeline", "Limitations"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"[View source on GitHub]({GITHUB_URL})")
st.sidebar.caption("Version 4.0 · ML-centered structure-group model")

if page == "Prediction":
    st.header("Predict a mutation's relative drug-response profile")
    c1, c2 = st.columns(2)
    mutation = c1.selectbox("EGFR mutation", sorted(data["mutation"].unique()))
    published_group = data.loc[data["mutation"] == mutation, "structure_group"].iloc[0]
    groups = metadata["known_structure_groups"]
    group = c2.selectbox("Published structure–function group", groups, index=groups.index(published_group))
    st.caption(f"Published group for {mutation}: **{published_group}**")
    if st.button("Predict response across all 18 TKIs", type="primary", use_container_width=True):
        frame = pd.DataFrame({"drug": metadata["known_drugs"], "structure_group": group})
        frame["predicted_log2_ratio"] = model.predict(frame[metadata["feature_columns"]])
        frame["predicted_IC50_fold_vs_WT"] = 2 ** frame["predicted_log2_ratio"]
        frame["interpretation"] = np.where(frame["predicted_log2_ratio"] > 0, "More resistant than WT", "More sensitive than WT")
        frame = frame.sort_values("predicted_log2_ratio", ascending=False)
        st.markdown('<div class="result"><strong>Interpretation:</strong> Positive values mean the mutant is predicted to require more drug than WT to reach the same IC50; negative values mean greater relative sensitivity.</div>', unsafe_allow_html=True)
        st.bar_chart(frame.set_index("drug")[["predicted_log2_ratio"]])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button("Download predicted profile", frame.to_csv(index=False), f"{mutation}_predicted_TKI_profile.csv", "text/csv")

elif page == "Model Comparison":
    st.header("Does structure–function grouping outperform exon location?")
    table = comparison.set_index("model")
    exon = table.loc["drug_plus_exon"]
    structure = table.loc["drug_plus_structure"]
    cols = st.columns(4)
    cols[0].metric("Exon model R²", f"{exon.R2:.3f}")
    cols[1].metric("Structure-group R²", f"{structure.R2:.3f}", delta=f"+{structure.R2-exon.R2:.3f}")
    cols[2].metric("Exon Spearman", f"{exon.Spearman:.3f}")
    cols[3].metric("Structure-group Spearman", f"{structure.Spearman:.3f}", delta=f"+{structure.Spearman-exon.Spearman:.3f}")
    st.success("The drug + structure–function group representation explains more held-out response variation and ranks mutation–drug responses more accurately than drug + exon.")
    st.bar_chart(comparison.set_index("model")[["R2", "Spearman"]])
    st.dataframe(comparison[["model", "n_features", "MAE", "RMSE", "R2", "Spearman"]], hide_index=True, use_container_width=True)
    st.subheader("Final-model feature importance")
    st.bar_chart(importance.set_index("feature")[["importance"]])
    st.caption("Importance describes predictive use by CatBoost; it does not prove causality.")

elif page == "Validation Results":
    st.header("Mutation-held-out validation")
    metrics = metadata["cross_validated_metrics"]
    cols = st.columns(4)
    cols[0].metric("R²", f"{metrics['R2']:.3f}")
    cols[1].metric("Spearman", f"{metrics['Spearman']:.3f}")
    cols[2].metric("MAE", f"{metrics['MAE']:.3f}")
    cols[3].metric("RMSE", f"{metrics['RMSE']:.3f}")
    st.info("All measurements for a mutation were held together. Each out-of-fold prediction was therefore made when that mutation was absent from training.")
    st.scatter_chart(oof[["response", "prediction"]], x="response", y="prediction")
    st.subheader("Performance by drug")
    st.dataframe(per_drug.sort_values("Spearman", ascending=False), hide_index=True, use_container_width=True)

elif page == "How It Works":
    st.header("What the model learns")
    st.markdown("""
### Target
The model predicts the median **log2(mutant IC50 / wild-type IC50)**. A value of 0 means similar IC50 to WT, +1 means about twice the WT IC50, and -1 means about half the WT IC50.

### Inputs
- **Drug identity:** one of 18 TKIs.
- **Structure–function group:** Classical-Like, PACC, Ex20ins-L, T790M-like-3S, or T790M-like-3R.

### Algorithm
CatBoost is a gradient-boosted decision-tree model designed to handle categorical variables and nonlinear interactions. The best held-out model used only drug and structure–function group because adding exon and simple mutation descriptors did not improve validation.
""")

elif page == "Pipeline":
    st.header("End-to-end machine-learning pipeline")
    steps = [
        ("1 · Load Nature Figure 2 data", "Read mutation, structure-group, exon, drug, and triplicate relative IC50 measurements."),
        ("2 · Aggregate replicates", "Use the median response for each mutation–drug pair."),
        ("3 · Define the target", "Predict log2(mutant IC50 / WT IC50)."),
        ("4 · Compare representations", "Compare drug-only, exon-based, structure-group, combined, and enhanced inputs."),
        ("5 · Group by mutation", "Hold out entire mutations during cross-validation to prevent leakage."),
        ("6 · Tune CatBoost", "Tune only the best representation using the same held-out-mutation design."),
        ("7 · Fit final model", "Retrain the selected model on all 1,380 rows."),
        ("8 · Deploy", "Predict a relative response profile across all 18 TKIs."),
    ]
    for title, description in steps:
        st.markdown(f'<div class="result"><strong>{title}</strong><br>{description}</div>', unsafe_allow_html=True)

elif page == "Limitations":
    st.header("Limitations and responsible interpretation")
    st.error("This is an experimental relative-response model, not a clinical treatment recommendation.")
    st.markdown("""
- The model uses published expert-defined structure–function labels; it does not infer a group from a raw PDB structure.
- The labels summarize mechanisms rather than individual continuous atomic measurements.
- R² around 0.43 is meaningful but not clinically sufficient.
- Predictions are limited to the 18 drugs and five groups in the training workbook.
- External experimental validation is still required.
- Feature importance is predictive, not causal.
""")

st.markdown('<div class="footer">EGFR Structure–Function Drug Response Model · Experimental research tool · Not clinical advice</div>', unsafe_allow_html=True)
