
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.predict import predict_auc_ratio


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "egfr_all_supported_mutation_drug_features.csv"
UNSUPPORTED_PATH = ROOT / "data" / "unsupported_mutation_drug_pairs.csv"
METRICS_PATH = ROOT / "results" / "model_metrics.json"
GITHUB_URL = "https://github.com/Evan-CRD/egfr-resistance-predictor"
THRESHOLD = 1.50

st.set_page_config(
    page_title="EGFR Structural Mutation Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --bg:#030405; --panel:#0b1015; --panel2:#101720; --line:#24313e;
  --text:#f4f7fa; --muted:#9ba9b7; --cyan:#42d8e8; --teal:#35d4a8;
  --gold:#f3ca68; --red:#ff6f7e; --purple:#a68bfa;
}
.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(66,216,232,.085), transparent 27%),
    radial-gradient(circle at 94% 10%, rgba(166,139,250,.06), transparent 24%),
    linear-gradient(180deg,#030405,#070a0e);
  color:var(--text);
}
.block-container {max-width:1280px;padding-top:1.3rem;padding-bottom:3rem;}
section[data-testid="stSidebar"] {
  background:#06090d;border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] * {color:var(--text);}
.hero {
  padding:2.35rem 2.5rem;border-radius:25px;
  border:1px solid rgba(66,216,232,.27);
  background:linear-gradient(135deg,rgba(14,23,32,.98),rgba(7,16,22,.98));
  box-shadow:0 24px 65px rgba(0,0,0,.45);margin-bottom:1.35rem;
}
.hero .kicker {color:var(--cyan);font-size:.76rem;font-weight:800;
  letter-spacing:.15em;text-transform:uppercase;margin-bottom:.7rem;}
.hero h1 {color:#fff;font-size:2.65rem;line-height:1.08;margin:0;letter-spacing:-.03em;}
.hero p {color:#bac6d1;font-size:1.08rem;line-height:1.62;max-width:930px;margin:.9rem 0 0;}
.flow {
  display:flex;align-items:center;justify-content:center;gap:.45rem;flex-wrap:wrap;
  padding:1.1rem;border:1px solid var(--line);border-radius:17px;
  background:rgba(11,16,21,.9);margin:1rem 0 1.5rem;
}
.flow .node {padding:.6rem .8rem;border:1px solid rgba(66,216,232,.24);
  border-radius:10px;background:#0d151d;color:#eaf9fb;font-size:.88rem;font-weight:650;}
.flow .arrow {color:var(--cyan);font-weight:900;}
.card {
  border:1px solid rgba(100,120,140,.27);border-radius:17px;padding:1.15rem 1.2rem;
  background:linear-gradient(180deg,rgba(16,23,32,.97),rgba(9,14,19,.97));
  box-shadow:0 12px 30px rgba(0,0,0,.22);
}
.card h4 {margin:0 0 .45rem;color:#fff}.card p {margin:0;color:#aeb9c5;line-height:1.5}
.section-label {color:var(--cyan);font-size:.75rem;text-transform:uppercase;
  letter-spacing:.13em;font-weight:800;margin-bottom:.35rem}
.mechanism {
  border-left:4px solid var(--teal);border-radius:0 15px 15px 0;
  background:linear-gradient(90deg,rgba(53,212,168,.11),rgba(11,16,21,.85));
  padding:1rem 1.1rem;margin:.7rem 0;
}
.warning-box {
  border:1px solid rgba(243,202,104,.42);border-radius:15px;
  background:rgba(56,43,11,.42);padding:1rem 1.1rem;
}
.observed {border-color:rgba(53,212,168,.43);background:rgba(9,35,29,.5)}
.extrapolation {border-color:rgba(243,202,104,.45);background:rgba(48,36,8,.5)}
div[data-testid="stMetric"] {
  border:1px solid rgba(100,120,140,.27);border-radius:16px;padding:.9rem 1rem;
  background:linear-gradient(180deg,rgba(16,23,32,.98),rgba(8,12,17,.98));
}
div[data-testid="stMetricLabel"] {color:#aab6c2;font-weight:650}
div[data-testid="stMetricValue"] {color:#fff}
.stButton>button,.stDownloadButton>button {
  background:linear-gradient(135deg,var(--cyan),var(--teal));color:#021012;
  border:none;border-radius:12px;font-weight:800;padding:.65rem 1rem;
}
div[data-baseweb="select"]>div,div[data-testid="stFileUploaderDropzone"] {
  background:#0b1015;border-color:var(--line);color:#fff;
}
h1,h2,h3,h4 {color:#fff} p,li,label {color:#d6dee6}
a {color:var(--cyan)!important}
.footer {text-align:center;color:#73808d;font-size:.82rem;padding-top:2rem}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


@st.cache_data
def load_unsupported() -> pd.DataFrame:
    return pd.read_csv(UNSUPPORTED_PATH) if UNSUPPORTED_PATH.exists() else pd.DataFrame()


def selection_controls(data: pd.DataFrame, key: str) -> tuple[str, str, pd.DataFrame]:
    c1, c2 = st.columns(2)
    mutation = c1.selectbox(
        "EGFR mutation",
        sorted(data["mutation"].unique()),
        key=f"{key}_mutation",
    )
    drugs = sorted(data.loc[data["mutation"] == mutation, "drug"].unique())
    drug = c2.selectbox(
        "Inhibitor",
        drugs,
        key=f"{key}_drug",
    )
    selected = data[(data["mutation"] == mutation) & (data["drug"] == drug)].copy()
    return mutation, drug, selected


def mean_row(frame: pd.DataFrame) -> pd.DataFrame:
    row = frame.iloc[[0]].copy()
    for col in frame.select_dtypes(include=[np.number]).columns:
        row.at[row.index[0], col] = pd.to_numeric(frame[col], errors="coerce").mean()
    return row


def value(frame: pd.DataFrame, col: str, default=np.nan) -> float:
    if col not in frame.columns:
        return default
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def signed(value_: float, digits: int = 1) -> str:
    if not np.isfinite(value_):
        return "Unavailable"
    return f"{value_:+.{digits}f}"


def structural_narrative(row: pd.DataFrame) -> list[str]:
    statements = []
    vol = value(row, "delta_sidechain_volume_A3")
    packing = value(row, "delta_local_packing_atoms_6A")
    clashes = value(row, "delta_steric_clashes_2A")
    atp = value(row, "distance_to_ATP_pocket_centroid_A")
    dfi = value(row, "mutation_site_DFI")

    if np.isfinite(vol):
        if vol > 30:
            statements.append("The substitution introduces a substantially larger side chain.")
        elif vol < -30:
            statements.append("The substitution creates a substantially smaller side chain and potential local void space.")
        else:
            statements.append("The substitution causes only a modest side-chain volume change.")
    if np.isfinite(atp):
        if atp < 10:
            statements.append("The mutation lies close to the ATP-binding pocket and may directly affect pocket geometry.")
        elif atp < 20:
            statements.append("The mutation is at an intermediate distance from the ATP pocket.")
        else:
            statements.append("The mutation is structurally distant from the ATP-pocket centroid.")
    if np.isfinite(packing):
        if packing > 5:
            statements.append("Local atomic packing becomes tighter in the modeled mutant.")
        elif packing < -5:
            statements.append("Local atomic packing becomes looser in the modeled mutant.")
    if np.isfinite(clashes) and clashes > 0:
        statements.append("The modeled mutation increases short-range steric overlap with the inhibitor environment.")
    if np.isfinite(dfi):
        statements.append(
            "The mutation-site DFI supplies a reference-state flexibility context rather than a mutant-specific dynamics simulation."
        )
    return statements


def interaction_narrative(row: pd.DataFrame) -> list[str]:
    statements = []
    contacts = value(row, "delta_protein_drug_atom_contacts_4p5A")
    side_contacts = value(row, "delta_sidechain_drug_contacts_4p5A")
    polar_lost = value(row, "putative_polar_contacts_lost")
    pocket_lost = value(row, "pocket_residues_lost_4p5A")
    clashes = value(row, "delta_steric_clashes_2A")

    if np.isfinite(contacts):
        if contacts < 0:
            statements.append(f"The mutant loses approximately {abs(contacts):.0f} protein–drug atom contacts within 4.5 Å.")
        elif contacts > 0:
            statements.append(f"The mutant gains approximately {contacts:.0f} protein–drug atom contacts within 4.5 Å.")
        else:
            statements.append("The total protein–drug atom-contact count is unchanged in this static model.")
    if np.isfinite(side_contacts) and side_contacts != 0:
        direction = "gains" if side_contacts > 0 else "loses"
        statements.append(f"The mutated side chain itself {direction} {abs(side_contacts):.0f} close drug contacts.")
    if np.isfinite(polar_lost) and polar_lost > 0:
        statements.append(f"{polar_lost:.0f} putative polar protein–drug contacts are lost.")
    if np.isfinite(pocket_lost) and pocket_lost > 0:
        statements.append(f"{pocket_lost:.0f} drug-contacting pocket residues are no longer counted within the cutoff.")
    if np.isfinite(clashes) and clashes > 0:
        statements.append("Increased steric overlap suggests that structural relaxation could be required.")
    return statements


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)


data = load_data()
metrics = load_metrics()
unsupported = load_unsupported()

st.markdown(
    """
<div class="hero">
  <div class="kicker">Computational structural biology platform</div>
  <h1>EGFR Structural Mutation Analysis</h1>
  <p>
    Quantify how kinase-domain mutations alter local structure, functional-site geometry,
    and inhibitor interactions; generate a mechanistic interpretation; and use resistance
    prediction only as a downstream exploratory module.
  </p>
</div>
<div class="flow">
  <span class="node">Drug-bound structure</span><span class="arrow">→</span>
  <span class="node">Computational mutation</span><span class="arrow">→</span>
  <span class="node">Structural analysis</span><span class="arrow">→</span>
  <span class="node">Mechanistic interpretation</span><span class="arrow">→</span>
  <span class="node">Optional prediction</span>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🧬 Platform")
page = st.sidebar.radio(
    "Navigation",
    [
        "🧬 Structural Analysis",
        "⚛ Drug Interaction Analysis",
        "🧠 Mechanistic Interpretation",
        "📊 Resistance Prediction",
        "📈 Model Performance",
        "🔬 Computational Workflow",
        "📚 Methods & Limitations",
        "🗂 Combination Coverage",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"[Source code on GitHub]({GITHUB_URL})")
st.sidebar.caption("Version 3.0 · structure-first interface")


if page == "🧬 Structural Analysis":
    st.header("Structural Analysis")
    st.write(
        "This page characterizes the mutation and its three-dimensional context before making any resistance prediction."
    )
    mutation, drug, selected = selection_controls(data, "struct")

    st.markdown("### Mutation and functional-site geometry")
    metric_row(
        [
            ("Substitution", mutation),
            ("Residue position", f'{int(value(selected,"position",0))}'),
            ("Distance to ATP pocket", f'{value(selected,"distance_to_ATP_pocket_centroid_A"):.1f} Å'),
            ("Distance to T790", f'{value(selected,"distance_to_T790_A"):.1f} Å'),
            ("Distance to C797", f'{value(selected,"distance_to_C797_A"):.1f} Å'),
        ]
    )

    st.markdown("### Local structural changes")
    metric_row(
        [
            ("Side-chain volume Δ", f'{signed(value(selected,"delta_sidechain_volume_A3"))} Å³'),
            ("Local packing Δ", signed(value(selected,"delta_local_packing_atoms_6A"),0)),
            ("Local backbone RMSD", f'{value(selected,"local_backbone_rmsd_A"):.3f} Å'),
            ("Steric-clash Δ", signed(value(selected,"delta_steric_clashes_2A"),0)),
        ]
    )

    st.markdown("### Dynamics and structural-network context")
    metric_row(
        [
            ("Mutation-site DFI", f'{value(selected,"mutation_site_DFI"):.4f}'),
            ("DFI percentile", f'{value(selected,"mutation_site_DFI_percentile"):.3f}'),
            ("Contact-graph degree", f'{value(selected,"contact_graph_degree"):.0f}'),
            ("Betweenness centrality", f'{value(selected,"betweenness_centrality"):.4f}'),
        ]
    )

    st.markdown("### Structural reading")
    for sentence in structural_narrative(selected):
        st.markdown(f'<div class="mechanism">{sentence}</div>', unsafe_allow_html=True)

    with st.expander("Advanced structural feature vector"):
        structural_cols = [
            c for c in selected.columns
            if any(token in c for token in [
                "distance_to_", "delta_sidechain", "delta_local",
                "backbone_rmsd", "bfactor", "centrality",
                "clustering", "graph_degree", "DFI", "rigidity"
            ])
        ]
        st.dataframe(
            mean_row(selected)[structural_cols].T.rename(columns={mean_row(selected).index[0]:"value"}),
            use_container_width=True,
        )


elif page == "⚛ Drug Interaction Analysis":
    st.header("Drug Interaction Analysis")
    st.write(
        "This page compares the wild-type and modeled mutant interaction environments using geometric contact cutoffs."
    )
    mutation, drug, selected = selection_controls(data, "drug")

    st.markdown("### Protein–drug interaction changes")
    metric_row(
        [
            ("All atom-contact Δ", signed(value(selected,"delta_protein_drug_atom_contacts_4p5A"),0)),
            ("Mutated side-chain contact Δ", signed(value(selected,"delta_sidechain_drug_contacts_4p5A"),0)),
            ("Pocket residues lost", f'{value(selected,"pocket_residues_lost_4p5A"):.0f}'),
            ("Pocket residues gained", f'{value(selected,"pocket_residues_gained_4p5A"):.0f}'),
            ("Polar contacts lost", f'{value(selected,"putative_polar_contacts_lost"):.0f}'),
        ]
    )

    st.markdown("### Wild type versus mutant")
    comparison = pd.DataFrame(
        {
            "Interaction descriptor": [
                "Protein–drug atom contacts (≤4.5 Å)",
                "Side-chain–drug contacts (≤4.5 Å)",
                "Pocket residue count (≤4.5 Å)",
                "Putative polar contacts (≤3.5 Å)",
                "Steric clashes (≤2.0 Å)",
                "Local packing atoms (≤6.0 Å)",
            ],
            "Wild type": [
                value(selected,"wt_protein_drug_atom_contacts_4p5A"),
                value(selected,"wt_sidechain_drug_contacts_4p5A"),
                value(selected,"wt_pocket_residue_count_4p5A"),
                value(selected,"wt_putative_polar_contacts_3p5A"),
                value(selected,"wt_steric_clashes_2A"),
                value(selected,"wt_local_packing_atoms_6A"),
            ],
            "Mutant": [
                value(selected,"mut_protein_drug_atom_contacts_4p5A"),
                value(selected,"mut_sidechain_drug_contacts_4p5A"),
                value(selected,"mut_pocket_residue_count_4p5A"),
                value(selected,"mut_putative_polar_contacts_3p5A"),
                value(selected,"mut_steric_clashes_2A"),
                value(selected,"mut_local_packing_atoms_6A"),
            ],
        }
    )
    comparison["Change"] = comparison["Mutant"] - comparison["Wild type"]
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("### Interaction interpretation")
    notes = interaction_narrative(selected)
    if not notes:
        notes = ["No large interaction-count changes are detected under the current geometric cutoffs."]
    for sentence in notes:
        st.markdown(f'<div class="mechanism">{sentence}</div>', unsafe_allow_html=True)

    st.caption(
        "These are geometric proxies from static modeled structures; they are not measured binding energies."
    )


elif page == "🧠 Mechanistic Interpretation":
    st.header("Mechanistic Interpretation")
    st.write(
        "This page combines the structural and interaction descriptors into a cautious biological narrative."
    )
    mutation, drug, selected = selection_controls(data, "mech")

    all_notes = structural_narrative(selected) + interaction_narrative(selected)
    for i, note in enumerate(all_notes, start=1):
        st.markdown(
            f'<div class="mechanism"><strong>{i}.</strong> {note}</div>',
            unsafe_allow_html=True,
        )

    atp = value(selected,"distance_to_ATP_pocket_centroid_A")
    contacts = value(selected,"delta_protein_drug_atom_contacts_4p5A")
    clashes = value(selected,"delta_steric_clashes_2A")
    vol = value(selected,"delta_sidechain_volume_A3")

    if atp < 12 and (contacts < 0 or clashes > 0 or abs(vol) > 30):
        summary = (
            "The modeled mutation combines close ATP-pocket proximity with a meaningful "
            "local or interaction perturbation, making a direct effect on inhibitor binding plausible."
        )
    elif atp >= 12 and (contacts != 0 or clashes != 0):
        summary = (
            "The mutation is not immediately adjacent to the ATP-pocket centroid, but the static model "
            "still detects changes in the inhibitor interaction environment; an indirect or allosteric mechanism is possible."
        )
    else:
        summary = (
            "The static structural model shows only modest perturbations, so any functional effect may depend "
            "on conformational dynamics, ATP competition, kinetics, or mechanisms not captured by these descriptors."
        )

    st.markdown("### Integrated summary")
    st.markdown(f'<div class="card"><p>{summary}</p></div>', unsafe_allow_html=True)
    st.warning(
        "This interpretation is rule-based and descriptive. It is not a validated causal mechanism."
    )


elif page == "📊 Resistance Prediction":
    st.header("Resistance Prediction")
    st.write(
        "Prediction is intentionally downstream of the structural analysis. The deployed Random Forest uses the full generated feature vector."
    )
    mutation, drug, selected = selection_controls(data, "pred")
    observed = selected["auc_ratio_vs_wt"].notna().any()

    if observed:
        st.markdown(
            '<div class="card observed"><h4>Experimental measurement available</h4>'
            '<p>The final deployed model was fitted on all labeled data, so this is not an independent blind test.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="card extrapolation"><h4>Model-only extrapolation</h4>'
            '<p>A structural feature vector exists, but no experimental AUC label is available for this pair.</p></div>',
            unsafe_allow_html=True,
        )

    if st.button("Calculate exploratory prediction", type="primary", use_container_width=True):
        pred = float(predict_auc_ratio(mean_row(selected))["predicted_auc_ratio_vs_wt"].iloc[0])
        label = "Resistant-like" if pred >= THRESHOLD else "Sensitive-like"
        items = [
            ("Predicted AUC ratio vs WT", f"{pred:.3f}"),
            ("Exploratory class", label),
            ("Threshold", f"{THRESHOLD:.2f}"),
        ]
        if observed:
            obs = float(pd.to_numeric(selected["auc_ratio_vs_wt"], errors="coerce").mean())
            items.append(("Observed mean AUC", f"{obs:.3f}"))
        else:
            items.append(("Observed mean AUC", "Unavailable"))
        metric_row(items)

        st.info(
            "The prediction is calculated by the model at click time from the precomputed structural feature vector; "
            "it is not a stored experimental answer."
        )


elif page == "📈 Model Performance":
    st.header("Model Performance")
    metric_row(
        [
            ("Grouped CV MAE", f'{metrics.get("rich_model_grouped_cv_MAE",.362):.3f}'),
            ("Grouped CV RMSE", f'{metrics.get("rich_model_grouped_cv_RMSE",.483):.3f}'),
            ("Grouped CV R²", f'{metrics.get("rich_model_grouped_cv_R2",-0.259):.3f}'),
            ("Grouped CV Spearman", f'{metrics.get("rich_model_grouped_cv_Spearman",-0.093):.3f}'),
        ]
    )
    st.error(
        "The negative grouped cross-validated R² means the model does not yet predict completely unseen mutations "
        "better than a training-fold mean baseline."
    )
    st.markdown(
        """
### Why retain the prediction module?

- It tests whether the engineered structural descriptors contain generalizable response information.
- It exposes the current small-data bottleneck honestly.
- It provides a deployable framework that can be retrained when additional labeled mutation–drug data become available.
- It is one downstream application of the structural-analysis platform, not the platform's sole scientific contribution.
"""
    )


elif page == "🔬 Computational Workflow":
    st.header("Computational Workflow")
    steps = [
        ("1 · Experimental assay extraction", "Parse dose–response measurements and calculate AUC ratios relative to WT."),
        ("2 · Drug-bound structure selection", "Use experimentally determined EGFR kinase–inhibitor PDB structures."),
        ("3 · Computational mutagenesis", "Replace the WT amino acid and construct the mutant side chain with standard geometry."),
        ("4 · Structural feature engineering", "Calculate functional-site distances, packing, clashes, contacts, graph metrics, DFI, and drug descriptors."),
        ("5 · Auxiliary mutation model", "Learn a general EGFR mutation-effect score from approximately 4,800 variants."),
        ("6 · Resistance model", "Train a Random Forest regressor on labeled mutation–drug AUC ratios."),
        ("7 · Mutation-grouped validation", "Keep all rows from the same mutation together to test generalization to unseen mutations."),
        ("8 · Deployment", "Precompute supported structural feature vectors and perform fast model inference in Streamlit."),
    ]
    for title, desc in steps:
        st.markdown(
            f'<div class="mechanism"><strong>{title}</strong><br><span style="color:#aeb9c5">{desc}</span></div>',
            unsafe_allow_html=True,
        )


elif page == "📚 Methods & Limitations":
    st.header("Methods & Limitations")
    with st.expander("What is the experimental AUC ratio?", expanded=True):
        st.write(
            "AUC summarizes the full dose–response curve across drug concentrations. "
            "The project normalizes mutant AUC to the corresponding WT response, so larger ratios generally indicate greater survival and resistance-like behavior."
        )
    with st.expander("How is a mutant structure generated?"):
        st.write(
            "The pipeline starts from an experimentally determined drug-bound EGFR kinase structure, replaces the specified WT residue, "
            "constructs the new side chain using standard amino-acid geometry, and adds missing atoms. It does not predict a new global fold."
        )
    with st.expander("What does DFI mean here?"):
        st.write(
            "DFI is a reference-state flexibility descriptor for the residue position. It provides dynamics context, but the pipeline does not recalculate mutant-specific DFI."
        )
    with st.expander("Why Random Forest?"):
        st.write(
            "Random Forest can represent nonlinear feature interactions, works with mixed scales after preprocessing, and is more appropriate than a deep network for a very small labeled dataset."
        )
    st.markdown("### Major limitations")
    limitations = pd.DataFrame(
        {
            "Limitation": [
                "Small labeled resistance dataset",
                "Static mutant structures",
                "No validated ligand-aware relaxation",
                "Geometry-based interaction proxies",
                "Reference-state DFI",
                "Restricted structural coverage",
                "Negative unseen-mutation R²",
            ],
            "Consequence": [
                "The model cannot learn a robust high-dimensional response function.",
                "Large conformational changes may be missed.",
                "Side-chain and pocket rearrangements may be underestimated.",
                "Contacts and clashes are not binding free energies.",
                "Mutation-specific dynamic changes are not captured.",
                "Only supported kinase-domain substitutions and drugs are displayed.",
                "The current predictor is exploratory rather than reliable for novel mutations.",
            ],
        }
    )
    st.dataframe(limitations, hide_index=True, use_container_width=True)


elif page == "🗂 Combination Coverage":
    st.header("Combination Coverage")
    coverage = (
        data.assign(status="Generated")
        .pivot_table(index="mutation", columns="drug", values="status", aggfunc="first", fill_value="—")
        .reset_index()
    )
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    if not unsupported.empty:
        with st.expander("Unsupported and failed pairs"):
            cols = [c for c in ["mutation","drug","generation_status","reason"] if c in unsupported.columns]
            st.dataframe(unsupported[cols], hide_index=True, use_container_width=True)


st.markdown(
    '<div class="footer">EGFR Structural Mutation Analysis Platform · Educational and research use only · Not clinically validated</div>',
    unsafe_allow_html=True,
)
