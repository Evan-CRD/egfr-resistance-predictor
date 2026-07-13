# 🧬 EGFR TKI Resistance Predictor

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://egfr-resistance-predictor-zezpmx3zjwyzy5tuajuzcd.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-exploratory-orange)](#limitations)

An end-to-end computational biology project integrating experimental EGFR inhibitor-response data, automated mutant structure generation, structural feature engineering, auxiliary mutation-effect learning, grouped machine-learning evaluation, and an interactive Streamlit application.

## Live application

**[Open the EGFR TKI Resistance Explorer](https://egfr-resistance-predictor-zezpmx3zjwyzy5tuajuzcd.streamlit.app)**

## Research question

> **Which structural changes in mutant EGFR are associated with resistance to EGFR tyrosine kinase inhibitors?**

The project focuses on lung-cancer-associated EGFR variants and four inhibitors represented in the source assay and structural workflow:

- erlotinib
- afatinib
- dacomitinib
- osimertinib

## Project architecture

```text
Experimental dose-response workbooks
        ↓
Mutation–drug AUC-ratio extraction
        ↓
Drug-bound EGFR kinase structures
        ↓
Automated mutant structure generation
        ↓
Local + kinase-wide + drug-aware features
        ↓
Auxiliary model trained on 4,802 EGFR variants
        ↓
Mutation-grouped model evaluation
        ↓
Streamlit research dashboard
```

## What makes this project substantial

- Parses real experimental dose-response measurements from nonstandard Excel layouts
- Generates supported drug-bound EGFR mutant structures automatically
- Computes local chemistry, contacts, packing, kinase-landmark distances, graph features, and DFI
- Adds drug physicochemical descriptors and geometry-based interaction changes
- Trains an auxiliary mutation-effect model on 4,802 variants
- Uses mutation-grouped validation to reduce leakage
- Packages the model into reusable Python code and a deployed web application

## Main modeling result

The richer representation lowered prediction error relative to a smaller local-feature representation. However, the rich model did **not** outperform a mean-response baseline for unseen mutation groups.

Current grouped cross-validation metrics:

| Metric | Rich model |
|---|---:|
| MAE | 0.362 |
| RMSE | 0.483 |
| R² | -0.259 |
| Spearman correlation | -0.093 |

This is an honest small-data result: the engineering pipeline works, while the current resistance dataset is insufficient for reliable prediction of unseen mutations.

## Application features

The Streamlit dashboard includes:

- mutation and inhibitor selection
- exploratory AUC-ratio prediction
- observed-vs-predicted comparison for represented pairs
- structural feature profile relative to the dataset
- model-results page
- pipeline overview
- batch prediction from rich-feature CSV files
- explicit responsible-use and limitation sections

## Repository structure

```text
egfr-resistance-predictor/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   └── predict.py
├── models/
│   ├── egfr_rich_regressor.joblib
│   └── feature_schema.json
├── data/
│   ├── egfr_structural_features_rich.csv
│   └── example_prediction_rows.csv
├── results/
│   └── model_metrics.json
└── notebooks/
    ├── EGFR_Colab_Step_by_Step.ipynb
    ├── 02_generate_EGFR_mutants.ipynb
    ├── 03_compute_EGFR_structural_features.ipynb
    ├── 03B_add_EGFR_kinase_wide_features.ipynb
    ├── 04_train_EGFR_resistance_model.ipynb
    ├── 05_train_EGFR_auxiliary_mutation_model.ipynb
    ├── 06_build_EGFR_rich_feature_table.ipynb
    └── 07_compare_EGFR_models.ipynb
```

## Run locally

```bash
git clone https://github.com/Evan-CRD/egfr-resistance-predictor.git
cd egfr-resistance-predictor

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
.venv\Scripts\activate
```

## Modeling decisions

### Continuous target

The primary model predicts the experimental AUC ratio relative to WT rather than discarding information through an immediate binary conversion.

### Grouped validation

All rows for the same mutation are kept in the same fold. This prevents the model from training on one drug measurement for a mutation and testing on another measurement for the same mutation.

### Auxiliary model

A separate model learns general mutation-effect scores from 4,802 EGFR kinase variants. Its prediction is then incorporated as an additional resistance-model feature rather than incorrectly merging two different biological labels.

## Limitations

- Only 50 supported kinase-domain mutation–drug rows were available.
- Several rows are repeated measurements rather than fully independent mutation–drug systems.
- Only substitutions represented in the chosen kinase-domain structures were modeled.
- Mutant structures were not subjected to validated ligand-aware minimization.
- Hydrogen-bond, clash, and contact features are geometric proxies.
- Grouped cross-validation remained below the mean-prediction baseline.
- Predictions are exploratory and must not guide patient treatment.

## Future development

- integrate independent EGFR mutation–drug response datasets
- collapse and model biological replicates more carefully
- add calibrated uncertainty estimates
- calculate validated stability and pocket-volume features
- evaluate per-drug and multitask models
- add external validation
- explore graph neural networks only after sufficient data are available

## Responsible-use statement

This software is intended for education and computational research. It is not a clinical decision-support system, medical device, or substitute for professional oncology guidance.
