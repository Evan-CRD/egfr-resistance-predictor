# EGFR Structure–Function Drug Response Model

This application predicts relative TKI response using drug identity and published EGFR structure–function groups.

## Main result

Using 5-fold mutation-grouped cross-validation across 77 mutations, 18 drugs, and 1,380 mutation–drug rows:

- Drug + exon: R² = 0.224; Spearman = 0.482
- Drug + structure–function group: R² = 0.427; Spearman = 0.678

The main conclusion is that biologically informed structure–function groups predict held-out experimental TKI response better than exon location alone.

## Target

Median log2(mutant IC50 / WT IC50). Positive values indicate relative resistance; negative values indicate relative sensitivity.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Experimental research tool only; not clinical advice.
