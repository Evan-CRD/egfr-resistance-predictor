# EGFR Structure–Function Drug Response Website

This is a complete Streamlit deployment package.

## Deploy

1. Upload every file and folder in this package to the root of the GitHub repository.
2. Keep the `strong_model_outputs` directory and all of its files.
3. Set the Streamlit main file path to `app.py`.
4. Reboot the app after the commit.

The mutation selector now automatically assigns and locks the published structure–function group.

## Validated model performance

- R²: 0.427
- Spearman: 0.678
- MAE: 0.702
- RMSE: 0.930
- Best feature set: `drug_plus_structure`

## Important limitation

Because the final selected model uses only drug identity and structural group, mutations in the same group receive the same predicted response profile.

This is an experimental research tool and not clinical advice.
