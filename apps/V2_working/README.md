# SmogLens V2 - PM2.5 Prediction App

Real-time PM2.5 predictions for outdoor activities in Shibuya, Tokyo. Six-hour forecasts from the LightGBM + XGBoost + CatBoost ensemble, with activity-based safety thresholds and K-NN spatial imputation.

## Run Locally

From the repository root:

```bash
pip install -e ".[app]"
streamlit run apps/V2_working/app.py
```

Requires the shared data directory (see the root [README](../../README.md)): the enriched dataset and the trained models resolved via `SMOGLENS_DATA_PATH`. On first start the app builds a local parquet cache in `data/`.

## Deployment Status

Not currently deployed. The `STREAMLIT_CLOUD` code path (bundled `data/shibuya_2024.csv.gz` + a repo-local `models/` directory) is kept for a future deployment pass, which needs a model-hosting solution since the model files are too large to commit.
