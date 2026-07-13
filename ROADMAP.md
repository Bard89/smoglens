# Roadmap

The destination: a deployed, user-friendly app anyone in Japan can use to see PM2.5 forecasts up to ~12 hours ahead — built on historical data first, then fed by live data collection.

Completed work moves to [CHANGELOG.md](CHANGELOG.md).

## [0.3.0] - Training Pipeline in the Package

- Port enrichment, feature building, and training from the v4 notebooks into `smoglens` (`scripts/enrich.py`, `scripts/train.py`)
- One feature implementation shared by training and serving; canonical schema module
- Consolidate the four training scripts into one trainer (parallel, resumable, weight-optimized)

## [0.4.0] - Retrained Models

- Retrain the ensemble on the packaged pipeline -> `models/ensemble_v5` (1-6h, 12h, 24h)
- Train/serve feature quirks fixed by construction; evaluate traffic-feature contribution (drop or make optional for live serving)
- Regenerate golden fixtures for the new models

## [0.5.0] - Multi-Location App

- Location selector for any covered hexagon (not just Shibuya), 12h horizon in the UI
- Fix the h3 spatial-fallback bug; honest uncertainty bands from measured per-horizon MAE

## [0.6.0] - Live Data & Deployment

- `smoglens/sources/` windowed collectors (OpenAQ, OpenMeteo, JARTIC) against one canonical schema; backfill and live are the same code path (`scripts/collect.py`)
- Rewrite of the archived [air-quality-pipeline](https://github.com/Bard89/air-quality-pipeline) collection inside this repo
- Scheduled collection, live inference in the app, per-source data-freshness handling
- Model hosting and public deployment

## [0.7.0] - External Data Enrichment

- Additional data sources integration w/ new features
  - Explore and potentially add features from other already downloaded datasets.
  - Get more data from elsewhere. Need some outside of Japan pm2.5 measurements to be able to predict delayed transmission to Japan. Potentially data from Korea we have already or some low res satelite data paired with weather.
  - Yellow dust dataset from the Climate Data Store API (NetCDF zip, Copernicus Atmosphere Monitoring Service), including PM2.5, PM10, and wind from China (earlier exploration preserved in git history).
