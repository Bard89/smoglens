# SmogLens

Predict the next hours of PM2.5 concentration in Japan better than baseline.

Baseline is the last available hour of data, so the models must beat simply repeating the most recent measurement. Source data is collected and preprocessed with [air-quality-pipeline](https://github.com/Bard89/air-quality-pipeline).

## Repository Layout

```
src/smoglens/     shared package: config, data access, features, inference, spatial, visualization
apps/V2_working/  Streamlit app with real ensemble inference (Shibuya)
apps/V1_demo/     early design mock (random forecasts, kept as design history)
notebooks/        frozen research log, v1_baseline -> v4_multiyear (see notebooks/README.md)
tests/            golden prediction tests, app smoke tests, feature unit tests, config tests
```

## Setup (macOS)

```bash
brew install pyenv pyenv-virtualenv

git clone git@github.com:Bard89/smoglens.git
cd smoglens

pyenv install 3.11.13
pyenv virtualenv 3.11.13 smoglens311
pip install -e ".[dev]"
```

## Data Setup

All data and trained models live in one external directory, resolved from the `SMOGLENS_DATA_PATH` environment variable (set it in `.env` or your shell):

```
smoglens-data/
├── pm25_enriched_2023_2025_v4_20250830_222050.csv   enriched dataset (~4.3 GB)
└── models/ensemble_v4/                              models_{1..6,12,24}h.pkl + metadata.pkl
```

The enriched CSV is produced by the [v4 enrichment pipeline](notebooks/v4_multiyear/03_data_processing/pm25_enrichment_pipeline_v4.ipynb); the models were trained by [train_ensemble_granular.py](notebooks/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/scripts/train_ensemble_granular.py).

## Run the App

```bash
streamlit run apps/V2_working/app.py
```

## Tests

```bash
pytest
```

The golden prediction tests pin the exact model outputs and guard every refactor; they skip on machines without the local dataset and models. Regenerate fixtures with `python tests/fixtures/generate.py` (only after an intentional behavior change).

## Changelog & Roadmap

Version history -> [CHANGELOG.md](CHANGELOG.md). What's coming next -> [ROADMAP.md](ROADMAP.md).

## Development Workflow

Use git flow -> [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
brew install git-flow

git flow init
git flow feature start my-feature

git push -u origin feature/my-feature

# Then create PR on GitHub to merge into develop
```

## Abbreviations

- **oAQ**: OpenAQ (PM2.5 air quality measurements)
- **oM**: OpenMeteo (weather data)
- **J**: JARTIC (traffic data)
