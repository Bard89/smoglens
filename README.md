# SmogLens

Predict the next hours of PM2.5 concentration in Japan better than baseline.

Baseline is the last available hour of data, so the models must beat simply repeating the most recent measurement. Source data is collected and preprocessed with [air-quality-pipeline](https://github.com/Bard89/air-quality-pipeline).

## Presentation

SmogLens was presented at the Le Wagon Demo Day on 05/09/2025 in Tokyo (an earlier version of the project). Check out our — together with [Yuriya](https://github.com/YuriyaJP) — fantastic presentation skills -> [recording](https://youtu.be/4xmj3REIGe0?si=JVKjPSs4vlxfIjxk&t=1233)

The project was also featured on Le Wagon's best projects page -> [project page](https://projects.lewagon.com/projects/smoglens)

## Setup (macOS)

```bash
brew install pyenv pyenv-virtualenv

git clone git@github.com:Bard89/smoglens.git
cd smoglens

pyenv install 3.11.13
pyenv virtualenv 3.11.13 smoglens311
pip install -e ".[dev]"
```

## Data

Data and trained models live in one external directory, resolved from the `SMOGLENS_DATA_PATH` environment variable (set it in `.env` or your shell):

## Run

```bash
streamlit run apps/V2_working/app.py
```

## Changelog

Version history and planned work -> [CHANGELOG.md](CHANGELOG.md)

## Abbreviations

- **oAQ**: OpenAQ (PM2.5 air quality measurements)
- **oM**: OpenMeteo (weather data)
- **J**: JARTIC (traffic data)
