# SmogLens

The goal is to predict the next few hours of PM2.5 particle concentration in any place in Japan.

The original pitch -> https://docs.google.com/presentation/d/1HZ1HlAWyo8_HaEVPiJ0_i2sVmD-E5akWW_KputF3Itw/edit?usp=sharing

The data are collected using GH project -> https://github.com/Bard89/air-quality-pipeline

## Setup (macOS)

```bash
brew install pyenv

pyenv install 3.10.6
pyenv virtualenv 3.10.6 smoglens
pyenv local smoglens

pip install -r requirements.txt
```

## Data Setup

The project uses a shared data directory. Set it up with:

```bash
mkdir -p ~/smoglens-data
ln -s ~/smoglens-data data # Link it to the project
```

## Changelog

For detailed version history and updates -> [CHANGELOG.md](CHANGELOG.md)

## Development Workflow
Use git flow -> [CONTRIBUTING.md](CONTRIBUTING.md). 

### Quick Start with Git Flow (macOS)

```bash
brew install git-flow

git flow init -d
git flow feature start my-feature

git push -u origin feature/my-feature  # push feature branch for the first time (-u sets upstream tracking)
git push

# Then create PR on GitHub to merge into develop
```
