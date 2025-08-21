# SmogLens
Predict the next few hours of air-pollution in any place in Japan. 

The original pitch -> https://docs.google.com/presentation/d/1HZ1HlAWyo8_HaEVPiJ0_i2sVmD-E5akWW_KputF3Itw/edit?usp=sharing

The data are collected using -> https://github.com/Bard89/air-quality-pipeline

## Setup

```bash
# Install pyenv (macOS)
brew install pyenv

# Setup Python environment
pyenv install 3.12.9
pyenv virtualenv 3.12.9 smoglens
pyenv local smoglens

# Install dependencies
pip install -r requirements.txt
```

**Data Location:** `/Users/vojtech/Code/Bard89/Project-Data/data/processed/`

## Data Setup

The project uses a shared data directory. Set it up with:

```bash
# Create your local data directory
mkdir -p ~/smoglens-data

# Link it to the project
ln -s ~/smoglens-data data
```

## Development Workflow
This project uses Git Flow -> [CONTRIBUTING.md](CONTRIBUTING.md). 

### Quick Start with Git Flow

```bash
# Install git-flow
brew install git-flow     # macOS
apt-get install git-flow  # Linux
choco install gitflow-avh # Windows (via Chocolatey)

git flow init -d # initialize git-flow
git flow feature start my-feature # start a new feature

git push -u origin feature/my-feature  # push feature branch for the first time (-u sets upstream tracking)
git push                               # subsequent pushes

# Then create PR on GitHub to merge into develop
```
