# SmogLens
Predict the next few hours of air-pollution in any place in Japan. 

The original pitch -> https://docs.google.com/presentation/d/1HZ1HlAWyo8_HaEVPiJ0_i2sVmD-E5akWW_KputF3Itw/edit?usp=sharing

The data are collected using -> https://github.com/Bard89/air-quality-pipeline

## Development Workflow
This project uses Git Flow -> [CONTRIBUTING.md](CONTRIBUTING.md). 

### Quick Start

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
