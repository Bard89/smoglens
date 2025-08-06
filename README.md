# smoglens
Predict the air pollution in any place in Japan

## Development Workflow

This project uses Git Flow. More in [CONTRIBUTING.md](CONTRIBUTING.md). 

### Quick Start

```bash
# Install git-flow
brew install git-flow  # macOS
apt-get install git-flow  # linux

git flow init -d # initialize git-flow
git flow feature start my-feature # start a new feature

git push -u origin feature/my-feature  # push feature branch for the first time (-u sets upstream tracking)
git push                               # subsequent pushes

# Then create PR on GitHub to merge into develop
```
