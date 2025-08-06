# Contributing Guidelines

## Git Flow Workflow

This project follows Git Flow. Please adhere to these guidelines:

### Branch Structure
- `main` - Production-ready code only
- `develop` - Integration branch for features
- `feature/*` - New features and enhancements
- `release/*` - Release preparation
- `hotfix/*` - Emergency production fixes

### Workflow Commands

#### Starting a New Feature
```bash
git flow feature start <feature-name>
# Work on your feature
git add .
git commit -m "Description of changes"
git push -u origin feature/<feature-name>
# Create PR to develop branch on GitHub
```

#### Creating a Release
```bash
git flow release start <version>
git flow release finish <version> 
git push origin main develop --tags
```

#### Emergency Hotfix
```bash
git flow hotfix start <version>
git flow hotfix finish <version>
git push origin main develop --tags
```

### Pull Request Process
1. Create feature branch from `develop`
2. Make your changes
3. Push to GitHub
4. Open PR against `develop`
5. Request review
6. Merge after approval