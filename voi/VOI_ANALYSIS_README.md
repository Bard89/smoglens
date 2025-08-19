## Using pyenv

### 1. Create Python Environment
```bash
pyenv virtualenv 3.12.9 smoglens-analysis
pyenv local smoglens-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook 01_air_quality_openaq_eda.ipynb
```

## Alternative: Use Existing pyenv

```bash
pyenv activate your-env-name
pip install -r requirements.txt
jupyter notebook
```

## Data Location
`/Users/vojtech/Code/Bard89/Project-Data/data/processed/`