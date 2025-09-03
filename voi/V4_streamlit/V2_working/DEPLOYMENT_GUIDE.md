# SmogLens V2 Deployment Guide

## Challenge: Large Data & Models
- **Data**: 4.3GB CSV file (9.4M records)
- **Models**: ~200MB total (6 model files)
- **Streamlit Cloud Limits**: 1GB RAM, file size restrictions

## Deployment Strategy

### Step 1: Create Sample Data (Required)
```bash
python -c "
import pandas as pd
import h3

df = pd.read_csv('/Users/vojtech/Code/Bard89/smoglens-02/data/pm25_enriched_2023_2025_v4_20250830_222050.csv', 
                 parse_dates=['timestamp'])

shibuya_hex = '872e44d04ffffff'
nearby_hexes = list(h3.grid_disk(shibuya_hex, 2))

shibuya_data = df[df['hex7_id'].isin(nearby_hexes)]
recent_data = shibuya_data[shibuya_data['timestamp'] >= '2024-01-01']

recent_data.to_csv('data/shibuya_2024.csv.gz', index=False, compression='gzip')
print(f'Created sample: {len(recent_data):,} records, {len(recent_data.hex7_id.unique())} hexagons')
"
```

### Step 2: Copy Models to Repository
```bash
cd /Users/vojtech/Code/Bard89/smoglens/voi/V4_streamlit/V2_working
rm models  # Remove symlink if exists
mkdir models
cp /Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/models_[1-6]h.pkl models/
cp /Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/metadata.pkl models/
```

### Step 3: Set Up Git LFS for Models
```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
git add models/
git commit -m "Add model files with Git LFS"
```

### Step 4: Update config.py for Cloud
```python
import os

if os.getenv('STREAMLIT_CLOUD'):
    DATA_PATH = Path(__file__).parent / 'data' / 'shibuya_2024.csv.gz'
    MODEL_DIR = Path(__file__).parent / 'models'
else:
    DATA_PATH = Path('/Users/vojtech/Code/Bard89/smoglens-02/data/pm25_enriched_2023_2025_v4_20250830_222050.csv')
    MODEL_DIR = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained')
```

### Step 5: Optimize Memory Usage
Update `utils/data_processor.py`:
```python
@st.cache_resource
def load_data(self):
    if os.getenv('STREAMLIT_CLOUD'):
        # Load only necessary columns for cloud
        cols = ['timestamp', 'hex7_id', 'pm25_ugm3_mean', 
                'temperature_c_mean', 'humidity_pct_mean', 
                'pressure_hpa_mean', 'avg_traffic_volume']
        df = pd.read_csv(config.DATA_PATH, usecols=cols, 
                        parse_dates=['timestamp'])
        # Downsample to reduce memory
        df = df[df['timestamp'].dt.minute == 0]  # Keep only hourly data
    else:
        df = pd.read_csv(config.DATA_PATH, parse_dates=['timestamp'])
    return df
```

### Step 6: Commit and Push
```bash
git add -f data/shibuya_2024.csv.gz
git add runtime.txt requirements.txt
git commit -m "Prepare for Streamlit Cloud deployment"
git push
```

### Step 7: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Select your repository and branch
3. Set app path: `voi/V4_streamlit/V2_working/app.py`
4. In Advanced Settings:
   - Python version: Select 3.11
   - Secrets: Add `STREAMLIT_CLOUD=true`

## Alternative: Use Parquet Format
For better performance, convert data to Parquet:
```python
df.to_parquet('data/shibuya_2024.parquet', compression='snappy')
# 4.3GB CSV → ~500MB Parquet
```

## Monitoring
- Check memory usage in Streamlit Cloud dashboard
- If OOM errors occur, further reduce data:
  - Keep only last 6 months
  - Sample to every 3 hours
  - Reduce hexagon coverage radius

## Troubleshooting

### "Memory limit exceeded"
- Reduce data timeframe to 3 months
- Use `del df` and `gc.collect()` after processing
- Load models one at a time

### "Git LFS quota exceeded"
- Use cloud storage instead (see README_DEPLOYMENT.md)
- Or use Hugging Face for model hosting

### "Module not found"
- Ensure all dependencies in requirements.txt
- Check Python version compatibility