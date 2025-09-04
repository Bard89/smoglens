# SmogLens Deployment Guide

## Quick Deploy to Streamlit Cloud

### 1. Prepare Models
Since the models are large (~200MB), you have options:

**Option A: Git LFS (Recommended for <1GB)**
```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track model files
cd voi/V4_streamlit/V2_working
rm models  # Remove symlink
cp -r /Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained models
git lfs track "models/*.pkl"
git add .gitattributes models/
git commit -m "Add model files with LFS"
git push
```

**Option B: Cloud Storage**
1. Upload models to Google Drive/Dropbox/S3
2. Download in app initialization:
```python
@st.cache_resource
def download_models():
    import gdown  # for Google Drive
    if not Path('models').exists():
        gdown.download_folder('YOUR_GDRIVE_FOLDER_ID', output='models')
```

**Option C: Streamlit Secrets**
For private S3/Azure:
1. Add credentials to Streamlit Secrets
2. Download using boto3/azure-storage

### 2. Handle Large Data File

**For the 9.4M record CSV:**

Create a smaller sample:
```bash
# Take recent 3 months of Shibuya data only
python -c "
import pandas as pd
df = pd.read_csv('/Users/vojtech/Code/Bard89/smoglens-02/data/pm25_enriched_2023_2025_v4_20250830_222050.csv', parse_dates=['timestamp'])
shibuya = df[df['hex7_id'] == '872e44d04ffffff']
recent = shibuya[shibuya['timestamp'] > '2025-04-01']
recent.to_csv('voi/V4_streamlit/V2_working/data/shibuya_recent.csv', index=False)
print(f'Saved {len(recent)} records')
"
```

### 3. Environment Variables

Set in Streamlit Cloud dashboard:
```
STREAMLIT_CLOUD=true
```

### 4. Memory Optimization

Streamlit Cloud has 1GB RAM limit. Add to app:
```python
import gc
gc.collect()  # After loading models
```

## Local Testing of Cloud Setup

```bash
# Test with cloud config
export STREAMLIT_CLOUD=true
streamlit run app.py
```

## Deployment URL

Once deployed, your app will be available at:
```
https://smoglens-shibuya.streamlit.app
```

## Monitoring

- Check logs in Streamlit Cloud dashboard
- Set up email alerts for app crashes
- Monitor resource usage (RAM/CPU)

## Troubleshooting

### "Module not found" error
- Check requirements.txt has all dependencies
- Verify Python version compatibility

### "Memory limit exceeded"
- Reduce model complexity
- Use data sampling
- Clear cache more aggressively

### "Data file not found"
- Verify file paths are relative
- Check Git LFS tracked properly
- Ensure cloud storage URLs are accessible

## Alternative Deployment Options

### Hugging Face Spaces
```bash
# Create space at huggingface.co/spaces
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smoglens
git push hf main
```

### Railway/Render
- Better for larger models (>1GB)
- More RAM available
- Costs ~$5-20/month

### Docker + Cloud Run
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port $PORT
```