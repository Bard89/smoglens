import pandas as pd
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

@st.cache_data(ttl=3600)
def load_enriched_data() -> pd.DataFrame:
    df = pd.read_csv(
        config.DATA_PATH,
        parse_dates=['timestamp'],
        usecols=[
            'timestamp', 'hex7_id', 'lat', 'lon', 'pm25_ugm3_mean',
            'temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean',
            'avg_traffic_volume', 'congestion_index', 'data_completeness_score',
            'hour', 'day_of_week', 'month', 'year', 'is_weekend'
        ]
    )
    
    df = df.sort_values(['hex7_id', 'timestamp']).reset_index(drop=True)
    
    coverage_stats = df.groupby('hex7_id').agg({
        'pm25_ugm3_mean': lambda x: x.notna().mean(),
        'timestamp': ['min', 'max']
    })
    coverage_stats.columns = ['coverage_ratio', 'start_date', 'end_date']
    coverage_stats['duration_days'] = (
        coverage_stats['end_date'] - coverage_stats['start_date']
    ).dt.days
    
    high_coverage_hexagons = coverage_stats[
        (coverage_stats['coverage_ratio'] >= config.COVERAGE_THRESHOLD) & 
        (coverage_stats['duration_days'] >= config.MIN_DURATION_DAYS)
    ].index.tolist()
    
    df = df[df['hex7_id'].isin(high_coverage_hexagons)]
    
    return df