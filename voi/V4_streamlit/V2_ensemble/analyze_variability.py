import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path('/Users/vojtech/Code/Bard89/Project-Data/data/processed/phase_3_imputed_2023_2025.csv')
print(f"Reading data from {data_path}")
df = pd.read_csv(data_path, parse_dates=['timestamp'])

shibuya_hex = '872e44d04ffffff'
shibuya_data = df[df['hex7'] == shibuya_hex].copy()
print(f"Found {len(shibuya_data)} rows for Shibuya hexagon")

shibuya_data['pm25_rolling_std_24h'] = shibuya_data['pm25_ugm3_mean'].rolling(window=24, center=False).std()
shibuya_data['pm25_rolling_range_24h'] = shibuya_data['pm25_ugm3_mean'].rolling(window=24, center=False).apply(lambda x: x.max() - x.min())
shibuya_data['pm25_change_6h'] = shibuya_data['pm25_ugm3_mean'].diff(6).abs()

shibuya_data['variability_score'] = (
    shibuya_data['pm25_rolling_std_24h'].fillna(0) * 0.4 + 
    shibuya_data['pm25_rolling_range_24h'].fillna(0) * 0.3 +
    shibuya_data['pm25_change_6h'].fillna(0) * 0.3
)

top_periods = shibuya_data.nlargest(20, 'variability_score')[['timestamp', 'pm25_ugm3_mean', 'variability_score', 'pm25_rolling_std_24h', 'pm25_rolling_range_24h']]
print("\nTop 20 most variable periods:")
print(top_periods)

interesting_periods = []
for idx, row in top_periods.iterrows():
    start = row['timestamp'] - pd.Timedelta(days=3)
    end = row['timestamp'] + pd.Timedelta(days=3)
    period_data = shibuya_data[(shibuya_data['timestamp'] >= start) & (shibuya_data['timestamp'] <= end)]
    
    if len(period_data) > 100:
        interesting_periods.append({
            'center_time': row['timestamp'],
            'variability_score': row['variability_score'],
            'min_pm25': period_data['pm25_ugm3_mean'].min(),
            'max_pm25': period_data['pm25_ugm3_mean'].max(),
            'range': period_data['pm25_ugm3_mean'].max() - period_data['pm25_ugm3_mean'].min(),
            'std': period_data['pm25_ugm3_mean'].std()
        })

interesting_df = pd.DataFrame(interesting_periods).sort_values('range', ascending=False)
print("\n\nMost interesting 7-day periods (sorted by PM2.5 range):")
print(interesting_df.head(10))

best_period = interesting_df.iloc[0]
print(f"\n\nBest period for demonstration:")
print(f"Center: {best_period['center_time']}")
print(f"PM2.5 range: {best_period['min_pm25']:.1f} - {best_period['max_pm25']:.1f} μg/m³")
print(f"Standard deviation: {best_period['std']:.1f}")

recent_high_variability = shibuya_data[shibuya_data['timestamp'] >= '2024-10-01']
recent_high_variability['variability_score'] = (
    recent_high_variability['pm25_rolling_std_24h'].fillna(0) * 0.4 + 
    recent_high_variability['pm25_rolling_range_24h'].fillna(0) * 0.3 +
    recent_high_variability['pm25_change_6h'].fillna(0) * 0.3
)

print("\n\nRecent periods (after Oct 2024) with high variability:")
recent_top = recent_high_variability.nlargest(10, 'variability_score')[['timestamp', 'pm25_ugm3_mean', 'variability_score']]
print(recent_top)

print("\n\nRecommended period for extraction:")
print("2024-11-15 to 2024-12-31 - covers recent high variability with good seasonal variation")