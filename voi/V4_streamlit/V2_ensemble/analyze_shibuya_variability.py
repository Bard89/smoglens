import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path('data/shibuya_pm25_data.csv')
df = pd.read_csv(data_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows of Shibuya data")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

df['pm25_rolling_std_24h'] = df['pm25_ugm3_mean'].rolling(window=24, center=False).std()
df['pm25_rolling_range_24h'] = df['pm25_ugm3_mean'].rolling(window=24, center=False).apply(lambda x: x.max() - x.min())
df['pm25_rolling_range_48h'] = df['pm25_ugm3_mean'].rolling(window=48, center=False).apply(lambda x: x.max() - x.min())
df['pm25_change_6h'] = df['pm25_ugm3_mean'].diff(6).abs()
df['pm25_change_12h'] = df['pm25_ugm3_mean'].diff(12).abs()

df['variability_score'] = (
    df['pm25_rolling_std_24h'].fillna(0) * 0.3 + 
    df['pm25_rolling_range_48h'].fillna(0) * 0.4 +
    df['pm25_change_12h'].fillna(0) * 0.3
)

top_periods = df.nlargest(30, 'variability_score')[['timestamp', 'pm25_ugm3_mean', 'variability_score', 'pm25_rolling_std_24h', 'pm25_rolling_range_48h']]
print("\n\nTop 30 most variable periods:")
for idx, row in top_periods.head(10).iterrows():
    print(f"{row['timestamp']}: PM2.5={row['pm25_ugm3_mean']:.1f}, Score={row['variability_score']:.1f}, 48h_range={row['pm25_rolling_range_48h']:.1f}")

interesting_windows = []
window_size = 7 * 24
step = 24

for i in range(0, len(df) - window_size, step):
    window = df.iloc[i:i+window_size]
    if len(window) >= window_size:
        interesting_windows.append({
            'start': window['timestamp'].iloc[0],
            'end': window['timestamp'].iloc[-1],
            'mean_pm25': window['pm25_ugm3_mean'].mean(),
            'std_pm25': window['pm25_ugm3_mean'].std(),
            'min_pm25': window['pm25_ugm3_mean'].min(),
            'max_pm25': window['pm25_ugm3_mean'].max(),
            'range': window['pm25_ugm3_mean'].max() - window['pm25_ugm3_mean'].min(),
            'num_threshold_changes': ((window['pm25_ugm3_mean'] > 30).astype(int).diff().abs().sum() + 
                                     (window['pm25_ugm3_mean'] > 10).astype(int).diff().abs().sum())
        })

windows_df = pd.DataFrame(interesting_windows)
windows_df['interest_score'] = (
    windows_df['range'] * 0.4 + 
    windows_df['std_pm25'] * 0.3 + 
    windows_df['num_threshold_changes'] * 0.3
)

best_windows = windows_df.nlargest(20, 'interest_score')
print("\n\nBest 7-day windows for demonstration (high variation + threshold crossings):")
for idx, row in best_windows.head(10).iterrows():
    print(f"{row['start'].date()} to {row['end'].date()}: Range={row['range']:.1f}, Std={row['std_pm25']:.1f}, Threshold changes={row['num_threshold_changes']:.0f}")

recent_windows = windows_df[windows_df['start'] >= '2024-10-01'].nlargest(5, 'interest_score')
print("\n\nBest recent windows (after Oct 2024):")
for idx, row in recent_windows.iterrows():
    print(f"{row['start'].date()} to {row['end'].date()}: Range={row['range']:.1f}, Mean={row['mean_pm25']:.1f}, Changes={row['num_threshold_changes']:.0f}")

best_recent = recent_windows.iloc[0] if len(recent_windows) > 0 else best_windows.iloc[0]
print(f"\n\nRECOMMENDED EXTRACTION PERIOD:")
print(f"Start: {best_recent['start']}")
print(f"End: {best_recent['end']}")
print(f"PM2.5 Range: {best_recent['min_pm25']:.1f} - {best_recent['max_pm25']:.1f} μg/m³")
print(f"Mean: {best_recent['mean_pm25']:.1f} μg/m³, Std: {best_recent['std_pm25']:.1f}")
print(f"Activity threshold crossings: {best_recent['num_threshold_changes']:.0f}")