import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path('data/shibuya_pm25_data.csv')
df = pd.read_csv(data_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows")

growth_windows = []

for i in range(0, len(df) - 12):
    window = df.iloc[i:i+12]
    start_val = window['pm25_ugm3_mean'].iloc[0]
    
    if start_val < 10:
        max_in_6h = window['pm25_ugm3_mean'].iloc[:6].max()
        max_in_12h = window['pm25_ugm3_mean'].iloc[:12].max()
        
        if max_in_6h > 25 or max_in_12h > 30:
            growth_6h = max_in_6h - start_val
            growth_12h = max_in_12h - start_val
            
            growth_windows.append({
                'timestamp': window['timestamp'].iloc[0],
                'start_val': start_val,
                'max_6h': max_in_6h,
                'max_12h': max_in_12h,
                'growth_6h': growth_6h,
                'growth_12h': growth_12h,
                'crosses_both': (max_in_6h > 10 and max_in_12h > 30)
            })

if growth_windows:
    growth_df = pd.DataFrame(growth_windows)
    growth_df = growth_df.sort_values('growth_6h', ascending=False)
    
    print("\nBest growth periods (low start -> high in 6h):")
    for idx, row in growth_df.head(10).iterrows():
        print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')}: {row['start_val']:.1f} -> {row['max_6h']:.1f} in 6h (growth: {row['growth_6h']:.1f})")

print("\n\nSearching for 7-day periods with multiple growth events...")

best_weeks = []
for i in range(0, len(df) - 7*24, 24):
    window = df.iloc[i:i+7*24]
    
    if len(window) < 7*24:
        continue
    
    growth_events = 0
    for j in range(0, len(window) - 12, 6):
        sub_window = window.iloc[j:j+12]
        if len(sub_window) >= 12:
            start = sub_window['pm25_ugm3_mean'].iloc[0]
            end_6h = sub_window['pm25_ugm3_mean'].iloc[6] if len(sub_window) > 6 else start
            if start < 15 and end_6h > start + 15:
                growth_events += 1
    
    min_val = window['pm25_ugm3_mean'].min()
    max_val = window['pm25_ugm3_mean'].max()
    mean_val = window['pm25_ugm3_mean'].mean()
    
    below_10 = (window['pm25_ugm3_mean'] < 10).sum()
    between_10_30 = ((window['pm25_ugm3_mean'] >= 10) & (window['pm25_ugm3_mean'] < 30)).sum()
    above_30 = (window['pm25_ugm3_mean'] >= 30).sum()
    
    if growth_events > 0 and max_val > 30 and min_val < 10:
        best_weeks.append({
            'start': window['timestamp'].iloc[0],
            'end': window['timestamp'].iloc[-1],
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'growth_events': growth_events,
            'below_10': below_10,
            'between_10_30': between_10_30,
            'above_30': above_30
        })

if best_weeks:
    weeks_df = pd.DataFrame(best_weeks)
    weeks_df = weeks_df.sort_values('growth_events', ascending=False)
    
    print("\nBest 7-day periods with growth patterns:")
    for idx, row in weeks_df.head(5).iterrows():
        print(f"\n{row['start'].date()} to {row['end'].date()}:")
        print(f"  Range: {row['min']:.1f} - {row['max']:.1f} μg/m³")
        print(f"  Growth events: {row['growth_events']}")
        print(f"  Distribution: <10: {row['below_10']}h, 10-30: {row['between_10_30']}h, >30: {row['above_30']}h")

april_start = pd.Timestamp('2024-04-17 00:00:00', tz='UTC')
april_end = pd.Timestamp('2024-04-19 00:00:00', tz='UTC')
april_data = df[(df['timestamp'] >= april_start) & (df['timestamp'] <= april_end)]

print("\n\nApril 17-18, 2024 Analysis (known high growth):")
print(f"Data points: {len(april_data)}")
for i in range(0, min(48, len(april_data)), 3):
    val = april_data.iloc[i]['pm25_ugm3_mean']
    time = april_data.iloc[i]['timestamp']
    print(f"{time.strftime('%m-%d %H:%M')}: {val:.1f} μg/m³")