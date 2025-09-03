import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path('data/shibuya_pm25_data.csv')
df = pd.read_csv(data_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows of Shibuya data")

window_size = 12
walking_threshold = 30
running_threshold = 10

best_windows = []

for i in range(0, len(df) - window_size):
    window = df.iloc[i:i+window_size]
    
    min_val = window['pm25_ugm3_mean'].min()
    max_val = window['pm25_ugm3_mean'].max()
    mean_val = window['pm25_ugm3_mean'].mean()
    
    if min_val < 12 and max_val > 25 and max_val < 40:
        crosses_10 = ((window['pm25_ugm3_mean'] > running_threshold).astype(int).diff().abs().sum())
        crosses_30 = ((window['pm25_ugm3_mean'] > walking_threshold).astype(int).diff().abs().sum())
        
        if crosses_10 >= 1 and crosses_30 >= 1:
            best_windows.append({
                'start': window['timestamp'].iloc[0],
                'end': window['timestamp'].iloc[-1],
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'range': max_val - min_val,
                'crosses_10': crosses_10,
                'crosses_30': crosses_30,
                'ideal_score': abs(mean_val - 20) * -1
            })

if best_windows:
    windows_df = pd.DataFrame(best_windows)
    windows_df = windows_df.sort_values('ideal_score', ascending=False).head(20)
    
    print("\nBest 12-hour windows that cross both thresholds nicely:")
    for idx, row in windows_df.head(10).iterrows():
        print(f"{row['start'].strftime('%Y-%m-%d %H:%M')}: {row['min']:.1f} to {row['max']:.1f} (mean={row['mean']:.1f}, crosses 10: {row['crosses_10']}, crosses 30: {row['crosses_30']})")

print("\n\nSearching for 7-day periods with good threshold crossings...")

week_windows = []
for i in range(0, len(df) - 7*24, 24):
    window = df.iloc[i:i+7*24]
    
    if len(window) < 7*24:
        continue
        
    min_val = window['pm25_ugm3_mean'].min()
    max_val = window['pm25_ugm3_mean'].max()
    mean_val = window['pm25_ugm3_mean'].mean()
    
    below_10 = (window['pm25_ugm3_mean'] < 10).sum()
    between_10_30 = ((window['pm25_ugm3_mean'] >= 10) & (window['pm25_ugm3_mean'] < 30)).sum()
    above_30 = (window['pm25_ugm3_mean'] >= 30).sum()
    
    if below_10 > 20 and between_10_30 > 40 and above_30 > 10:
        week_windows.append({
            'start': window['timestamp'].iloc[0],
            'end': window['timestamp'].iloc[-1],
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'below_10': below_10,
            'between_10_30': between_10_30,
            'above_30': above_30,
            'score': min(below_10, between_10_30, above_30)
        })

if week_windows:
    week_df = pd.DataFrame(week_windows)
    week_df = week_df.sort_values('score', ascending=False)
    
    print("\nBest 7-day periods with good zone distribution:")
    for idx, row in week_df.head(5).iterrows():
        print(f"\n{row['start'].date()} to {row['end'].date()}:")
        print(f"  Range: {row['min']:.1f} - {row['max']:.1f} μg/m³ (mean: {row['mean']:.1f})")
        print(f"  Hours < 10: {row['below_10']} ({row['below_10']/168*100:.1f}%)")
        print(f"  Hours 10-30: {row['between_10_30']} ({row['between_10_30']/168*100:.1f}%)")
        print(f"  Hours > 30: {row['above_30']} ({row['above_30']/168*100:.1f}%)")
    
    best = week_df.iloc[0]
    print(f"\n\nRECOMMENDED PERIOD:")
    print(f"Start: {best['start']}")
    print(f"End: {best['end']}")
else:
    print("No ideal 7-day periods found, looking for shorter periods...")
    
    for i in range(0, len(df) - 3*24, 12):
        window = df.iloc[i:i+3*24]
        
        if len(window) < 3*24:
            continue
            
        min_val = window['pm25_ugm3_mean'].min()
        max_val = window['pm25_ugm3_mean'].max()
        mean_val = window['pm25_ugm3_mean'].mean()
        
        below_10 = (window['pm25_ugm3_mean'] < 10).sum()
        between_10_30 = ((window['pm25_ugm3_mean'] >= 10) & (window['pm25_ugm3_mean'] < 30)).sum()
        above_30 = (window['pm25_ugm3_mean'] >= 30).sum()
        
        if below_10 > 10 and between_10_30 > 20 and above_30 > 5:
            print(f"\n3-day period: {window['timestamp'].iloc[0].date()} to {window['timestamp'].iloc[-1].date()}")
            print(f"  Range: {min_val:.1f} - {max_val:.1f} μg/m³")
            print(f"  Distribution: <10: {below_10}h, 10-30: {between_10_30}h, >30: {above_30}h")
            break