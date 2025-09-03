import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path('data/shibuya_pm25_data.csv')
df = pd.read_csv(data_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows")

growth_periods = []

for i in range(0, len(df) - 24):
    window = df.iloc[i:i+24]
    
    current = window['pm25_ugm3_mean'].iloc[0]
    if 8 <= current <= 12:
        next_6h = window['pm25_ugm3_mean'].iloc[1:7]
        next_12h = window['pm25_ugm3_mean'].iloc[1:13]
        
        max_6h = next_6h.max() if len(next_6h) > 0 else current
        max_12h = next_12h.max() if len(next_12h) > 0 else current
        
        if max_6h > 25 or max_12h > 30:
            growth_periods.append({
                'timestamp': window['timestamp'].iloc[0],
                'current': current,
                'hour_3': window['pm25_ugm3_mean'].iloc[3] if len(window) > 3 else current,
                'hour_6': window['pm25_ugm3_mean'].iloc[6] if len(window) > 6 else current,
                'max_6h': max_6h,
                'max_12h': max_12h,
                'growth_6h': max_6h - current,
                'crosses_30': max_6h > 30 or max_12h > 30
            })

if growth_periods:
    growth_df = pd.DataFrame(growth_periods)
    growth_df = growth_df[growth_df['crosses_30'] == True].sort_values('growth_6h', ascending=False)
    
    print("\nBest periods starting low (8-12) with significant growth to 30+:")
    for idx, row in growth_df.head(10).iterrows():
        print(f"\n{row['timestamp'].strftime('%Y-%m-%d %H:%M')}:")
        print(f"  Current: {row['current']:.1f} μg/m³")
        print(f"  After 3h: {row['hour_3']:.1f} μg/m³")
        print(f"  After 6h: {row['hour_6']:.1f} μg/m³")
        print(f"  Max in 6h: {row['max_6h']:.1f} (growth: +{row['growth_6h']:.1f})")
    
    if len(growth_df) > 0:
        best = growth_df.iloc[0]
        start_time = best['timestamp']
        end_time = start_time + pd.Timedelta(days=3)
        
        period_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        print(f"\n\nEXTRACTING BEST GROWTH PERIOD:")
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        print(f"Length: {len(period_data)} hours")
        
        output_path = Path('data/shibuya_growth_period.csv.gz')
        period_data.to_csv(output_path, compression='gzip', index=False)
        print(f"\nSaved to {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        print("\nFirst 12 hours progression:")
        for i in range(min(12, len(period_data))):
            val = period_data.iloc[i]['pm25_ugm3_mean']
            time = period_data.iloc[i]['timestamp']
            zone = "🟢 Running OK" if val < 10 else "🟡 Walking OK" if val < 30 else "🔴 Not safe"
            print(f"  Hour {i}: {val:.1f} μg/m³ - {zone}")