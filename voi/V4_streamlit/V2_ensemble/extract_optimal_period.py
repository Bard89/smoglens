import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path('data/shibuya_pm25_data.csv')
output_path = Path('data/shibuya_optimal_period.csv.gz')

df = pd.read_csv(input_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows total")

start_date = pd.Timestamp('2024-04-15 19:00:00', tz='UTC')
end_date = pd.Timestamp('2024-04-23 08:00:00', tz='UTC')

period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
print(f"\nExtracted period from {start_date} to {end_date}")
print(f"Rows in period: {len(period_data)}")

pm25_stats = period_data['pm25_ugm3_mean'].describe()
print(f"\nPM2.5 Statistics:")
print(f"Min: {pm25_stats['min']:.1f} μg/m³")
print(f"Max: {pm25_stats['max']:.1f} μg/m³")
print(f"Mean: {pm25_stats['mean']:.1f} μg/m³")
print(f"Std: {pm25_stats['std']:.1f} μg/m³")

below_10 = (period_data['pm25_ugm3_mean'] < 10).sum()
between_10_30 = ((period_data['pm25_ugm3_mean'] >= 10) & (period_data['pm25_ugm3_mean'] < 30)).sum()
above_30 = (period_data['pm25_ugm3_mean'] >= 30).sum()
total_hours = len(period_data)

print(f"\nZone Distribution:")
print(f"Safe for running (<10 μg/m³): {below_10} hours ({below_10*100/total_hours:.1f}%)")
print(f"Caution zone (10-30 μg/m³): {between_10_30} hours ({between_10_30*100/total_hours:.1f}%)")
print(f"Not safe for walking (>30 μg/m³): {above_30} hours ({above_30*100/total_hours:.1f}%)")

walking_safe = (period_data['pm25_ugm3_mean'] < 30).sum()
running_safe = (period_data['pm25_ugm3_mean'] < 10).sum()
print(f"\nActivity Safety:")
print(f"Safe for walking: {walking_safe} hours ({walking_safe*100/total_hours:.1f}%)")
print(f"Safe for running: {running_safe} hours ({running_safe*100/total_hours:.1f}%)")

period_data.to_csv(output_path, compression='gzip', index=False)
print(f"\nSaved optimal period to {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

print("\nSample of transitions:")
samples = period_data[['timestamp', 'pm25_ugm3_mean']].iloc[::6]
for _, row in samples.head(10).iterrows():
    val = row['pm25_ugm3_mean']
    zone = "🟢 Safe for running" if val < 10 else "🟡 Caution" if val < 30 else "🔴 Unsafe"
    print(f"{row['timestamp'].strftime('%m-%d %H:%M')}: {val:.1f} μg/m³ - {zone}")