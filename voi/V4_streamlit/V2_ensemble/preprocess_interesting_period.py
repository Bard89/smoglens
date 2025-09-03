import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path('data/shibuya_pm25_data.csv')
output_path = Path('data/shibuya_interesting_period.csv.gz')

df = pd.read_csv(input_path, parse_dates=['timestamp'], compression='gzip')
print(f"Loaded {len(df)} rows total")

start_date = pd.Timestamp('2024-04-10', tz='UTC')
end_date = pd.Timestamp('2024-04-25', tz='UTC')

period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
print(f"\nExtracted period from {start_date} to {end_date}")
print(f"Rows in period: {len(period_data)}")

pm25_stats = period_data['pm25_ugm3_mean'].describe()
print(f"\nPM2.5 Statistics for this period:")
print(f"Min: {pm25_stats['min']:.1f} μg/m³")
print(f"Max: {pm25_stats['max']:.1f} μg/m³")
print(f"Mean: {pm25_stats['mean']:.1f} μg/m³")
print(f"Std: {pm25_stats['std']:.1f} μg/m³")
print(f"Range: {pm25_stats['max'] - pm25_stats['min']:.1f} μg/m³")

walking_threshold = 30
running_threshold = 10
walking_safe = (period_data['pm25_ugm3_mean'] < walking_threshold).sum()
running_safe = (period_data['pm25_ugm3_mean'] < running_threshold).sum()
total_hours = len(period_data)

print(f"\nActivity Analysis:")
print(f"Hours safe for walking (<{walking_threshold} μg/m³): {walking_safe}/{total_hours} ({walking_safe*100/total_hours:.1f}%)")
print(f"Hours safe for running (<{running_threshold} μg/m³): {running_safe}/{total_hours} ({running_safe*100/total_hours:.1f}%)")

threshold_changes = ((period_data['pm25_ugm3_mean'] > walking_threshold).astype(int).diff().abs().sum() +
                    (period_data['pm25_ugm3_mean'] > running_threshold).astype(int).diff().abs().sum())
print(f"Activity threshold crossings: {threshold_changes}")

period_data.to_csv(output_path, compression='gzip', index=False)
print(f"\nSaved interesting period to {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

print("\n\nDetailed hourly variation preview:")
hourly_avg = period_data.groupby(period_data['timestamp'].dt.hour)['pm25_ugm3_mean'].agg(['mean', 'std', 'min', 'max'])
print(hourly_avg)