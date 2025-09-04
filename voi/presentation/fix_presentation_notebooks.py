#!/usr/bin/env python3

import json
import numpy as np

print("="*70)
print("UPDATING PRESENTATION NOTEBOOKS WITH REAL DATA")
print("="*70)

with open('real_data_analysis.json', 'r') as f:
    real_data = json.load(f)

horizon_for_display = '6h'

temporal_patterns = real_data['temporal_patterns'][horizon_for_display]
hourly_mae = temporal_patterns['hourly_mae']
weekday_mae = temporal_patterns['weekday_mae']

print(f"\nReal {horizon_for_display} Weekday MAE:")
print(f"weekday_mae = {[round(x, 2) for x in weekday_mae]}")

print(f"\nReal {horizon_for_display} Hourly MAE (selected hours):")
for h in [0, 6, 12, 18]:
    print(f"  Hour {h:02d}:00 - MAE: {hourly_mae[h]:.3f}")

print("\nTop hours with highest/lowest errors:")
max_hour = np.argmax(hourly_mae)
min_hour = np.argmin(hourly_mae)
print(f"  Highest error: Hour {max_hour:02d}:00 (MAE: {hourly_mae[max_hour]:.3f})")
print(f"  Lowest error:  Hour {min_hour:02d}:00 (MAE: {hourly_mae[min_hour]:.3f})")

feature_importance = real_data.get('feature_importance', {})
if '6h' in feature_importance:
    print(f"\nReal Feature Importance for {horizon_for_display}:")
    features_6h = feature_importance['6h'][:12]
    
    feature_dict = {}
    for feat in features_6h:
        feature_dict[feat['feature']] = feat['percentage']
    
    print("\nfeature_importance = {")
    for feat_name, pct in feature_dict.items():
        print(f"    '{feat_name}': {pct},")
    print("}")
    
    categories = {
        'EWM/Rolling': sum(pct for name, pct in feature_dict.items() 
                          if 'ewm' in name or 'rolling' in name),
        'Spatial': sum(pct for name, pct in feature_dict.items() 
                      if 'hex' in name),
        'Weather': sum(pct for name, pct in feature_dict.items()
                      if any(w in name for w in ['temp', 'humid', 'pressure', 'cloud', 'solar'])),
        'Temporal': sum(pct for name, pct in feature_dict.items()
                       if any(t in name for t in ['hour', 'dow', 'month', 'day'])),
        'Lag/Diff': sum(pct for name, pct in feature_dict.items()
                       if 'lag' in name or 'diff' in name),
        'Other': 0
    }
    
    categories['Other'] = 100 - sum(categories.values())
    
    print("\nFeature Categories:")
    for cat, pct in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {pct:.1f}%")

agreement = real_data['model_agreement']
print("\nReal Model Agreement by Horizon:")
print("agreement_by_horizon = {")
for h in ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']:
    if h in agreement:
        avg_agreement = agreement[h]['average_agreement']
        print(f"    '{h}': {avg_agreement:.3f},")
print("}")

print("\nAgreement matrix for 1h:")
if '1h' in agreement:
    matrix = agreement['1h']['matrix']
    print("agreement_matrix = np.array([")
    for row in matrix:
        print(f"    {[round(x, 3) for x in row]},")
    print("])")

error_dist = real_data['error_distributions'][horizon_for_display]
print(f"\nReal Error Statistics for {horizon_for_display}:")
for model in ['lgb', 'xgb', 'cat', 'ensemble']:
    stats = error_dist[model]
    print(f"\n{model.upper()}:")
    print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    print(f"  Q25: {stats['q25']:.2f}, Q50: {stats['q50']:.2f}, Q75: {stats['q75']:.2f}")

print("\n" + "="*70)
print("NOTEBOOK UPDATE CODE")
print("="*70)

print("\nReplace in 03_ensemble_insights.ipynb, Cell 11:")
print("-"*50)
print(f"""
# REAL DATA from actual model predictions
weekday_mae = {[round(x, 2) for x in weekday_mae]}

# Generate hourly pattern from real data
hourly_mae = {[round(x, 3) for x in hourly_mae]}
""")

print("\nReplace feature_importance dictionary:")
print("-"*50)
if '6h' in feature_importance:
    print("feature_importance = {")
    for feat in features_6h[:12]:
        print(f"    '{feat['feature']}': {feat['percentage']},")
    print("}")

print("\nReplace agreement_by_horizon:")
print("-"*50)
print("agreement_by_horizon = {")
for h in ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']:
    if h in agreement:
        print(f"    '{h}': {agreement[h]['average_agreement']:.3f},")
print("}")

print("\nKey Insights to Update:")
print("-"*50)
print("1. EWM features dominate (40-70%), NOT lag features")
print("2. Location (hex_encoded) is 2nd most important (15-23%)")
print("3. Models agree VERY highly (>0.92 even at 24h)")
print("4. Weekday errors are consistent (~3.4 for 6h), no weekend effect")
print("5. Hourly variation is minimal (< 0.1 μg/m³ difference)")