import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

print("Loading predictions and test data...")

with open('../../01_ensemble_training/trained/y_data.pkl', 'rb') as f:
    y_data = pickle.load(f)
    y_test = y_data['y_test']

predictions_test = {}
horizons = ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']

for horizon in horizons:
    try:
        pred_file = f'../../01_ensemble_training/trained/predictions_{horizon}.pkl'
        preds = joblib.load(pred_file)
        predictions_test[horizon] = preds['test']
        print(f"✓ Loaded {horizon} predictions")
    except Exception as e:
        print(f"✗ Could not load {horizon}: {e}")

with open('../ensemble_evaluation_results.json', 'r') as f:
    results = json.load(f)

baseline_targets = {
    '1h': {'mae': 2.08, 'r2': 0.845},
    '2h': {'mae': 2.55, 'r2': 0.765},
    '3h': {'mae': 2.90, 'r2': 0.697},
    '4h': {'mae': 3.17, 'r2': 0.641},
    '5h': {'mae': 3.39, 'r2': 0.592},
    '6h': {'mae': 3.57, 'r2': 0.550},
    '12h': {'mae': 4.24, 'r2': 0.379},
    '24h': {'mae': 4.84, 'r2': 0.201}
}

fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.ravel()

print("\nGenerating predicted vs true plots for all models...")

for idx, horizon in enumerate(horizons):
    ax = axes[idx]
    
    y_true = y_test[f'target_{horizon}']
    
    sample_size = min(5000, len(y_true))
    sample_idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
    
    colors = {'lgb': 'blue', 'xgb': 'green', 'cat': 'orange'}
    alphas = {'lgb': 0.3, 'xgb': 0.3, 'cat': 0.3}
    
    for model_name in ['lgb', 'xgb', 'cat']:
        y_pred = predictions_test[horizon][model_name]
        ax.scatter(y_true[sample_idx], y_pred[sample_idx], 
                  alpha=alphas[model_name], s=1, 
                  color=colors[model_name], label=model_name.upper())
    
    ax.plot([0, 60], [0, 60], 'r--', lw=1.5, label='Perfect prediction')
    
    ensemble_mae = results[horizon]['ensemble']['mae']
    ensemble_r2 = results[horizon]['ensemble']['r2']
    baseline_mae = baseline_targets[horizon]['mae']
    baseline_r2 = baseline_targets[horizon]['r2']
    
    mae_improvement = (baseline_mae - ensemble_mae) / baseline_mae * 100
    r2_change = (ensemble_r2 - baseline_r2) / baseline_r2 * 100
    
    title_lines = [
        f'{horizon} Forecast',
        f'Ensemble: MAE={ensemble_mae:.2f} (vs {baseline_mae:.2f}), R²={ensemble_r2:.3f} (vs {baseline_r2:.3f})',
        f'Improvement: MAE {mae_improvement:+.1f}%, R² {r2_change:+.1f}%'
    ]
    ax.set_title('\n'.join(title_lines), fontsize=9)
    
    ax.set_xlabel('True PM2.5 (μg/m³)')
    ax.set_ylabel('Predicted PM2.5 (μg/m³)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.legend(loc='upper left', fontsize=8)

plt.suptitle('PM2.5 Predictions vs True Values - All Models (LightGBM, XGBoost, CatBoost)', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('predictions_vs_true_all_models.png', dpi=150, bbox_inches='tight')
print("✓ Saved: predictions_vs_true_all_models.png")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.ravel()

print("\nGenerating ensemble-only predicted vs true plots...")

for idx, horizon in enumerate(horizons):
    ax = axes[idx]
    
    y_true = y_test[f'target_{horizon}']
    
    ensemble_pred = (predictions_test[horizon]['lgb'] + 
                    predictions_test[horizon]['xgb'] + 
                    predictions_test[horizon]['cat']) / 3
    
    sample_size = min(5000, len(y_true))
    sample_idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
    
    ax.scatter(y_true[sample_idx], ensemble_pred[sample_idx], 
              alpha=0.4, s=2, color='darkblue')
    
    ax.plot([0, 60], [0, 60], 'r--', lw=1.5, alpha=0.8)
    
    ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
    ensemble_r2 = r2_score(y_true, ensemble_pred)
    baseline_mae = baseline_targets[horizon]['mae']
    baseline_r2 = baseline_targets[horizon]['r2']
    
    mae_beat = '✓' if ensemble_mae < baseline_mae else '✗'
    r2_beat = '✓' if ensemble_r2 > baseline_r2 else '✗'
    
    title_lines = [
        f'{horizon} Ensemble Forecast',
        f'MAE: {ensemble_mae:.3f} vs {baseline_mae:.3f} {mae_beat}',
        f'R²: {ensemble_r2:.3f} vs {baseline_r2:.3f} {r2_beat}'
    ]
    ax.set_title('\n'.join(title_lines), fontsize=10)
    
    ax.set_xlabel('True PM2.5 (μg/m³)')
    ax.set_ylabel('Predicted PM2.5 (μg/m³)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

plt.suptitle('PM2.5 Ensemble Predictions vs True Values (vs Linear Regression Baseline)', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('predictions_vs_true_ensemble.png', dpi=150, bbox_inches='tight')
print("✓ Saved: predictions_vs_true_ensemble.png")
plt.show()

fig, axes = plt.subplots(3, 8, figsize=(24, 10))

print("\nGenerating individual model comparison grid...")

models = ['lgb', 'xgb', 'cat']
model_names = {'lgb': 'LightGBM', 'xgb': 'XGBoost', 'cat': 'CatBoost'}

for row_idx, model in enumerate(models):
    for col_idx, horizon in enumerate(horizons):
        ax = axes[row_idx, col_idx]
        
        y_true = y_test[f'target_{horizon}']
        y_pred = predictions_test[horizon][model]
        
        sample_size = min(2000, len(y_true))
        sample_idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
        
        ax.scatter(y_true[sample_idx], y_pred[sample_idx], 
                  alpha=0.3, s=0.5, color='blue')
        ax.plot([0, 50], [0, 50], 'r--', lw=0.8, alpha=0.7)
        
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax.set_title(f'{model_names[model]} {horizon}\nMAE={mae:.2f}, R²={r2:.3f}', 
                    fontsize=8)
        
        if row_idx == 2:
            ax.set_xlabel('True', fontsize=7)
        if col_idx == 0:
            ax.set_ylabel('Predicted', fontsize=7)
        
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.tick_params(labelsize=6)

plt.suptitle('Model Comparison: Predicted vs True PM2.5 for All Horizons', fontsize=14)
plt.tight_layout()
plt.savefig('predictions_vs_true_model_grid.png', dpi=150, bbox_inches='tight')
print("✓ Saved: predictions_vs_true_model_grid.png")
plt.show()

print("\n" + "="*60)
print("SUMMARY OF GENERATED PLOTS")
print("="*60)
print("1. predictions_vs_true_all_models.png - All models overlaid")
print("2. predictions_vs_true_ensemble.png - Ensemble only with baseline comparison")
print("3. predictions_vs_true_model_grid.png - Individual model grid (3x8)")
print("\nAll plots saved successfully!")