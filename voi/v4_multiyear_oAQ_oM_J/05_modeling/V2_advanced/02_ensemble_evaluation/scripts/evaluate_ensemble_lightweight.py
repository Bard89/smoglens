#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import pickle
import json
import gc
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LIGHTWEIGHT ENSEMBLE EVALUATION")
print("="*70)
print(f"Started: {datetime.now()}")
print("This script evaluates models without loading all data at once")
print("-"*70)

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

def ensemble_loss(weights, preds, y_true):
    ensemble_pred = np.zeros_like(y_true, dtype=np.float32)
    for i, pred in enumerate(preds):
        ensemble_pred += weights[i] * pred
    return mean_squared_error(y_true, ensemble_pred)

def process_horizon(horizon):
    print(f"\nProcessing {horizon}...")
    
    try:
        pred_file = f'../../01_ensemble_training/trained/predictions_{horizon}.pkl'
        if not os.path.exists(pred_file):
            print(f"  ✗ Predictions file not found: {pred_file}")
            return None
        
        with open(pred_file, 'rb') as f:
            preds = joblib.load(f)
        
        val_preds = [preds['val']['lgb'], preds['val']['xgb'], preds['val']['cat']]
        test_preds = [preds['test']['lgb'], preds['test']['xgb'], preds['test']['cat']]
        
        with open('../../01_ensemble_training/trained/y_data.pkl', 'rb') as f:
            y_data = pickle.load(f)
        
        y_val = y_data['y_val'][f'target_{horizon}']
        y_test = y_data['y_test'][f'target_{horizon}']
        
        print(f"  Optimizing ensemble weights...")
        initial_weights = np.array([1/3, 1/3, 1/3])
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * 3
        
        result = minimize(
            ensemble_loss,
            initial_weights,
            args=(val_preds, y_val),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        print(f"  Weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, Cat={weights[2]:.3f}")
        
        ensemble_test = np.zeros_like(test_preds[0])
        for i, pred in enumerate(test_preds):
            ensemble_test += weights[i] * pred
        ensemble_test = np.clip(ensemble_test, 0, 100)
        
        results = {
            'weights': weights.tolist(),
            'models': {}
        }
        
        for i, model_name in enumerate(['lgb', 'xgb', 'cat']):
            mae = mean_absolute_error(y_test, test_preds[i])
            rmse = np.sqrt(mean_squared_error(y_test, test_preds[i]))
            r2 = r2_score(y_test, test_preds[i])
            results['models'][model_name.upper()] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }
        
        ens_mae = mean_absolute_error(y_test, ensemble_test)
        ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test))
        ens_r2 = r2_score(y_test, ensemble_test)
        
        results['ensemble'] = {
            'mae': float(ens_mae),
            'rmse': float(ens_rmse),
            'r2': float(ens_r2)
        }
        
        baseline = baseline_targets[horizon]
        results['baseline'] = baseline
        results['improvement'] = {
            'mae_pct': float((baseline['mae'] - ens_mae) / baseline['mae'] * 100),
            'r2_pct': float((ens_r2 - baseline['r2']) / baseline['r2'] * 100),
            'beat_mae': ens_mae < baseline['mae'],
            'beat_r2': ens_r2 > baseline['r2']
        }
        
        print(f"  Ensemble MAE: {ens_mae:.3f} (baseline: {baseline['mae']:.3f})")
        print(f"  Ensemble R²:  {ens_r2:.3f} (baseline: {baseline['r2']:.3f})")
        print(f"  Beat MAE: {'✓' if results['improvement']['beat_mae'] else '✗'} ({results['improvement']['mae_pct']:+.1f}%)")
        print(f"  Beat R²:  {'✓' if results['improvement']['beat_r2'] else '✗'} ({results['improvement']['r2_pct']:+.1f}%)")
        
        del preds, val_preds, test_preds, y_data, y_val, y_test, ensemble_test
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None

def main():
    horizons = ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']
    all_results = {}
    
    for horizon in horizons:
        result = process_horizon(horizon)
        if result:
            all_results[horizon] = result
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    mae_data = []
    r2_data = []
    
    for h in horizons:
        if h in all_results:
            res = all_results[h]
            mae_row = {
                'Horizon': h,
                'LGB': res['models']['LGB']['mae'],
                'XGB': res['models']['XGB']['mae'],
                'CAT': res['models']['CAT']['mae'],
                'Ensemble': res['ensemble']['mae'],
                'Baseline': res['baseline']['mae'],
                'Improvement': f"{res['improvement']['mae_pct']:+.1f}%"
            }
            mae_data.append(mae_row)
            
            r2_row = {
                'Horizon': h,
                'LGB': res['models']['LGB']['r2'],
                'XGB': res['models']['XGB']['r2'],
                'CAT': res['models']['CAT']['r2'],
                'Ensemble': res['ensemble']['r2'],
                'Baseline': res['baseline']['r2'],
                'Improvement': f"{res['improvement']['r2_pct']:+.1f}%"
            }
            r2_data.append(r2_row)
    
    print("\nMAE Comparison (μg/m³):")
    print("-"*70)
    mae_df = pd.DataFrame(mae_data)
    print(mae_df.to_string(index=False))
    
    print("\n\nR² Score Comparison:")
    print("-"*70)
    r2_df = pd.DataFrame(r2_data)
    print(r2_df.to_string(index=False))
    
    total_beat_mae = sum(1 for h in all_results.values() if h['improvement']['beat_mae'])
    total_beat_r2 = sum(1 for h in all_results.values() if h['improvement']['beat_r2'])
    
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)
    print(f"Beat Linear Regression on MAE: {total_beat_mae}/{len(all_results)} horizons")
    print(f"Beat Linear Regression on R²:  {total_beat_r2}/{len(all_results)} horizons")
    
    avg_mae_imp = np.mean([r['improvement']['mae_pct'] for r in all_results.values()])
    avg_r2_imp = np.mean([r['improvement']['r2_pct'] for r in all_results.values()])
    print(f"\nAverage MAE improvement: {avg_mae_imp:+.1f}%")
    print(f"Average R² improvement:  {avg_r2_imp:+.1f}%")
    
    with open('../ensemble_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to ../ensemble_evaluation_results.json")
    
    mae_df.to_csv('mae_comparison.csv', index=False)
    r2_df.to_csv('r2_comparison.csv', index=False)
    print(f"Tables saved to mae_comparison.csv and r2_comparison.csv")
    
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()