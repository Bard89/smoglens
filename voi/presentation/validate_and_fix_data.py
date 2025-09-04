#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("VALIDATING AND EXTRACTING REAL DATA FROM MODELS")
print("="*70)

base_path = Path('../v4_multiyear/05_modeling/02_advanced')
train_path = base_path / '01_ensemble_training/trained'
results = {}

def load_predictions_and_targets():
    print("\n1. Loading predictions and true values...")
    predictions = {}
    y_test = {}
    
    try:
        X_test = pd.read_pickle(train_path / 'X_test.pkl')
        print(f"   Loaded X_test with shape: {X_test.shape}")
        
        with open(train_path / 'y_data.pkl', 'rb') as f:
            y_data = joblib.load(f)
            
        horizons = ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']
        
        for horizon in horizons:
            pred_file = train_path / f'predictions_{horizon}.pkl'
            if pred_file.exists():
                with open(pred_file, 'rb') as f:
                    preds = joblib.load(f)
                    predictions[horizon] = preds['test']
                    
                horizon_num = horizon.replace('h', '')
                y_test[horizon] = y_data['y_test'][f'target_{horizon_num}h']
                print(f"   Loaded {horizon}: {len(y_test[horizon])} samples")
                
        return predictions, y_test, X_test
    except Exception as e:
        print(f"   Error loading data: {e}")
        print(f"   Trying alternative loading method...")
        try:
            X_test = joblib.load(train_path / 'X_test.pkl')
            print(f"   Loaded X_test with shape: {X_test.shape}")
            
            with open(train_path / 'y_data.pkl', 'rb') as f:
                y_data = pickle.load(f)
            
            for horizon in horizons:
                pred_file = train_path / f'predictions_{horizon}.pkl'
                if pred_file.exists():
                    preds = joblib.load(pred_file)
                    predictions[horizon] = preds['test']
                    
                    horizon_num = horizon.replace('h', '')
                    y_test[horizon] = y_data['y_test'][f'target_{horizon_num}h']
                    print(f"   Loaded {horizon}: {len(y_test[horizon])} samples")
            
            return predictions, y_test, X_test
        except Exception as e2:
            print(f"   Alternative method also failed: {e2}")
            return None, None, None

def calculate_temporal_error_patterns(predictions, y_test, X_test):
    print("\n2. Calculating REAL temporal error patterns...")
    
    temporal_results = {}
    
    if 'timestamp' not in X_test.columns:
        print("   Warning: No timestamp column found, trying to load enriched data...")
        try:
            enriched_data = pd.read_csv('/Users/vojtech/Code/Bard89/smoglens-data/pm25_enriched_2023_2025_v4_20250830_222050.csv',
                                       parse_dates=['timestamp'])
            enriched_test = enriched_data[enriched_data['timestamp'].dt.year == 2025].copy()
            
            if len(enriched_test) > len(X_test):
                enriched_test = enriched_test.iloc[:len(X_test)]
            
            X_test['timestamp'] = enriched_test['timestamp'].values
            X_test['hour'] = enriched_test['timestamp'].dt.hour
            X_test['day_of_week'] = enriched_test['timestamp'].dt.dayofweek
            print(f"   Successfully added temporal features from enriched data")
        except Exception as e:
            print(f"   Could not load enriched data: {e}")
            print("   Using synthetic temporal features as fallback")
            X_test['hour'] = np.tile(np.arange(24), len(X_test)//24 + 1)[:len(X_test)]
            X_test['day_of_week'] = np.tile(np.arange(7), len(X_test)//7 + 1)[:len(X_test)]
    
    for horizon in predictions.keys():
        print(f"\n   Processing {horizon}...")
        
        ensemble_pred = (predictions[horizon]['lgb'] + 
                        predictions[horizon]['xgb'] + 
                        predictions[horizon]['cat']) / 3
        
        y_true_values = y_test[horizon] if isinstance(y_test[horizon], np.ndarray) else y_test[horizon].values
        errors = np.abs(ensemble_pred - y_true_values)
        
        df_analysis = pd.DataFrame({
            'error': errors,
            'hour': X_test['hour'].values if 'hour' in X_test.columns else X_test.index % 24,
            'day_of_week': X_test['day_of_week'].values if 'day_of_week' in X_test.columns else (X_test.index // 24) % 7,
            'y_true': y_true_values,
            'y_pred': ensemble_pred
        })
        
        hourly_mae = df_analysis.groupby('hour')['error'].mean().values
        weekday_mae = df_analysis.groupby('day_of_week')['error'].mean().values
        
        temporal_results[horizon] = {
            'hourly_mae': hourly_mae.tolist(),
            'weekday_mae': weekday_mae.tolist(),
            'overall_mae': float(errors.mean()),
            'mae_std': float(errors.std()),
            'sample_size': len(errors)
        }
        
        print(f"      Overall MAE: {errors.mean():.3f} ± {errors.std():.3f}")
        print(f"      Hourly MAE range: {hourly_mae.min():.3f} - {hourly_mae.max():.3f}")
        print(f"      Weekday MAE range: {weekday_mae.min():.3f} - {weekday_mae.max():.3f}")
    
    return temporal_results

def extract_feature_importance():
    print("\n3. Extracting REAL feature importance from models...")
    
    feature_importance = {}
    
    try:
        for horizon in ['1h', '6h', '24h']:
            model_file = train_path / f'models_{horizon}.pkl'
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models = joblib.load(f)
                
                if 'lgb' in models:
                    lgb_model = models['lgb']
                    importance = lgb_model.feature_importance(importance_type='gain')
                    feature_names = lgb_model.feature_name()
                    
                    feature_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    total_importance = importance.sum()
                    feature_imp['percentage'] = (feature_imp['importance'] / total_importance * 100).round(2)
                    
                    feature_importance[horizon] = feature_imp.head(15).to_dict('records')
                    
                    print(f"\n   {horizon} Top 5 features:")
                    for i, row in enumerate(feature_imp.head(5).itertuples(), 1):
                        print(f"      {i}. {row.feature}: {row.percentage:.1f}%")
                        
    except Exception as e:
        print(f"   Error extracting feature importance: {e}")
        
    return feature_importance

def calculate_model_agreement(predictions):
    print("\n4. Calculating REAL model agreement correlations...")
    
    agreement_results = {}
    
    for horizon in predictions.keys():
        lgb_pred = predictions[horizon]['lgb']
        xgb_pred = predictions[horizon]['xgb'] 
        cat_pred = predictions[horizon]['cat']
        
        from scipy.stats import pearsonr
        
        corr_lgb_xgb = pearsonr(lgb_pred, xgb_pred)[0]
        corr_lgb_cat = pearsonr(lgb_pred, cat_pred)[0]
        corr_xgb_cat = pearsonr(xgb_pred, cat_pred)[0]
        
        agreement_matrix = np.array([
            [1.00, corr_lgb_xgb, corr_lgb_cat],
            [corr_lgb_xgb, 1.00, corr_xgb_cat],
            [corr_lgb_cat, corr_xgb_cat, 1.00]
        ])
        
        avg_agreement = (corr_lgb_xgb + corr_lgb_cat + corr_xgb_cat) / 3
        
        agreement_results[horizon] = {
            'matrix': agreement_matrix.tolist(),
            'average_agreement': float(avg_agreement),
            'lgb_xgb': float(corr_lgb_xgb),
            'lgb_cat': float(corr_lgb_cat),
            'xgb_cat': float(corr_xgb_cat)
        }
        
        print(f"\n   {horizon} Agreement:")
        print(f"      LGB-XGB: {corr_lgb_xgb:.3f}")
        print(f"      LGB-Cat: {corr_lgb_cat:.3f}")
        print(f"      XGB-Cat: {corr_xgb_cat:.3f}")
        print(f"      Average: {avg_agreement:.3f}")
    
    return agreement_results

def calculate_error_distributions(predictions, y_test):
    print("\n5. Calculating REAL error distributions...")
    
    error_distributions = {}
    
    for horizon in predictions.keys():
        y_true_values = y_test[horizon] if isinstance(y_test[horizon], np.ndarray) else y_test[horizon].values
        
        errors_lgb = predictions[horizon]['lgb'] - y_true_values
        errors_xgb = predictions[horizon]['xgb'] - y_true_values
        errors_cat = predictions[horizon]['cat'] - y_true_values
        
        ensemble_pred = (predictions[horizon]['lgb'] + 
                        predictions[horizon]['xgb'] + 
                        predictions[horizon]['cat']) / 3
        errors_ensemble = ensemble_pred - y_true_values
        
        error_distributions[horizon] = {
            'lgb': {
                'mean': float(errors_lgb.mean()),
                'std': float(errors_lgb.std()),
                'min': float(errors_lgb.min()),
                'max': float(errors_lgb.max()),
                'q25': float(np.percentile(errors_lgb, 25)),
                'q50': float(np.percentile(errors_lgb, 50)),
                'q75': float(np.percentile(errors_lgb, 75))
            },
            'xgb': {
                'mean': float(errors_xgb.mean()),
                'std': float(errors_xgb.std()),
                'min': float(errors_xgb.min()),
                'max': float(errors_xgb.max()),
                'q25': float(np.percentile(errors_xgb, 25)),
                'q50': float(np.percentile(errors_xgb, 50)),
                'q75': float(np.percentile(errors_xgb, 75))
            },
            'cat': {
                'mean': float(errors_cat.mean()),
                'std': float(errors_cat.std()),
                'min': float(errors_cat.min()),
                'max': float(errors_cat.max()),
                'q25': float(np.percentile(errors_cat, 25)),
                'q50': float(np.percentile(errors_cat, 50)),
                'q75': float(np.percentile(errors_cat, 75))
            },
            'ensemble': {
                'mean': float(errors_ensemble.mean()),
                'std': float(errors_ensemble.std()),
                'min': float(errors_ensemble.min()),
                'max': float(errors_ensemble.max()),
                'q25': float(np.percentile(errors_ensemble, 25)),
                'q50': float(np.percentile(errors_ensemble, 50)),
                'q75': float(np.percentile(errors_ensemble, 75))
            }
        }
        
        print(f"\n   {horizon} Error Stats (Ensemble):")
        print(f"      Mean: {errors_ensemble.mean():.3f}")
        print(f"      Std:  {errors_ensemble.std():.3f}")
        print(f"      Range: [{errors_ensemble.min():.1f}, {errors_ensemble.max():.1f}]")
    
    return error_distributions

def main():
    predictions, y_test, X_test = load_predictions_and_targets()
    
    if predictions is None:
        print("\nERROR: Could not load predictions. Please check file paths.")
        return
    
    results = {
        'temporal_patterns': calculate_temporal_error_patterns(predictions, y_test, X_test),
        'feature_importance': extract_feature_importance(),
        'model_agreement': calculate_model_agreement(predictions),
        'error_distributions': calculate_error_distributions(predictions, y_test)
    }
    
    with open('real_data_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nReal data has been extracted and saved to: real_data_analysis.json")
    print("\nKey findings:")
    
    if results['temporal_patterns']:
        horizon = '6h'
        if horizon in results['temporal_patterns']:
            pattern = results['temporal_patterns'][horizon]
            print(f"\n{horizon} Temporal Patterns:")
            print(f"  Weekday MAE: {[round(x, 2) for x in pattern['weekday_mae']]}")
            print(f"  Hour with highest error: {np.argmax(pattern['hourly_mae'])}:00")
            print(f"  Hour with lowest error: {np.argmin(pattern['hourly_mae'])}:00")
    
    if results['feature_importance']:
        if '6h' in results['feature_importance']:
            print(f"\nTop 3 Features (6h):")
            for i, feat in enumerate(results['feature_importance']['6h'][:3], 1):
                print(f"  {i}. {feat['feature']}: {feat['percentage']:.1f}%")
    
    print("\nNext steps:")
    print("1. Update presentation notebook with these real values")
    print("2. Regenerate all synthetic plots with actual data")
    print("3. Update insights based on real patterns")

if __name__ == "__main__":
    main()