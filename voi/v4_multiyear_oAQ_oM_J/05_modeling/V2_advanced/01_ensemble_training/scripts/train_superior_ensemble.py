#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import differential_evolution
import joblib
import pickle
import gc
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SUPERIOR GRADIENT BOOSTING ENSEMBLE")
print("Optimized to beat Linear Regression baseline on R²")
print("="*70)
print(f"Started: {datetime.now()}")

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

print("\nLoading preprocessed data...")
X_train = pd.read_pickle('../trained/X_train.pkl')
X_val = pd.read_pickle('../trained/X_val.pkl')
X_test = pd.read_pickle('../trained/X_test.pkl')

with open('../trained/y_data.pkl', 'rb') as f:
    y_data = pickle.load(f)
    y_train = y_data['y_train']
    y_val = y_data['y_val']
    y_test = y_data['y_test']

with open('../trained/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    feature_cols = metadata['feature_cols']
    target_cols = metadata['target_cols']

print(f"Train: {len(X_train):,} samples")
print(f"Val: {len(X_val):,} samples")
print(f"Test: {len(X_test):,} samples")
print(f"Features: {len(feature_cols)}")

def get_optimized_lgb_params(horizon_hours):
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'force_col_wise': True,
        'max_bin': 255,
        'min_data_in_bin': 3
    }
    
    if horizon_hours <= 3:
        base_params.update({
            'num_leaves': 511,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'min_child_samples': 5,
            'lambda_l1': 0.01,
            'lambda_l2': 10,
            'min_gain_to_split': 0.001,
            'max_depth': 12,
            'path_smooth': 10,
            'extra_trees': True
        })
        n_estimators = 3000
    elif horizon_hours <= 6:
        base_params.update({
            'num_leaves': 255,
            'learning_rate': 0.015,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'min_child_samples': 10,
            'lambda_l1': 0.05,
            'lambda_l2': 5,
            'min_gain_to_split': 0.005,
            'max_depth': 10,
            'path_smooth': 5
        })
        n_estimators = 2500
    elif horizon_hours <= 12:
        base_params.update({
            'num_leaves': 127,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 3,
            'min_gain_to_split': 0.01,
            'max_depth': 8,
            'path_smooth': 2
        })
        n_estimators = 2000
    else:
        base_params.update({
            'num_leaves': 63,
            'learning_rate': 0.025,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.75,
            'bagging_freq': 7,
            'min_child_samples': 30,
            'lambda_l1': 0.5,
            'lambda_l2': 2,
            'min_gain_to_split': 0.02,
            'max_depth': 6,
            'path_smooth': 1
        })
        n_estimators = 1500
    
    return base_params, n_estimators

def get_optimized_xgb_params(horizon_hours):
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'grow_policy': 'lossguide'
    }
    
    if horizon_hours <= 3:
        base_params.update({
            'max_depth': 12,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'min_child_weight': 3,
            'reg_alpha': 0.01,
            'reg_lambda': 10,
            'gamma': 0.001,
            'max_leaves': 511
        })
        n_estimators = 3000
    elif horizon_hours <= 6:
        base_params.update({
            'max_depth': 10,
            'learning_rate': 0.015,
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'min_child_weight': 5,
            'reg_alpha': 0.05,
            'reg_lambda': 5,
            'gamma': 0.005,
            'max_leaves': 255
        })
        n_estimators = 2500
    elif horizon_hours <= 12:
        base_params.update({
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 3,
            'gamma': 0.01,
            'max_leaves': 127
        })
        n_estimators = 2000
    else:
        base_params.update({
            'max_depth': 6,
            'learning_rate': 0.025,
            'subsample': 0.75,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'min_child_weight': 20,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'gamma': 0.02,
            'max_leaves': 63
        })
        n_estimators = 1500
    
    return base_params, n_estimators

def get_optimized_cat_params(horizon_hours):
    base_params = {
        'loss_function': 'RMSE',
        'random_state': 42,
        'verbose': False,
        'thread_count': -1,
        'grow_policy': 'Lossguide',
        'bootstrap_type': 'Bayesian',
        'task_type': 'CPU'
    }
    
    if horizon_hours <= 3:
        base_params.update({
            'iterations': 3000,
            'learning_rate': 0.01,
            'depth': 10,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 5,
            'random_strength': 0.3,
            'bagging_temperature': 0.9,
            'border_count': 254,
            'max_leaves': 511,
            'subsample': 0.9,
            'sampling_frequency': 'PerTree',
            'boosting_type': 'Plain'
        })
    elif horizon_hours <= 6:
        base_params.update({
            'iterations': 2500,
            'learning_rate': 0.015,
            'depth': 9,
            'l2_leaf_reg': 5,
            'min_data_in_leaf': 10,
            'random_strength': 0.4,
            'bagging_temperature': 0.7,
            'border_count': 200,
            'max_leaves': 255,
            'subsample': 0.85,
            'sampling_frequency': 'PerTree',
            'boosting_type': 'Plain'
        })
    elif horizon_hours <= 12:
        base_params.update({
            'iterations': 2000,
            'learning_rate': 0.02,
            'depth': 8,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'border_count': 128,
            'max_leaves': 127,
            'subsample': 0.8,
            'sampling_frequency': 'PerTreeLevel',
            'boosting_type': 'Plain'
        })
    else:
        base_params.update({
            'iterations': 1500,
            'learning_rate': 0.025,
            'depth': 6,
            'l2_leaf_reg': 2,
            'min_data_in_leaf': 30,
            'random_strength': 0.6,
            'bagging_temperature': 0.3,
            'border_count': 64,
            'max_leaves': 63,
            'subsample': 0.75,
            'sampling_frequency': 'PerTreeLevel',
            'boosting_type': 'Ordered'
        })
    
    return base_params

def optimize_ensemble_weights(val_preds, y_val, metric='r2'):
    def objective(weights):
        weights = weights / np.sum(weights)
        ensemble_pred = np.zeros_like(val_preds[0])
        for i, pred in enumerate(val_preds):
            ensemble_pred += weights[i] * pred
        
        if metric == 'r2':
            return -r2_score(y_val, ensemble_pred)
        else:
            return mean_squared_error(y_val, ensemble_pred)
    
    bounds = [(0.1, 0.8) for _ in range(len(val_preds))]
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=300,
        popsize=15,
        atol=1e-7,
        tol=1e-7
    )
    
    weights = result.x / np.sum(result.x)
    return weights

def train_horizon(horizon, horizon_hours):
    print(f"\n{'='*60}")
    print(f"Training {horizon} prediction (target R² > {baseline_targets[horizon]['r2']:.3f})")
    print(f"{'='*60}")
    
    y_tr = y_train[f'target_{horizon}']
    y_vl = y_val[f'target_{horizon}']
    y_te = y_test[f'target_{horizon}']
    
    models = {}
    val_predictions = {}
    test_predictions = {}
    
    print("\n1. Training LightGBM...")
    lgb_params, lgb_n_estimators = get_optimized_lgb_params(horizon_hours)
    
    train_set = lgb.Dataset(X_train, label=y_tr)
    valid_set = lgb.Dataset(X_val, label=y_vl, reference=train_set)
    
    lgb_model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=lgb_n_estimators,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_predictions['lgb'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_predictions['lgb'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    models['lgb'] = lgb_model
    
    lgb_r2 = r2_score(y_vl, val_predictions['lgb'])
    print(f"   LightGBM Val R²: {lgb_r2:.4f}")
    
    print("\n2. Training XGBoost...")
    xgb_params, xgb_n_estimators = get_optimized_xgb_params(horizon_hours)
    
    dtrain = xgb.DMatrix(X_train, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_vl)
    dtest = xgb.DMatrix(X_test)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_n_estimators,
        evals=[(dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    val_predictions['xgb'] = xgb_model.predict(dval)
    test_predictions['xgb'] = xgb_model.predict(dtest)
    models['xgb'] = xgb_model
    
    xgb_r2 = r2_score(y_vl, val_predictions['xgb'])
    print(f"   XGBoost Val R²: {xgb_r2:.4f}")
    
    print("\n3. Training CatBoost...")
    cat_params = get_optimized_cat_params(horizon_hours)
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_tr,
        eval_set=(X_val, y_vl),
        early_stopping_rounds=100,
        verbose=False
    )
    
    val_predictions['cat'] = cat_model.predict(X_val)
    test_predictions['cat'] = cat_model.predict(X_test)
    models['cat'] = cat_model
    
    cat_r2 = r2_score(y_vl, val_predictions['cat'])
    print(f"   CatBoost Val R²: {cat_r2:.4f}")
    
    print("\n4. Optimizing ensemble weights for R²...")
    val_preds_list = [val_predictions['lgb'], val_predictions['xgb'], val_predictions['cat']]
    
    weights_r2 = optimize_ensemble_weights(val_preds_list, y_vl, metric='r2')
    print(f"   R²-optimized weights: LGB={weights_r2[0]:.3f}, XGB={weights_r2[1]:.3f}, Cat={weights_r2[2]:.3f}")
    
    weights_mae = optimize_ensemble_weights(val_preds_list, y_vl, metric='mae')
    print(f"   MAE-optimized weights: LGB={weights_mae[0]:.3f}, XGB={weights_mae[1]:.3f}, Cat={weights_mae[2]:.3f}")
    
    best_weights = weights_r2 if horizon_hours <= 12 else weights_mae
    
    ensemble_val = np.zeros_like(val_predictions['lgb'])
    ensemble_test = np.zeros_like(test_predictions['lgb'])
    
    for i, model_name in enumerate(['lgb', 'xgb', 'cat']):
        ensemble_val += best_weights[i] * val_predictions[model_name]
        ensemble_test += best_weights[i] * test_predictions[model_name]
    
    ensemble_test = np.clip(ensemble_test, 0, 100)
    
    print("\n5. Final Test Performance:")
    test_results = {}
    
    for model_name in ['lgb', 'xgb', 'cat']:
        mae = mean_absolute_error(y_te, test_predictions[model_name])
        r2 = r2_score(y_te, test_predictions[model_name])
        test_results[model_name] = {'mae': mae, 'r2': r2}
        print(f"   {model_name.upper()}: MAE={mae:.3f}, R²={r2:.4f}")
    
    ensemble_mae = mean_absolute_error(y_te, ensemble_test)
    ensemble_r2 = r2_score(y_te, ensemble_test)
    test_results['ensemble'] = {'mae': ensemble_mae, 'r2': ensemble_r2}
    
    baseline = baseline_targets[horizon]
    mae_improvement = (baseline['mae'] - ensemble_mae) / baseline['mae'] * 100
    r2_improvement = (ensemble_r2 - baseline['r2']) / baseline['r2'] * 100
    
    print(f"\n   ENSEMBLE: MAE={ensemble_mae:.3f}, R²={ensemble_r2:.4f}")
    print(f"   Baseline: MAE={baseline['mae']:.3f}, R²={baseline['r2']:.4f}")
    print(f"   Improvement: MAE {mae_improvement:+.1f}%, R² {r2_improvement:+.1f}%")
    
    beat_mae = ensemble_mae < baseline['mae']
    beat_r2 = ensemble_r2 > baseline['r2']
    print(f"   Beat baseline: MAE={'✓' if beat_mae else '✗'}, R²={'✓' if beat_r2 else '✗'}")
    
    joblib.dump(models, f'../trained/models_{horizon}_superior.pkl')
    joblib.dump({
        'val': val_predictions,
        'test': test_predictions,
        'ensemble_test': ensemble_test,
        'weights': best_weights
    }, f'../trained/predictions_{horizon}_superior.pkl')
    
    return {
        'models': test_results,
        'ensemble': {'mae': ensemble_mae, 'r2': ensemble_r2},
        'baseline': baseline,
        'improvement': {'mae_pct': mae_improvement, 'r2_pct': r2_improvement},
        'beat_mae': beat_mae,
        'beat_r2': beat_r2,
        'weights': best_weights.tolist()
    }

print("\n" + "="*70)
print("TRAINING ALL HORIZONS")
print("="*70)

all_results = {}
horizons = ['1h', '2h', '3h', '4h', '5h', '6h', '12h', '24h']
horizon_hours_map = {'1h': 1, '2h': 2, '3h': 3, '4h': 4, '5h': 5, '6h': 6, '12h': 12, '24h': 24}

for horizon in horizons:
    horizon_hours = horizon_hours_map[horizon]
    results = train_horizon(horizon, horizon_hours)
    all_results[horizon] = results
    
    gc.collect()

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

beat_mae_count = sum(1 for r in all_results.values() if r['beat_mae'])
beat_r2_count = sum(1 for r in all_results.values() if r['beat_r2'])

print(f"\nBeat Linear Regression on MAE: {beat_mae_count}/8 horizons")
print(f"Beat Linear Regression on R²: {beat_r2_count}/8 horizons")

print("\nDetailed Results:")
print("-"*60)
print(f"{'Horizon':<10} {'Ens MAE':<10} {'Base MAE':<10} {'Ens R²':<10} {'Base R²':<10} {'Status'}")
print("-"*60)

for horizon in horizons:
    r = all_results[horizon]
    status = '✓✓' if r['beat_mae'] and r['beat_r2'] else '✓-' if r['beat_mae'] else '-✓' if r['beat_r2'] else '--'
    print(f"{horizon:<10} {r['ensemble']['mae']:<10.3f} {r['baseline']['mae']:<10.3f} "
          f"{r['ensemble']['r2']:<10.4f} {r['baseline']['r2']:<10.4f} {status}")

with open('../superior_ensemble_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to ../superior_ensemble_results.json")
print(f"Completed: {datetime.now()}")