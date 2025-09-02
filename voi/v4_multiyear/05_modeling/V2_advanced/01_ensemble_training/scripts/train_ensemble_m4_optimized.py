#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import pickle
import gc
import json
import os
import sys
import time
import psutil
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['VECLIB_MAXIMUM_THREADS'] = '10'

OPTIMAL_THREADS = 10
MAX_MEMORY_GB = 40

def get_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def log_message(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def get_lgb_params_m4(horizon):
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'min_child_samples': 20,
        'lambda_l1': 0.05,
        'lambda_l2': 0.05,
        'min_gain_to_split': 0.001,
        'verbose': -1,
        'random_state': 42,
        'num_threads': OPTIMAL_THREADS,
        'device_type': 'cpu',
        'force_row_wise': True,
        'histogram_pool_size': 1024,
        'max_bin': 255
    }
    
    if horizon <= 6:
        base_params['num_leaves'] = 255
        base_params['learning_rate'] = 0.03
        base_params['min_child_samples'] = 15
        base_params['boosting_type'] = 'gbdt'
    else:
        base_params['boosting_type'] = 'dart'
        base_params['drop_rate'] = 0.1
        base_params['max_drop'] = 50
    
    return base_params

def get_xgb_params_m4(horizon):
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cpu',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 10,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'gamma': 0.001,
        'random_state': 42,
        'nthread': OPTIMAL_THREADS,
        'max_bin': 256,
        'grow_policy': 'lossguide',
        'max_leaves': 127
    }
    
    if horizon <= 6:
        base_params['max_depth'] = 10
        base_params['learning_rate'] = 0.03
        base_params['min_child_weight'] = 5
        base_params['max_leaves'] = 255
    
    return base_params

def get_cat_params_m4(horizon):
    base_params = {
        'loss_function': 'RMSE',
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'min_data_in_leaf': 20,
        'random_strength': 0.3,
        'bagging_temperature': 0.3,
        'border_count': 254,
        'grow_policy': 'Lossguide',
        'random_state': 42,
        'verbose': False,
        'thread_count': OPTIMAL_THREADS,
        'task_type': 'CPU',
        'devices': '0-9',
        'bootstrap_type': 'MVS',
        'sampling_frequency': 'PerTree',
        'max_leaves': 127
    }
    
    if horizon <= 6:
        base_params['depth'] = 10
        base_params['learning_rate'] = 0.03
        base_params['min_data_in_leaf'] = 15
        base_params['max_leaves'] = 255
    
    return base_params

def train_lgb_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, categorical_features = args
    
    try:
        log_message(f"Training LightGBM for {horizon}...")
        start_time = time.time()
        
        lgb_train = lgb.Dataset(
            X_train, label=y_train, 
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        lgb_val = lgb.Dataset(
            X_val, label=y_val, 
            reference=lgb_train, 
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        
        lgb_model = lgb.train(
            get_lgb_params_m4(h),
            lgb_train,
            num_boost_round=1500,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        
        val_mae = mean_absolute_error(y_val, pred_val)
        val_r2 = r2_score(y_val, pred_val)
        
        elapsed = time.time() - start_time
        log_message(f"✓ LightGBM {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
        
        return horizon, 'lgb', lgb_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ LightGBM {horizon} failed: {str(e)}", "ERROR")
        return horizon, 'lgb', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def train_xgb_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, _ = args
    
    try:
        log_message(f"Training XGBoost for {horizon}...")
        start_time = time.time()
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(
            get_xgb_params_m4(h),
            dtrain,
            num_boost_round=1500,
            evals=[(dval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        pred_val = xgb_model.predict(dval)
        pred_test = xgb_model.predict(dtest)
        
        val_mae = mean_absolute_error(y_val, pred_val)
        val_r2 = r2_score(y_val, pred_val)
        
        del dtrain, dval, dtest
        gc.collect()
        
        elapsed = time.time() - start_time
        log_message(f"✓ XGBoost {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
        
        return horizon, 'xgb', xgb_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ XGBoost {horizon} failed: {str(e)}", "ERROR")
        return horizon, 'xgb', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def train_cat_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, categorical_features = args
    
    try:
        log_message(f"Training CatBoost for {horizon}...")
        start_time = time.time()
        
        cat_features = [X_train.columns.get_loc('hex_encoded')]
        
        cat_model = CatBoostRegressor(**get_cat_params_m4(h))
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=False,
            plot=False
        )
        
        pred_val = cat_model.predict(X_val)
        pred_test = cat_model.predict(X_test)
        
        val_mae = mean_absolute_error(y_val, pred_val)
        val_r2 = r2_score(y_val, pred_val)
        
        elapsed = time.time() - start_time
        log_message(f"✓ CatBoost {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
        
        return horizon, 'cat', cat_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ CatBoost {horizon} failed: {str(e)}", "ERROR")
        return horizon, 'cat', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def load_progress():
    if os.path.exists('../training_progress.json'):
        with open('../training_progress.json', 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}

def save_progress(progress):
    with open('../training_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)

def train_models_parallel():
    log_message("Starting M4 Pro optimized gradient boosting ensemble training")
    log_message(f"Initial memory: {get_memory_gb():.1f} GB")
    log_message(f"Using {OPTIMAL_THREADS} threads per model, {cpu_count()} cores available")
    
    # Initialize progress file immediately
    progress = {'completed': [], 'failed': [], 'metrics': {}}
    save_progress(progress)
    log_message("Progress tracking initialized")
    
    log_message("Loading training data...")
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
    target_cols = metadata['target_cols']
    categorical_features = metadata['categorical_features']
    
    log_message(f"Data loaded: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    log_message(f"Memory after load: {get_memory_gb():.1f} GB")
    
    # Baseline targets for comparison
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
    
    progress = load_progress()
    
    models = {}
    predictions_val = {}
    predictions_test = {}
    
    total_horizons = len(target_cols)
    
    for idx, target_col in enumerate(target_cols):
        h = int(target_col.split('_')[1][:-1])
        horizon = f'{h}h'
        
        if horizon in progress['completed']:
            log_message(f"Skipping {horizon} (already completed)")
            continue
        
        print(f"\n{'='*60}")
        log_message(f"HORIZON {idx+1}/{total_horizons}: {horizon} prediction")
        print(f"{'='*60}")
        log_message(f"Memory: {get_memory_gb():.1f} GB")
        
        y_tr = y_train[target_col]
        y_vl = y_val[target_col]
        y_te = y_test[target_col]
        
        models[horizon] = {}
        predictions_val[horizon] = {}
        predictions_test[horizon] = {}
        
        args_lgb = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, categorical_features)
        args_xgb = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, None)
        args_cat = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, categorical_features)
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(train_lgb_model, args_lgb): 'lgb',
                executor.submit(train_xgb_model, args_xgb): 'xgb',
                executor.submit(train_cat_model, args_cat): 'cat'
            }
            
            for future in as_completed(futures):
                try:
                    horizon_result, model_type, model, pred_val, pred_test = future.result()
                    models[horizon][model_type] = model
                    predictions_val[horizon][model_type] = pred_val
                    predictions_test[horizon][model_type] = pred_test
                except Exception as e:
                    model_type = futures[future]
                    log_message(f"Failed to train {model_type} for {horizon}: {str(e)}", "ERROR")
                    models[horizon][model_type] = None
                    predictions_val[horizon][model_type] = np.zeros(len(y_vl))
                    predictions_test[horizon][model_type] = np.zeros(len(y_te))
        
        try:
            log_message(f"Saving models for {horizon}...")
            joblib.dump(models[horizon], f'../trained/models_{horizon}.pkl', compress=3)
            joblib.dump({
                'val': predictions_val[horizon],
                'test': predictions_test[horizon]
            }, f'../trained/predictions_{horizon}.pkl', compress=3)
            log_message(f"✓ Saved to trained/models_{horizon}.pkl and predictions_{horizon}.pkl")
        except Exception as e:
            log_message(f"✗ Failed to save: {str(e)}", "ERROR")
        
        # Calculate and display ensemble metrics
        if all(k in predictions_val[horizon] for k in ['lgb', 'xgb', 'cat']):
            ensemble_pred = (predictions_val[horizon]['lgb'] + 
                           predictions_val[horizon]['xgb'] + 
                           predictions_val[horizon]['cat']) / 3
            ensemble_mae = mean_absolute_error(y_vl, ensemble_pred)
            ensemble_r2 = r2_score(y_vl, ensemble_pred)
            
            # Compare with baseline
            if horizon in baseline_targets:
                baseline_mae = baseline_targets[horizon]['mae']
                baseline_r2 = baseline_targets[horizon]['r2']
                mae_improvement = (baseline_mae - ensemble_mae) / baseline_mae * 100
                r2_improvement = (ensemble_r2 - baseline_r2) / baseline_r2 * 100
                
                log_message(f"="*60)
                log_message(f"Ensemble {horizon}: MAE={ensemble_mae:.3f} (baseline={baseline_mae:.3f}, {mae_improvement:+.1f}%)")
                log_message(f"Ensemble {horizon}: R²={ensemble_r2:.3f} (baseline={baseline_r2:.3f}, {r2_improvement:+.1f}%)")
                beats_baseline = "✓" if ensemble_mae < baseline_mae else "✗"
                log_message(f"Beats Linear Regression: {beats_baseline}")
                
                progress['metrics'][horizon] = {
                    'ensemble_mae': ensemble_mae,
                    'ensemble_r2': ensemble_r2,
                    'baseline_mae': baseline_mae,
                    'baseline_r2': baseline_r2,
                    'mae_improvement': mae_improvement,
                    'r2_improvement': r2_improvement
                }
        
        progress['completed'].append(horizon)
        save_progress(progress)
        
        gc.collect()
        log_message(f"Completed {horizon}. Progress: {idx+1}/{total_horizons}")
        log_message(f"Memory after cleanup: {get_memory_gb():.1f} GB")
    
    log_message("="*60)
    log_message(f"Training complete! Processed {len(progress['completed'])} horizons")
    log_message(f"Final memory: {get_memory_gb():.1f} GB")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(train_models_parallel())
    except KeyboardInterrupt:
        log_message("Training interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"Fatal error: {str(e)}", "ERROR")
        sys.exit(1)