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
from sklearn.metrics import mean_absolute_error, r2_score

def get_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def log_message(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def get_lgb_params(horizon):
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if horizon <= 6:
        base_params['num_leaves'] = 255
        base_params['learning_rate'] = 0.03
        base_params['min_child_samples'] = 5
    
    return base_params

def get_xgb_params(horizon):
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.01,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if horizon <= 6:
        base_params['max_depth'] = 10
        base_params['learning_rate'] = 0.03
        base_params['min_child_weight'] = 3
    
    return base_params

def get_cat_params(horizon):
    base_params = {
        'loss_function': 'RMSE',
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'min_data_in_leaf': 10,
        'random_strength': 0.5,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'grow_policy': 'SymmetricTree',
        'random_state': 42,
        'verbose': False,
        'thread_count': -1
    }
    
    if horizon <= 6:
        base_params['depth'] = 10
        base_params['learning_rate'] = 0.03
        base_params['min_data_in_leaf'] = 5
    
    return base_params

def load_progress():
    if os.path.exists('../training_progress.json'):
        with open('../training_progress.json', 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}

def save_progress(progress):
    with open('../training_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)

def train_models():
    log_message("Starting gradient boosting ensemble training")
    log_message(f"Initial memory: {get_memory_gb():.1f} GB")
    
    # Initialize progress file immediately
    progress = {'completed': [], 'failed': [], 'metrics': {}}
    save_progress(progress)
    log_message("Progress tracking initialized")
    
    # Load data
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
    
    # Load or initialize progress
    progress = load_progress()
    
    # Initialize storage
    models = {}
    predictions_val = {}
    predictions_test = {}
    
    total_horizons = len(target_cols)
    
    for idx, target_col in enumerate(target_cols):
        h = int(target_col.split('_')[1][:-1])
        horizon = f'{h}h'
        
        # Skip if already completed
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
        
        # Train LightGBM
        try:
            log_message(f"Training LightGBM for {horizon}...")
            start_time = time.time()
            
            lgb_train = lgb.Dataset(X_train, label=y_tr, categorical_feature=categorical_features)
            lgb_val = lgb.Dataset(X_val, label=y_vl, reference=lgb_train, categorical_feature=categorical_features)
            
            lgb_model = lgb.train(
                get_lgb_params(h),
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            models[horizon]['lgb'] = lgb_model
            predictions_val[horizon]['lgb'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            predictions_test[horizon]['lgb'] = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
            
            # Calculate metrics
            val_mae = mean_absolute_error(y_vl, predictions_val[horizon]['lgb'])
            val_r2 = r2_score(y_vl, predictions_val[horizon]['lgb'])
            
            elapsed = time.time() - start_time
            log_message(f"✓ LightGBM complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f}, Memory: {get_memory_gb():.1f} GB)")
            
        except Exception as e:
            log_message(f"✗ LightGBM failed: {str(e)}", "ERROR")
            predictions_val[horizon]['lgb'] = np.zeros(len(y_vl))
            predictions_test[horizon]['lgb'] = np.zeros(len(y_te))
        
        # Train XGBoost
        try:
            log_message(f"Training XGBoost for {horizon}...")
            start_time = time.time()
            
            dtrain = xgb.DMatrix(X_train, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_vl)
            dtest = xgb.DMatrix(X_test)
            
            xgb_model = xgb.train(
                get_xgb_params(h),
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            models[horizon]['xgb'] = xgb_model
            predictions_val[horizon]['xgb'] = xgb_model.predict(dval)
            predictions_test[horizon]['xgb'] = xgb_model.predict(dtest)
            
            # Calculate metrics
            val_mae = mean_absolute_error(y_vl, predictions_val[horizon]['xgb'])
            val_r2 = r2_score(y_vl, predictions_val[horizon]['xgb'])
            
            del dtrain, dval, dtest
            gc.collect()
            
            elapsed = time.time() - start_time
            log_message(f"✓ XGBoost complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f}, Memory: {get_memory_gb():.1f} GB)")
            
        except Exception as e:
            log_message(f"✗ XGBoost failed: {str(e)}", "ERROR")
            predictions_val[horizon]['xgb'] = np.zeros(len(y_vl))
            predictions_test[horizon]['xgb'] = np.zeros(len(y_te))
        
        # Train CatBoost
        try:
            log_message(f"Training CatBoost for {horizon}...")
            start_time = time.time()
            
            cat_features = [X_train.columns.get_loc('hex_encoded')]
            
            cat_model = CatBoostRegressor(**get_cat_params(h))
            cat_model.fit(
                X_train, y_tr,
                eval_set=(X_val, y_vl),
                cat_features=cat_features,
                early_stopping_rounds=50,
                verbose=False
            )
            
            models[horizon]['cat'] = cat_model
            predictions_val[horizon]['cat'] = cat_model.predict(X_val)
            predictions_test[horizon]['cat'] = cat_model.predict(X_test)
            
            # Calculate metrics
            val_mae = mean_absolute_error(y_vl, predictions_val[horizon]['cat'])
            val_r2 = r2_score(y_vl, predictions_val[horizon]['cat'])
            
            elapsed = time.time() - start_time
            log_message(f"✓ CatBoost complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f}, Memory: {get_memory_gb():.1f} GB)")
            
        except Exception as e:
            log_message(f"✗ CatBoost failed: {str(e)}", "ERROR")
            predictions_val[horizon]['cat'] = np.zeros(len(y_vl))
            predictions_test[horizon]['cat'] = np.zeros(len(y_te))
        
        # Save models and predictions
        try:
            log_message(f"Saving models for {horizon}...")
            joblib.dump(models[horizon], f'../trained/models_{horizon}.pkl')
            joblib.dump({
                'val': predictions_val[horizon],
                'test': predictions_test[horizon]
            }, f'../trained/predictions_{horizon}.pkl')
            log_message(f"✓ Saved to trained/models_{horizon}.pkl and predictions_{horizon}.pkl")
        except Exception as e:
            log_message(f"✗ Failed to save: {str(e)}", "ERROR")
        
        # Calculate ensemble metrics (simple average for preview)
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
                
                log_message(f"Ensemble {horizon}: MAE={ensemble_mae:.3f} (baseline={baseline_mae:.3f}, {mae_improvement:+.1f}%)")
                log_message(f"Ensemble {horizon}: R²={ensemble_r2:.3f} (baseline={baseline_r2:.3f}, {r2_improvement:+.1f}%)")
                
                progress['metrics'][horizon] = {
                    'ensemble_mae': ensemble_mae,
                    'ensemble_r2': ensemble_r2,
                    'baseline_mae': baseline_mae,
                    'baseline_r2': baseline_r2,
                    'mae_improvement': mae_improvement,
                    'r2_improvement': r2_improvement
                }
        
        # Update progress
        progress['completed'].append(horizon)
        save_progress(progress)
        
        # Cleanup
        gc.collect()
        log_message(f"Completed {horizon}. Progress: {idx+1}/{total_horizons}")
    
    log_message("="*60)
    log_message(f"Training complete! Processed {len(progress['completed'])} horizons")
    log_message(f"Final memory: {get_memory_gb():.1f} GB")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(train_models())
    except KeyboardInterrupt:
        log_message("Training interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"Fatal error: {str(e)}", "ERROR")
        sys.exit(1)