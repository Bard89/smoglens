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
import shutil
from datetime import datetime
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
CHECKPOINT_DIR = 'catboost_checkpoints'
PROGRESS_FILE = '../training_progress_granular.json'

def get_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def log_message(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def load_granular_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'horizons': {},
        'global_status': {
            'total': 24,
            'completed': 0,
            'in_progress': 0,
            'pending': 24,
            'failed': 0
        },
        'start_time': datetime.now().isoformat()
    }

def save_granular_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def update_model_status(progress, horizon, model_type, status, **kwargs):
    if horizon not in progress['horizons']:
        progress['horizons'][horizon] = {}
    
    if model_type not in progress['horizons'][horizon]:
        progress['horizons'][horizon][model_type] = {}
    
    progress['horizons'][horizon][model_type]['status'] = status
    progress['horizons'][horizon][model_type].update(kwargs)
    
    if status == 'completed':
        progress['horizons'][horizon][model_type]['completed_at'] = datetime.now().strftime("%H:%M:%S")
    elif status == 'training':
        progress['horizons'][horizon][model_type]['started_at'] = datetime.now().strftime("%H:%M:%S")
    elif status == 'failed':
        progress['horizons'][horizon][model_type]['failed_at'] = datetime.now().strftime("%H:%M:%S")
    
    recalculate_global_status(progress)
    save_granular_progress(progress)

def recalculate_global_status(progress):
    total_models = 0
    completed = 0
    in_progress = 0
    failed = 0
    
    for horizon in progress['horizons'].values():
        for model in horizon.values():
            total_models += 1
            if model.get('status') == 'completed':
                completed += 1
            elif model.get('status') == 'training':
                in_progress += 1
            elif model.get('status') == 'failed':
                failed += 1
    
    pending = max(0, 24 - total_models)
    progress['global_status'] = {
        'total': 24,
        'completed': completed,
        'in_progress': in_progress,
        'pending': pending,
        'failed': failed
    }

def check_existing_model(horizon, model_type):
    model_file = f'../trained/model_{horizon}_{model_type}.pkl'
    pred_file = f'../trained/predictions_{horizon}_{model_type}.pkl'
    
    if os.path.exists(model_file) and os.path.exists(pred_file):
        try:
            model = joblib.load(model_file)
            preds = joblib.load(pred_file)
            log_message(f"Found existing {model_type.upper()} model for {horizon}, skipping training")
            return model, preds['val'], preds['test']
        except Exception as e:
            log_message(f"Failed to load existing {model_type} for {horizon}: {e}", "WARNING")
    return None, None, None

def save_individual_model(horizon, model_type, model, pred_val, pred_test):
    try:
        model_file = f'../trained/model_{horizon}_{model_type}.pkl'
        pred_file = f'../trained/predictions_{horizon}_{model_type}.pkl'
        
        joblib.dump(model, model_file, compress=3)
        joblib.dump({'val': pred_val, 'test': pred_test}, pred_file, compress=3)
        
        log_message(f"✓ Saved {model_type.upper()} for {horizon} to {model_file}")
        return True
    except Exception as e:
        log_message(f"✗ Failed to save {model_type} for {horizon}: {e}", "ERROR")
        return False

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
        'max_leaves': 127,
        'save_snapshot': True,
        'snapshot_interval': 200
    }
    
    if horizon <= 6:
        base_params['depth'] = 10
        base_params['learning_rate'] = 0.03
        base_params['min_data_in_leaf'] = 15
        base_params['max_leaves'] = 255
    
    return base_params

def train_lgb_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, categorical_features, progress = args
    
    existing_model, existing_val, existing_test = check_existing_model(horizon, 'lgb')
    if existing_model is not None:
        update_model_status(progress, horizon, 'lgb', 'completed', source='cached')
        return horizon, 'lgb', existing_model, existing_val, existing_test
    
    try:
        update_model_status(progress, horizon, 'lgb', 'training')
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
        
        if save_individual_model(horizon, 'lgb', lgb_model, pred_val, pred_test):
            update_model_status(progress, horizon, 'lgb', 'completed', 
                              mae=val_mae, r2=val_r2, time=elapsed, 
                              iterations=lgb_model.best_iteration)
            log_message(f"✓ LightGBM {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
        else:
            update_model_status(progress, horizon, 'lgb', 'failed', error='save_failed')
        
        return horizon, 'lgb', lgb_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ LightGBM {horizon} failed: {str(e)}", "ERROR")
        update_model_status(progress, horizon, 'lgb', 'failed', error=str(e))
        return horizon, 'lgb', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def train_xgb_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, _, progress = args
    
    existing_model, existing_val, existing_test = check_existing_model(horizon, 'xgb')
    if existing_model is not None:
        update_model_status(progress, horizon, 'xgb', 'completed', source='cached')
        return horizon, 'xgb', existing_model, existing_val, existing_test
    
    try:
        update_model_status(progress, horizon, 'xgb', 'training')
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
        
        if save_individual_model(horizon, 'xgb', xgb_model, pred_val, pred_test):
            update_model_status(progress, horizon, 'xgb', 'completed',
                              mae=val_mae, r2=val_r2, time=elapsed,
                              iterations=xgb_model.best_iteration)
            log_message(f"✓ XGBoost {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
        else:
            update_model_status(progress, horizon, 'xgb', 'failed', error='save_failed')
        
        return horizon, 'xgb', xgb_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ XGBoost {horizon} failed: {str(e)}", "ERROR")
        update_model_status(progress, horizon, 'xgb', 'failed', error=str(e))
        return horizon, 'xgb', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def train_cat_model(args):
    X_train, X_val, y_train, y_val, X_test, horizon, h, categorical_features, progress = args
    
    existing_model, existing_val, existing_test = check_existing_model(horizon, 'cat')
    if existing_model is not None:
        update_model_status(progress, horizon, 'cat', 'completed', source='cached')
        return horizon, 'cat', existing_model, existing_val, existing_test
    
    try:
        update_model_status(progress, horizon, 'cat', 'training')
        log_message(f"Training CatBoost for {horizon}...")
        start_time = time.time()
        
        cat_features = [X_train.columns.get_loc('hex_encoded')]
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, horizon)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        params = get_cat_params_m4(h)
        params['snapshot_file'] = os.path.join(checkpoint_path, 'snapshot.cbsnapshot')
        
        snapshot_exists = os.path.exists(params['snapshot_file'])
        if snapshot_exists:
            log_message(f"Found CatBoost checkpoint for {horizon}, resuming...")
        
        cat_model = CatBoostRegressor(**params)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=False,
            plot=False,
            use_best_model=True
        )
        
        pred_val = cat_model.predict(X_val)
        pred_test = cat_model.predict(X_test)
        
        val_mae = mean_absolute_error(y_val, pred_val)
        val_r2 = r2_score(y_val, pred_val)
        
        elapsed = time.time() - start_time
        
        if save_individual_model(horizon, 'cat', cat_model, pred_val, pred_test):
            update_model_status(progress, horizon, 'cat', 'completed',
                              mae=val_mae, r2=val_r2, time=elapsed,
                              iterations=cat_model.best_iteration_)
            log_message(f"✓ CatBoost {horizon} complete ({elapsed:.1f}s, MAE={val_mae:.3f}, R²={val_r2:.3f})")
            
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
        else:
            update_model_status(progress, horizon, 'cat', 'failed', error='save_failed')
        
        return horizon, 'cat', cat_model, pred_val, pred_test
        
    except Exception as e:
        log_message(f"✗ CatBoost {horizon} failed: {str(e)}", "ERROR")
        update_model_status(progress, horizon, 'cat', 'failed', error=str(e))
        return horizon, 'cat', None, np.zeros(len(y_val)), np.zeros(len(X_test))

def combine_horizon_models(horizon, models, predictions_val, predictions_test):
    try:
        combined_models_file = f'../trained/models_{horizon}.pkl'
        combined_preds_file = f'../trained/predictions_{horizon}.pkl'
        
        joblib.dump(models[horizon], combined_models_file, compress=3)
        joblib.dump({
            'val': predictions_val[horizon],
            'test': predictions_test[horizon]
        }, combined_preds_file, compress=3)
        
        log_message(f"✓ Created combined files: {combined_models_file}, {combined_preds_file}")
        
        for model_type in ['lgb', 'xgb', 'cat']:
            individual_model = f'../trained/model_{horizon}_{model_type}.pkl'
            individual_pred = f'../trained/predictions_{horizon}_{model_type}.pkl'
            if os.path.exists(individual_model):
                os.remove(individual_model)
            if os.path.exists(individual_pred):
                os.remove(individual_pred)
        
        return True
    except Exception as e:
        log_message(f"Failed to combine models for {horizon}: {e}", "ERROR")
        return False

def train_models_granular():
    log_message("="*70)
    log_message("GRANULAR TRAINING WITH INTERMEDIATE SAVES")
    log_message("="*70)
    log_message(f"Initial memory: {get_memory_gb():.1f} GB")
    log_message(f"Using {OPTIMAL_THREADS} threads per model")
    
    progress = load_granular_progress()
    log_message(f"Progress tracking initialized: {PROGRESS_FILE}")
    
    if progress['global_status']['completed'] > 0:
        log_message(f"Resuming training: {progress['global_status']['completed']} models already completed")
    
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
    
    models = {}
    predictions_val = {}
    predictions_test = {}
    
    total_horizons = len(target_cols)
    
    for idx, target_col in enumerate(target_cols):
        h = int(target_col.split('_')[1][:-1])
        horizon = f'{h}h'
        
        print(f"\n{'='*60}")
        log_message(f"HORIZON {idx+1}/{total_horizons}: {horizon} prediction")
        print(f"{'='*60}")
        
        all_models_complete = True
        if horizon in progress['horizons']:
            for model_type in ['lgb', 'xgb', 'cat']:
                if model_type not in progress['horizons'][horizon] or \
                   progress['horizons'][horizon][model_type].get('status') != 'completed':
                    all_models_complete = False
                    break
        else:
            all_models_complete = False
        
        if all_models_complete:
            log_message(f"All models for {horizon} already completed, loading...")
            models[horizon] = {}
            predictions_val[horizon] = {}
            predictions_test[horizon] = {}
            
            for model_type in ['lgb', 'xgb', 'cat']:
                model, pred_val, pred_test = check_existing_model(horizon, model_type)
                if model is not None:
                    models[horizon][model_type] = model
                    predictions_val[horizon][model_type] = pred_val
                    predictions_test[horizon][model_type] = pred_test
            
            combine_horizon_models(horizon, models, predictions_val, predictions_test)
            continue
        
        log_message(f"Memory: {get_memory_gb():.1f} GB")
        
        y_tr = y_train[target_col]
        y_vl = y_val[target_col]
        y_te = y_test[target_col]
        
        models[horizon] = {}
        predictions_val[horizon] = {}
        predictions_test[horizon] = {}
        
        args_lgb = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, categorical_features, progress)
        args_xgb = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, None, progress)
        args_cat = (X_train, X_val, y_tr, y_vl, X_test, horizon, h, categorical_features, progress)
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(train_lgb_model, args_lgb): 'lgb',
                executor.submit(train_xgb_model, args_xgb): 'xgb',
                executor.submit(train_cat_model, args_cat): 'cat'
            }
            
            for future in as_completed(futures):
                try:
                    horizon_result, model_type, model, pred_val, pred_test = future.result()
                    if model is not None:
                        models[horizon][model_type] = model
                        predictions_val[horizon][model_type] = pred_val
                        predictions_test[horizon][model_type] = pred_test
                except Exception as e:
                    model_type = futures[future]
                    log_message(f"Failed to train {model_type} for {horizon}: {str(e)}", "ERROR")
                    update_model_status(progress, horizon, model_type, 'failed', error=str(e))
        
        if all(k in predictions_val[horizon] for k in ['lgb', 'xgb', 'cat']):
            ensemble_pred = (predictions_val[horizon]['lgb'] + 
                           predictions_val[horizon]['xgb'] + 
                           predictions_val[horizon]['cat']) / 3
            ensemble_mae = mean_absolute_error(y_vl, ensemble_pred)
            ensemble_r2 = r2_score(y_vl, ensemble_pred)
            
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
                
                if horizon not in progress['horizons']:
                    progress['horizons'][horizon] = {}
                progress['horizons'][horizon]['ensemble'] = {
                    'mae': ensemble_mae,
                    'r2': ensemble_r2,
                    'baseline_mae': baseline_mae,
                    'baseline_r2': baseline_r2,
                    'mae_improvement': mae_improvement,
                    'r2_improvement': r2_improvement
                }
                save_granular_progress(progress)
            
            combine_horizon_models(horizon, models, predictions_val, predictions_test)
        
        gc.collect()
        log_message(f"Completed {horizon}. Progress: {idx+1}/{total_horizons}")
        log_message(f"Memory after cleanup: {get_memory_gb():.1f} GB")
    
    log_message("="*60)
    log_message(f"Training complete! Status:")
    log_message(f"  Completed: {progress['global_status']['completed']}")
    log_message(f"  Failed: {progress['global_status']['failed']}")
    log_message(f"  Pending: {progress['global_status']['pending']}")
    log_message(f"Final memory: {get_memory_gb():.1f} GB")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(train_models_granular())
    except KeyboardInterrupt:
        log_message("Training interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"Fatal error: {str(e)}", "ERROR")
        sys.exit(1)