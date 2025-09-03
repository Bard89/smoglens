import pandas as pd
import numpy as np
from datetime import timedelta

def prepare_ensemble_features(data, current_time):
    """Prepare features for ensemble model (69 features)"""
    latest_idx = data[data['timestamp'] <= current_time].index[-1] if len(data[data['timestamp'] <= current_time]) > 0 else len(data) - 1
    latest_row = data.iloc[latest_idx]
    
    features = {}
    
    features['avg_traffic_volume'] = latest_row.get('avg_traffic_volume', 0.0)
    features['max_traffic_volume'] = 0.0
    features['congestion_index'] = latest_row.get('congestion_index', 0.0)
    features['traffic_measurement_count'] = 0.0
    features['traffic_distance_km'] = 0.0
    features['traffic_intensity'] = 0.0
    
    features['temperature_c_mean'] = latest_row.get('temperature_c_mean', 15.0)
    features['humidity_pct_mean'] = latest_row.get('humidity_pct_mean', 70.0)
    features['pressure_hpa_mean'] = latest_row.get('pressure_hpa_mean', 1013.0)
    features['cloud_cover_pct_mean'] = 50.0
    
    features['is_weekend'] = latest_row.get('is_weekend', 0)
    features['hour_sin'] = latest_row.get('hour_sin', 0)
    features['hour_cos'] = latest_row.get('hour_cos', 0)
    features['dow_sin'] = latest_row.get('dow_sin', 0)
    features['dow_cos'] = latest_row.get('dow_cos', 0)
    features['month_sin'] = latest_row.get('month_sin', 0)
    features['month_cos'] = latest_row.get('month_cos', 0)
    
    for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168]:
        old_col = f'pm25_lag_{lag}h'
        new_col = f'lag_{lag}h'
        features[new_col] = latest_row.get(old_col, latest_row['pm25_ugm3_mean'])
    
    features['diff_1h'] = features['lag_1h'] - latest_row['pm25_ugm3_mean']
    features['diff_6h'] = features['lag_6h'] - latest_row['pm25_ugm3_mean']
    features['diff_24h'] = features['lag_24h'] - latest_row['pm25_ugm3_mean']
    features['rate_6h'] = features['diff_6h'] / 6
    features['rate_24h'] = features['diff_24h'] / 24
    
    for window in [3, 6, 12, 24, 48]:
        for stat in ['mean', 'std', 'max', 'min']:
            new_col = f'rolling_{stat}_{window}h'
            if window <= 24 and stat in ['mean', 'std']:
                old_col = f'pm25_rolling_{stat}_{window}h'
                features[new_col] = latest_row.get(old_col, latest_row['pm25_ugm3_mean'] if stat == 'mean' else 1.0)
            else:
                if stat == 'mean' or stat == 'max':
                    features[new_col] = latest_row['pm25_ugm3_mean']
                elif stat == 'min':
                    features[new_col] = latest_row['pm25_ugm3_mean'] * 0.8
                else:
                    features[new_col] = 2.0
    
    features['ewm_0.1'] = latest_row['pm25_ugm3_mean']
    features['ewm_0.3'] = latest_row['pm25_ugm3_mean']
    features['ewm_0.5'] = latest_row['pm25_ugm3_mean']
    
    features['hour_sin_1'] = features['hour_sin']
    features['hour_cos_1'] = features['hour_cos']
    features['dow_sin_1'] = features['dow_sin']
    features['dow_cos_1'] = features['dow_cos']
    features['hour_sin_2'] = np.sin(4 * np.pi * latest_row.get('hour', current_time.hour) / 24)
    features['hour_cos_2'] = np.cos(4 * np.pi * latest_row.get('hour', current_time.hour) / 24)
    features['dow_sin_2'] = np.sin(4 * np.pi * latest_row.get('day_of_week', current_time.dayofweek) / 7)
    features['dow_cos_2'] = np.cos(4 * np.pi * latest_row.get('day_of_week', current_time.dayofweek) / 7)
    
    features['temp_humidity'] = features['temperature_c_mean'] * features['humidity_pct_mean'] / 100
    features['temp_hour'] = features['temperature_c_mean'] * features['hour_sin_1']
    features['traffic_hour'] = features['avg_traffic_volume'] * features['hour_sin_1']
    features['traffic_weekend'] = features['avg_traffic_volume'] * features['is_weekend']
    
    features['hex_encoded'] = 0
    
    return pd.DataFrame([features])

def predict_with_ensemble(models, data, current_time, hours_ahead=6):
    """Generate predictions using ensemble models"""
    predictions = []
    
    for h in range(1, hours_ahead + 1):
        features = prepare_ensemble_features(data, current_time)
        pred_time = current_time + timedelta(hours=h)
        features['hour_sin'] = np.sin(2 * np.pi * pred_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * pred_time.hour / 24)
        features['hour_sin_1'] = features['hour_sin']
        features['hour_cos_1'] = features['hour_cos']
        features['hour_sin_2'] = np.sin(4 * np.pi * pred_time.hour / 24)
        features['hour_cos_2'] = np.cos(4 * np.pi * pred_time.hour / 24)
        
        model_key = f'{h}h'
        if model_key in models:
            model = models[model_key]
        elif h <= 6 and '6h' in models:
            model = models['6h']
        elif '12h' in models:
            model = models['12h']
        else:
            model = list(models.values())[0] if models else None
        
        if model and isinstance(model, dict):
            preds = []
            if 'xgb' in model:
                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(features)
                    pred = model['xgb'].predict(dmatrix)[0]
                    preds.append(pred)
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")
            
            if 'lgb' in model:
                try:
                    pred = model['lgb'].predict(features)[0]
                    preds.append(pred)
                except Exception as e:
                    print(f"LightGBM prediction failed: {e}")
            
            if 'cat' in model:
                try:
                    pred = model['cat'].predict(features)[0]
                    preds.append(pred)
                except Exception as e:
                    print(f"CatBoost prediction failed: {e}")
            
            if preds:
                pred_value = np.mean(preds)
                std_error = np.std(preds) if len(preds) > 1 else 2.0
            else:
                pred_value = data.iloc[-1]['pm25_ugm3_mean']
                std_error = 3.0
        else:
            pred_value = data.iloc[-1]['pm25_ugm3_mean'] + np.random.normal(0, 2)
            std_error = 3.0
        
        predictions.append({
            'hour': h,
            'time': pred_time,
            'pm25': max(0, pred_value),
            'std_error': std_error
        })
    
    return pd.DataFrame(predictions)