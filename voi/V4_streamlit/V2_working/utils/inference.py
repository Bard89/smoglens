import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool
import config

class SimplePredictor:
    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.load_resources()
        
    def load_resources(self):
        import pickle
        metadata_path = config.MODEL_DIR / 'metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_cols = metadata['feature_cols']
        
        for horizon in config.HORIZONS:
            model_path = config.MODEL_DIR / f'models_{horizon}.pkl'
            if model_path.exists():
                self.models[horizon] = joblib.load(model_path)
    
    def predict_horizon(self, features_df, features_array, horizon):
        if horizon not in self.models:
            return np.nan, (np.nan, np.nan)
        
        model_dict = self.models[horizon]
        
        lgb_pred = model_dict['lgb'].predict(features_array, num_iteration=model_dict['lgb'].num_trees())
        
        dmatrix = xgb.DMatrix(features_array, feature_names=self.feature_cols)
        xgb_pred = model_dict['xgb'].predict(dmatrix)
        
        features_df_cat = pd.DataFrame(features_array, columns=self.feature_cols)
        features_df_cat['hex_encoded'] = features_df_cat['hex_encoded'].astype(int)
        cat_pred = model_dict['cat'].predict(features_df_cat)
        
        ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3.0
        
        if isinstance(ensemble_pred, np.ndarray):
            pred_value = float(ensemble_pred[0])
        else:
            pred_value = float(ensemble_pred)
        
        pred_value = np.clip(pred_value, 0, config.PM25_CAP)
        
        std_dev = np.std([lgb_pred[0] if isinstance(lgb_pred, np.ndarray) else lgb_pred,
                         xgb_pred[0] if isinstance(xgb_pred, np.ndarray) else xgb_pred,
                         cat_pred[0] if isinstance(cat_pred, np.ndarray) else cat_pred])
        
        lower = float(np.clip(pred_value - 1.96 * std_dev, 0, config.PM25_CAP))
        upper = float(np.clip(pred_value + 1.96 * std_dev, 0, config.PM25_CAP))
        
        return pred_value, (lower, upper)
    
    def predict_all(self, features_df):
        if self.feature_cols:
            missing_cols = set(self.feature_cols) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0.0
            features_df = features_df[self.feature_cols].copy()
        
        features_df['hex_encoded'] = features_df['hex_encoded'].astype(int)
        
        features_array = features_df.values
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)
        
        predictions = {}
        confidence_intervals = {}
        
        for horizon in config.HORIZONS:
            pred, ci = self.predict_horizon(features_df, features_array, horizon)
            predictions[horizon] = pred
            confidence_intervals[horizon] = ci
        
        return predictions, confidence_intervals