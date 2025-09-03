import joblib
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import config

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.load_models()
        self.load_metadata()
        
    def load_models(self):
        for horizon in config.HORIZONS:
            model_path = config.MODEL_DIR / f'models_{horizon}.pkl'
            if model_path.exists():
                self.models[horizon] = joblib.load(model_path)
                print(f"Loaded model for {horizon}")
            else:
                print(f"Model not found for {horizon}: {model_path}")
    
    def load_metadata(self):
        metadata_path = config.MODEL_DIR / 'metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_cols = metadata['feature_cols']
        else:
            print(f"Metadata not found: {metadata_path}")
    
    def predict(self, features_df):
        import xgboost as xgb
        import lightgbm as lgb
        
        predictions = {}
        confidence_intervals = {}
        
        if self.feature_cols:
            missing_cols = set(self.feature_cols) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0.0
            
            features_df = features_df[self.feature_cols]
        
        X = features_df.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        for horizon in config.HORIZONS:
            if horizon not in self.models:
                predictions[horizon] = np.nan
                confidence_intervals[horizon] = (np.nan, np.nan)
                continue
            
            model_dict = self.models[horizon]
            
            lgb_pred = model_dict['lgb'].predict(X, num_iteration=model_dict['lgb'].num_trees())
            
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_cols)
            xgb_pred = model_dict['xgb'].predict(dmatrix)
            
            cat_pred = model_dict['cat'].predict(X)
            
            ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3.0
            
            predictions[horizon] = float(np.clip(ensemble_pred[0], 0, config.PM25_CAP))
            
            std_dev = np.std([lgb_pred[0], xgb_pred[0], cat_pred[0]])
            lower = float(np.clip(ensemble_pred[0] - 1.96 * std_dev, 0, config.PM25_CAP))
            upper = float(np.clip(ensemble_pred[0] + 1.96 * std_dev, 0, config.PM25_CAP))
            confidence_intervals[horizon] = (lower, upper)
        
        return predictions, confidence_intervals
    
    def predict_sequence(self, features_df):
        predictions = []
        confidences = []
        
        for i in range(len(config.HORIZONS)):
            horizon = config.HORIZONS[i]
            pred, conf = self.predict(features_df.iloc[-1:])
            predictions.append(pred[horizon])
            confidences.append(conf[horizon])
        
        return predictions, confidences
    
    def get_activity_recommendations(self, predictions):
        recommendations = {}
        
        for activity, threshold in config.ACTIVITY_LIMITS.items():
            safe_hours = []
            for i, (horizon, pred) in enumerate(predictions.items()):
                if pred <= threshold:
                    safe_hours.append(int(horizon[:-1]))
            
            recommendations[activity] = {
                'safe_hours': safe_hours,
                'is_safe_now': 1 in safe_hours if safe_hours else False,
                'best_time': min(safe_hours) if safe_hours else None
            }
        
        return recommendations