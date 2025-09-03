import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import pytz
import config

class DataProcessor:
    def __init__(self):
        self.data = None
        self.shibuya_data = None
        self.nearby_hexagons = None
        self.cache_path = config.DATA_DIR / 'shibuya_processed.parquet'
        self.neighbor_cache_path = config.DATA_DIR / 'neighbor_cache.pkl'
        
    def load_data(self):
        if self.cache_path.exists():
            self.data = pd.read_parquet(self.cache_path)
        else:
            print("Loading raw data...")
            df = pd.read_csv(config.DATA_PATH, 
                           parse_dates=['timestamp'],
                           dtype={'hex7_id': str})
            
            df = df.sort_values(['hex7_id', 'timestamp'])
            df['pm25'] = df['pm25_ugm3_mean'].clip(upper=config.PM25_CAP)
            
            self.data = df
            self.save_cache()
        
        self.shibuya_data = self.data[self.data['hex7_id'] == config.SHIBUYA_HEXAGON].copy()
        return self.data
    
    def save_cache(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(self.cache_path, compression='snappy')
    
    def find_nearby_hexagons(self):
        if self.neighbor_cache_path.exists():
            with open(self.neighbor_cache_path, 'rb') as f:
                self.nearby_hexagons = pickle.load(f)
        else:
            import h3
            
            hexagon_counts = self.data.groupby('hex7_id').size()
            valid_hexagons = hexagon_counts[hexagon_counts > 1000].index.tolist()
            
            center_lat_lon = h3.cell_to_latlng(config.SHIBUYA_HEXAGON)
            
            hex_distances = []
            for hex_id in valid_hexagons:
                if hex_id == config.SHIBUYA_HEXAGON:
                    continue
                hex_lat_lon = h3.cell_to_latlng(hex_id)
                distance = h3.great_circle_distance(center_lat_lon, hex_lat_lon, unit='km')
                hex_distances.append((hex_id, distance))
            
            hex_distances.sort(key=lambda x: x[1])
            self.nearby_hexagons = hex_distances[:config.K_NEIGHBORS]
            
            with open(self.neighbor_cache_path, 'wb') as f:
                pickle.dump(self.nearby_hexagons, f)
        
        return self.nearby_hexagons
    
    def get_data_at_time(self, timestamp):
        timestamp = pd.to_datetime(timestamp)
        
        exact_match = self.shibuya_data[self.shibuya_data['timestamp'] == timestamp]
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        closest_before = self.shibuya_data[self.shibuya_data['timestamp'] <= timestamp]
        if not closest_before.empty:
            return closest_before.iloc[-1]
        
        return self.impute_from_neighbors(timestamp)
    
    def impute_from_neighbors(self, timestamp):
        if self.nearby_hexagons is None:
            self.find_nearby_hexagons()
        
        weighted_sum = 0
        weight_total = 0
        
        for hex_id, distance in self.nearby_hexagons:
            hex_data = self.data[(self.data['hex7_id'] == hex_id) & 
                                 (self.data['timestamp'] == timestamp)]
            if not hex_data.empty:
                weight = 1.0 / max(distance, 1)
                weighted_sum += hex_data.iloc[0]['pm25'] * weight
                weight_total += weight
        
        if weight_total > 0:
            imputed_row = self.shibuya_data.iloc[0].copy()
            imputed_row['timestamp'] = timestamp
            imputed_row['pm25'] = weighted_sum / weight_total
            return imputed_row
        
        return None
    
    def get_historical_window(self, end_timestamp, hours_back=168):
        end_timestamp = pd.to_datetime(end_timestamp)
        start_timestamp = end_timestamp - timedelta(hours=hours_back)
        
        window_data = self.shibuya_data[
            (self.shibuya_data['timestamp'] >= start_timestamp) & 
            (self.shibuya_data['timestamp'] <= end_timestamp)
        ].copy()
        
        if len(window_data) < hours_back * 0.5:
            for hex_id, distance in self.nearby_hexagons[:3]:
                hex_data = self.data[
                    (self.data['hex7_id'] == hex_id) & 
                    (self.data['timestamp'] >= start_timestamp) & 
                    (self.data['timestamp'] <= end_timestamp)
                ]
                if len(hex_data) > len(window_data):
                    window_data = hex_data.copy()
                    window_data['hex7_id'] = config.SHIBUYA_HEXAGON
                    break
        
        return window_data