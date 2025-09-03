import numpy as np
import pandas as pd
import h3
from typing import List, Tuple, Optional
import pickle
from pathlib import Path
import config

class SpatialImputer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.neighbor_distances = None
        self.cache_path = config.DATA_DIR / 'spatial_cache.pkl'
        
    def get_hexagon_neighbors(self, hex_id: str, k: int = 1) -> List[str]:
        return list(h3.grid_disk(hex_id, k))
    
    def calculate_hex_distance(self, hex1: str, hex2: str) -> float:
        lat_lon1 = h3.cell_to_latlng(hex1)
        lat_lon2 = h3.cell_to_latlng(hex2)
        return h3.great_circle_distance(lat_lon1, lat_lon2, unit='km')
    
    def find_neighbors_with_data(self, target_hex: str, timestamp: pd.Timestamp, max_radius: int = 3) -> List[Tuple[str, float]]:
        neighbors_with_data = []
        
        for radius in range(1, max_radius + 1):
            ring_neighbors = h3.grid_disk(target_hex, radius) - h3.grid_disk(target_hex, radius - 1)
            
            for neighbor in ring_neighbors:
                neighbor_data = self.data_processor.data[
                    (self.data_processor.data['hex7_id'] == neighbor) &
                    (self.data_processor.data['timestamp'] == timestamp)
                ]
                
                if not neighbor_data.empty:
                    distance = self.calculate_hex_distance(target_hex, neighbor)
                    neighbors_with_data.append((neighbor, distance, neighbor_data.iloc[0]))
            
            if len(neighbors_with_data) >= config.K_NEIGHBORS:
                break
        
        neighbors_with_data.sort(key=lambda x: x[1])
        return neighbors_with_data[:config.K_NEIGHBORS]
    
    def impute_value(self, target_hex: str, timestamp: pd.Timestamp, column: str = 'pm25') -> Optional[float]:
        existing_data = self.data_processor.data[
            (self.data_processor.data['hex7_id'] == target_hex) &
            (self.data_processor.data['timestamp'] == timestamp)
        ]
        
        if not existing_data.empty and not pd.isna(existing_data.iloc[0][column]):
            return existing_data.iloc[0][column]
        
        neighbors = self.find_neighbors_with_data(target_hex, timestamp)
        
        if not neighbors:
            return None
        
        weighted_sum = 0
        weight_total = 0
        
        for neighbor_hex, distance, neighbor_data in neighbors:
            if column in neighbor_data and not pd.isna(neighbor_data[column]):
                weight = 1.0 / max(distance, 0.1)
                weighted_sum += neighbor_data[column] * weight
                weight_total += weight
        
        if weight_total > 0:
            return weighted_sum / weight_total
        
        return None
    
    def impute_dataframe(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            columns = ['pm25_ugm3_mean', 'temperature_c_mean', 'humidity_pct_mean', 
                      'pressure_hpa_mean', 'avg_traffic_volume']
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            missing_mask = df[col].isna()
            if missing_mask.any():
                for idx in df[missing_mask].index:
                    row = df.loc[idx]
                    imputed_value = self.impute_value(
                        row['hex7_id'], 
                        row['timestamp'], 
                        col
                    )
                    if imputed_value is not None:
                        df.loc[idx, col] = imputed_value
        
        return df
    
    def get_spatial_coverage(self, timestamp: pd.Timestamp, radius: int = 2) -> dict:
        target_hex = config.SHIBUYA_HEXAGON
        neighbors = h3.grid_disk(target_hex, radius)
        
        coverage = {
            'total_hexagons': len(neighbors),
            'hexagons_with_data': 0,
            'coverage_percentage': 0.0,
            'average_distance': 0.0
        }
        
        timestamp = pd.to_datetime(timestamp)
        time_window_start = timestamp - pd.Timedelta(hours=1)
        time_window_end = timestamp + pd.Timedelta(hours=1)
        
        hexagons_with_data = []
        distances = []
        
        for neighbor in neighbors:
            neighbor_data = self.data_processor.data[
                (self.data_processor.data['hex7_id'] == neighbor) &
                (self.data_processor.data['timestamp'] >= time_window_start) &
                (self.data_processor.data['timestamp'] <= time_window_end)
            ]
            
            if not neighbor_data.empty:
                hexagons_with_data.append(neighbor)
                if neighbor != target_hex:
                    distances.append(self.calculate_hex_distance(target_hex, neighbor))
        
        coverage['hexagons_with_data'] = len(hexagons_with_data)
        coverage['coverage_percentage'] = 100.0 * len(hexagons_with_data) / len(neighbors)
        
        if distances:
            coverage['average_distance'] = np.mean(distances)
        
        if coverage['hexagons_with_data'] == 0:
            available_hexes = self.data_processor.data[
                (self.data_processor.data['timestamp'] >= time_window_start) &
                (self.data_processor.data['timestamp'] <= time_window_end)
            ]['hex7_id'].nunique()
            coverage['note'] = f"No nearby data. {available_hexes} hexagons have data in this time window."
        
        return coverage