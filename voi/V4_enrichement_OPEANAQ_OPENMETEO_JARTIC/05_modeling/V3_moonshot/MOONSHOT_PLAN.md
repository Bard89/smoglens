# V3 Moonshot Model Plan - Neural Architecture for Superior PM2.5 Prediction

## Executive Summary

After deep analysis of V1 baseline and V2 ensemble models, this plan outlines a revolutionary hybrid neural-gradient architecture optimized for M4 Pro (50GB RAM, unified memory) that aims to surpass both LinearRegression R² scores and LightGBM ensemble MAE performance.

## Current Performance Analysis

### V1 Baseline (Linear Regression)
**Strengths:**
- Excellent R² scores: 0.845 (1h), 0.550 (6h), 0.379 (12h), 0.201 (24h)
- Simple, interpretable, stable predictions
- MAE: 2.08 (1h), 3.57 (6h), 4.24 (12h), 4.84 (24h)

**Weaknesses:**
- Limited non-linear pattern capture
- No interaction terms
- Poor at extreme values

### V2 Advanced (Gradient Boosting Ensemble)
**Strengths:**
- Beat MAE on ALL 8 horizons (3-4% improvement)
- Strong feature engineering (69 features)
- Ensemble of LightGBM/XGBoost/CatBoost

**Weaknesses:**
- Failed on R² (especially long horizons: -31% at 12h, -55% at 24h)
- High variance in predictions
- Memory intensive (9.4GB)

### Key Dataset Insights
- **9.4M records**, 634 hexagons, 2 years of data
- **20% missing PM2.5** data (handled via interpolation)
- **Weak linear correlations** (<0.15) suggest complex non-linear relationships
- **Strong temporal patterns**: hourly, daily, weekly cycles
- **Spatial sparsity**: Only 30 weather stations for 634 hexagons

## V3 Moonshot Architecture

### Core Innovation: Hybrid Neural-Gradient Architecture
Combine the best of both worlds:
- Neural networks for complex temporal dependencies
- Gradient boosting for tabular feature interactions
- Meta-learner fusion for optimal combination

### Model Components

#### 1. Temporal Transformer Module (PyTorch MPS-accelerated)
```python
class TemporalTransformer:
    - Input: 168-hour sequences (1 week)
    - Architecture:
        * 6 transformer layers
        * 8 attention heads
        * 256 hidden dimensions
        * Learnable positional encodings
        * Dropout: 0.1
    - Output: Temporal representation (256d)
```

#### 2. Enhanced Gradient Boosting
```python
class EnhancedLightGBM:
    - Features: 100+ engineered features
    - Configuration:
        * num_leaves: 511
        * max_depth: 15
        * learning_rate: 0.02
        * min_child_samples: 3
    - Trained on residuals from transformer
```

#### 3. Spatial Graph Network
```python
class SpatialGNN:
    - Graph construction: K-nearest hexagons (K=5)
    - Message passing: 3 layers
    - Aggregation: Mean pooling
    - Output: Spatial context (128d)
```

#### 4. Meta-Learner Fusion
```python
class MetaLearner:
    - Inputs: [transformer_output, lgb_output, gnn_output]
    - Architecture:
        * 3 dense layers [512, 256, 128]
        * Horizon-specific heads (8 outputs)
        * Uncertainty quantification branch
    - Loss: Weighted MSE + uncertainty regularization
```

## Advanced Feature Engineering

### Temporal Features (New)
1. **Wavelet Decomposition**
   - Decompose PM2.5 into trend, seasonal, residual
   - Multi-resolution analysis (2h, 6h, 24h, 168h)

2. **DTW-based Similar Day Matching**
   - Find top-5 most similar historical days
   - Use their outcomes as features

3. **Fourier Features with Harmonics**
   - Up to 3rd harmonic for hour, day, month cycles
   - Phase shifts for lag compensation

### Spatial Features (New)
1. **Hexagon Embeddings**
   - Learnable 32d embeddings per hexagon
   - Capture location-specific patterns

2. **Spatial Lag Features**
   - PM2.5 from neighboring hexagons
   - Distance-weighted averages

3. **Wind-adjusted Upwind PM2.5**
   - Consider wind direction and speed
   - Get PM2.5 from upwind hexagons

### Interaction Features (Enhanced)
1. **Deep Interactions**
   - Temperature × Humidity × Hour
   - Traffic × Weekend × Hour
   - Pressure change × Temperature gradient

2. **Temporal Stability Metrics**
   - Variance over different windows
   - Trend strength indicators
   - Changepoint detection features

## Training Strategy

### Multi-Task Learning Setup
```python
losses = {
    '1h': weight=1.0,
    '2h': weight=0.9,
    '3h': weight=0.8,
    '4h': weight=0.7,
    '5h': weight=0.6,
    '6h': weight=0.5,
    '12h': weight=0.3,
    '24h': weight=0.2
}
```

### Curriculum Learning
1. **Phase 1** (Epochs 1-20): Train on 1-6h horizons only
2. **Phase 2** (Epochs 21-50): Add 12h horizon
3. **Phase 3** (Epochs 51-100): Add 24h horizon
4. **Phase 4** (Epochs 101-150): Fine-tune all horizons

### Data Augmentation
1. **Mixup**: Blend training samples (alpha=0.2)
2. **Noise Injection**: Add Gaussian noise (std=0.1)
3. **Temporal Shifts**: Random shifts ±1 hour

### Ensemble Strategy
Train 5 models with different:
- Random seeds
- Feature subsets (90% sampling)
- Initial learning rates
- Architecture variations (depth ±1 layer)

## M4 Pro Optimization

### Hardware Utilization
```python
import torch
device = torch.device("mps")  # Apple Silicon GPU

# Optimizations
torch.backends.mps.enabled = True
torch.set_float32_matmul_precision('high')
```

### Memory Management
```python
# Gradient accumulation for effective batch size 2048
micro_batch_size = 256
accumulation_steps = 8

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Parallel Processing
```python
# DataLoader optimization
num_workers = 8  # Use 8 CPU cores
pin_memory = True
prefetch_factor = 4
```

### Chunked Processing
- Process hexagons in chunks of 100
- Aggregate predictions after each chunk
- Keep memory under 40GB

## Performance Targets

### Primary Goals (Must Achieve)
| Horizon | Current Best MAE | Target MAE | Current Best R² | Target R² |
|---------|-----------------|------------|-----------------|-----------|
| 1h      | 2.00 (V2)       | **1.75**   | 0.845 (V1)      | **0.88**  |
| 6h      | 3.44 (V2)       | **3.00**   | 0.550 (V1)      | **0.60**  |
| 12h     | 4.14 (V2)       | **3.60**   | 0.379 (V1)      | **0.45**  |
| 24h     | 4.66 (V2)       | **4.20**   | 0.201 (V1)      | **0.30**  |

### Secondary Goals
- Training time: <2 hours on M4 Pro
- Inference time: <100ms per hexagon
- Memory usage: <40GB peak
- Model size: <500MB serialized

## Implementation Timeline

### Phase 1: Setup (Day 1)
1. Create folder structure
2. Install PyTorch with MPS support
3. Set up data pipeline
4. Implement base transformer

### Phase 2: Core Development (Days 2-3)
1. Implement spatial GNN
2. Enhanced feature engineering
3. Gradient boosting integration
4. Meta-learner fusion

### Phase 3: Training (Day 4)
1. Multi-task loss implementation
2. Curriculum learning schedule
3. Training loop with validation
4. Hyperparameter tuning

### Phase 4: Evaluation (Day 5)
1. Comprehensive metrics
2. Comparison visualizations
3. Error analysis
4. Uncertainty calibration

## Expected Innovations

### Technical Breakthroughs
1. **Hybrid Architecture**: First to combine transformer + GNN + gradient boosting
2. **Curriculum Multi-Task**: Novel training strategy for time series
3. **Uncertainty Quantification**: Prediction intervals for risk assessment

### Performance Breakthroughs
1. **Best of Both Worlds**: Low MAE from ensembles + High R² from linear models
2. **Long-Horizon Excellence**: First model to achieve R² > 0.25 at 24h
3. **Real-time Capable**: Fast inference for production use

## Risk Mitigation

### Potential Challenges
1. **Overfitting**: Use strong regularization, dropout, early stopping
2. **Memory Issues**: Implement gradient checkpointing if needed
3. **Training Instability**: Use gradient clipping, learning rate scheduling
4. **Long Training Time**: Pre-compute features, use cached datasets

### Fallback Options
1. If transformer too slow → Use LSTM/GRU
2. If memory exceeds 50GB → Reduce batch size, use gradient accumulation
3. If R² doesn't improve → Add linear skip connection
4. If training unstable → Use simpler architecture first

## Success Criteria

### Minimum Viable Model
- Beat V2 MAE on 6/8 horizons
- Beat V1 R² on 4/8 horizons
- Training completes in <4 hours
- Memory usage <50GB

### Stretch Goals
- Beat both MAE and R² on ALL horizons
- Achieve R² > 0.9 for 1h predictions
- Reduce 24h MAE below 4.0
- Create real-time inference API

## Next Steps

1. **Immediate**: Create notebook `01_neural_gradient_hybrid.ipynb`
2. **Install Requirements**: `pip install torch torchvision torchaudio pytorch-lightning wandb`
3. **Data Preparation**: Load and preprocess V4 enriched dataset
4. **Start Simple**: Implement transformer-only baseline first
5. **Iterate**: Add components incrementally, validating each addition

## Conclusion

This V3 Moonshot model represents a paradigm shift in PM2.5 prediction, combining cutting-edge deep learning with proven gradient boosting techniques. By leveraging the M4 Pro's unified memory architecture and neural engine, we can train a model that surpasses all previous baselines while maintaining practical inference speeds for production deployment.

The hybrid architecture addresses the key weaknesses of both V1 (limited complexity) and V2 (high variance), promising to deliver the first model that excels at both MAE and R² metrics across all prediction horizons.