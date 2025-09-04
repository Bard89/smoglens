# SmogLens PM2.5 Prediction - Presentation Guide

## Quick Navigation
- [Executive Summary](#executive-summary)
- [Presentation Flow](#presentation-flow)
- [Key Talking Points](#key-talking-points)
- [Technical Details](#technical-details)
- [Q&A Preparation](#qa-preparation)

---

## Executive Summary

**Project:** SmogLens - PM2.5 Air Quality Prediction for Japan  
**Goal:** Predict PM2.5 concentrations 1-24 hours ahead  
**Achievement:** 14.7% improvement over baseline through feature engineering and ensemble methods

### Three Model Iterations
1. **V1 Baseline:** Linear Regression with PM2.5 time series only
2. **V4 Baseline:** Linear Regression with weather & traffic features  
3. **V4 Ensemble:** XGBoost + LightGBM + CatBoost with 69 features

### Key Results
- **1-hour forecast:** MAE = 2.00 μg/m³ (from 2.35 baseline)
- **24-hour forecast:** MAE = 4.66 μg/m³ (from 5.85 baseline)
- **Trade-off:** Better accuracy (MAE) but R² challenges at long horizons

---

## Presentation Flow

### Slide 1: Title & Context
**SmogLens: Evolution of PM2.5 Prediction Models**
- Air quality affects 3.7M deaths annually (WHO)
- Japan: 643 monitoring stations, hourly data
- Challenge: Accurate forecasting for public health

### Slide 2: Data Overview
**Multi-Source Data Integration (2023-2025)**
- **PM2.5:** 634 hexagons, 9.4M records
- **Weather:** OpenMeteo (30 hexagons, 100% coverage)
- **Traffic:** JARTIC (1,018 hexagons)
- **Spatial:** H3 hexagonal indexing (5.16 km²)

### Slide 3: Model Evolution Journey
Show side-by-side comparison graphs for 1h, 6h, 24h:
- **Graph:** `graphs/comparison_1h.png`
- **Graph:** `graphs/comparison_6h.png`
- **Graph:** `graphs/comparison_24h.png`

**Key Message:** Progressive improvement through feature engineering and ensemble methods

### Slide 4: Performance Metrics
**Graph:** `graphs/mae_comparison_bar.png`
- V1 → V4 Base: 11.5% improvement (features matter!)
- V4 Base → Ensemble: 3.4% additional gain
- Total: 14.7% MAE reduction

### Slide 5: R² Score Analysis
**Graph:** `graphs/r2_degradation_line.png`
- Strong short-term predictions (R² > 0.65 for 1-3h)
- Degradation after 6 hours (R² < 0.5)
- Physical limitation: chaos in atmospheric systems

### Slide 6: Ensemble Architecture
**Graph:** `insights/ensemble_architecture.png`
- 69 features → 3 models → weighted average
- Each model captures different patterns
- Diversity reduces prediction variance

### Slide 7: Feature Importance
**Graph:** `insights/feature_importance.png`
- Lag features dominate (>50%)
- Weather contributes 13%
- Traffic adds 4%
- Spatial context limited (3%)

### Slide 8: Model Agreement Analysis
**Graph:** `insights/model_agreement.png`
- High agreement at 1h (91%)
- Decreases to 61% at 24h
- Uncertainty grows with horizon

### Slide 9: Error Patterns
**Graph:** `insights/temporal_error_patterns.png`
- Rush hours show 15-20% higher errors
- Weekend predictions more accurate
- Morning peak harder to predict

### Slide 10: Key Learnings & Next Steps
**What We Learned:**
1. Feature engineering > model complexity
2. Ensemble reduces variance, not bias
3. Physical limits exist (R² ceiling)
4. Spatial features underutilized

**Next Steps:**
1. Add neighboring hexagon features
2. Include satellite data
3. Implement online learning
4. Add uncertainty quantification

---

## Key Talking Points

### On Model Progression
"We started with a simple linear regression using only PM2.5 history. By adding weather and traffic data, we improved accuracy by 11.5%. The ensemble pushed us further, but the biggest gain came from understanding the problem domain and engineering relevant features."

### On Ensemble Benefits
"Think of it like asking three experts for their opinion. LightGBM is fast and catches local patterns. XGBoost is robust and stable. CatBoost handles our location data well. By combining them, we get more reliable predictions."

### On R² Challenges
"While our MAE improved consistently, R² scores struggle at long horizons. This isn't a model failure - it's a fundamental limit. Weather systems are chaotic. We can predict trends, but exact values become impossible beyond 12 hours."

### On Practical Impact
"A 2 μg/m³ error at 1 hour means we can reliably issue health warnings. Even at 24 hours, ±5 μg/m³ is good enough for next-day activity planning. This translates to real public health benefits."

---

## Technical Details

### Data Processing Pipeline
```
Raw Data → H3 Aggregation → K-NN Interpolation → Feature Engineering → Model Training
   ↓            ↓                  ↓                    ↓                   ↓
 3 sources   5.16km² hex      Fill missing         69 features         3 models
```

### Training Details
- **Data Split:** 2023-2024 train, 2025 test
- **Samples:** 4.9M train, 1.8M test
- **Memory:** ~8GB peak usage
- **Training Time:** ~2 hours for full ensemble
- **Inference:** <100ms per prediction

### Feature Categories
```
Temporal (28): Lags, rolling stats, differences
Weather (6):   Temp, humidity, pressure, etc.
Traffic (2):   Volume, congestion
Spatial (1):   Hexagon ID
Derived (32):  Interactions, rates, EWMs
```

### Model Hyperparameters
```python
LightGBM: num_leaves=127-255, lr=0.03-0.05
XGBoost:  max_depth=8-10, lr=0.03-0.05
CatBoost: depth=8-10, l2_reg=3
```

---

## Q&A Preparation

### Q: "Why not use deep learning/LSTM?"
**A:** We tested LSTM in preliminary experiments. While it matched ensemble performance at short horizons, it was:
- 10x slower to train
- Harder to interpret
- Required more data preprocessing
- Didn't significantly beat gradient boosting

### Q: "What about real-time predictions?"
**A:** Our inference is fast (<100ms). The bottleneck is data availability - we need recent PM2.5, weather, and traffic data. With proper data pipeline, real-time is feasible.

### Q: "Can this transfer to other cities?"
**A:** The methodology yes, but models need retraining. Each city has unique:
- Emission sources
- Geography/topology  
- Weather patterns
- Traffic flows

### Q: "What's the business value?"
**A:** 
- **Health apps:** Personalized activity recommendations
- **Government:** Pollution alerts, traffic management
- **Healthcare:** Hospital resource planning
- **Insurance:** Risk assessment for respiratory claims

### Q: "Main limitations?"
**A:**
1. Spatial coverage gaps (only 30 weather hexagons)
2. No external PM2.5 sources (China, Korea)
3. Missing extreme event data (wildfires, dust storms)
4. Static model (doesn't adapt to changes)

### Q: "Why equal ensemble weights?"
**A:** Optimization showed marginal gains (<1%) from custom weights. Equal weights are:
- More robust
- Simpler to maintain
- Less prone to overfitting
- Easier to explain

### Q: "Next research direction?"
**A:** Graph Neural Networks for spatial modeling. Current models treat hexagons independently, missing pollution transport between regions. GNNs could model this flow.

---

## Presentation Tips

1. **Start with impact:** Air quality affects everyone
2. **Show progression visually:** Use the 3-panel comparisons
3. **Acknowledge limitations:** R² degradation is physics, not failure
4. **Focus on improvements:** 14.7% better saves lives
5. **Keep technical details for Q&A:** Main story is evolution and learning

## Files to Have Ready

**Notebooks to run:**
1. `01_model_evolution_comparison.ipynb` - Main comparison plots
2. `02_performance_summary.ipynb` - Metrics and tables
3. `03_ensemble_insights.ipynb` - Deep dive visualizations

**Generated graphs in:**
- `graphs/` - Comparison and performance plots
- `insights/` - Architecture and analysis plots

**Reference documents:**
- `MODEL_EXPLANATIONS.md` - Technical details
- Original notebooks in `voi/v4_multiyear/` for code review

---

*Good luck with your presentation!*