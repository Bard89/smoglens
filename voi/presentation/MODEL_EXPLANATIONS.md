# Model Explanations: From Linear Regression to Ensemble

## Table of Contents
1. [Linear Regression Models](#linear-regression-models)
2. [Gradient Boosting Ensemble](#gradient-boosting-ensemble)
3. [How the Ensemble Works](#how-the-ensemble-works)
4. [Key Differences & Improvements](#key-differences--improvements)

---

## Linear Regression Models

### V1 Baseline: Simple Time Series Linear Regression

**What it is:**
Linear regression finds the best-fitting straight line through data points by minimizing the sum of squared errors. For time series prediction, it learns a linear relationship between past values and future values.

**Mathematical Formula:**
```
y(t+h) = β₀ + β₁×y(t-1) + β₂×y(t-2) + ... + βₙ×y(t-n) + ε
```
Where:
- `y(t+h)` = PM2.5 value h hours in the future
- `β₀` = intercept (baseline PM2.5 level)
- `β₁...βₙ` = coefficients (weights for each lag)
- `y(t-1)...y(t-n)` = past PM2.5 values (lags)
- `ε` = error term

**V1 Features (PM2.5 only):**
- Lag features: PM2.5 at t-1, t-2, t-3, t-6, t-12, t-24 hours
- Rolling statistics: 6h and 24h moving averages
- Temporal encoding: hour of day, day of week (cyclical)
- **Total: ~15 features**

**Strengths:**
- Simple and interpretable
- Fast to train and predict
- No hyperparameter tuning needed
- Works well for linear relationships

**Weaknesses:**
- Assumes linear relationships only
- Cannot capture complex interactions
- Sensitive to outliers
- Limited by collinearity in features

### V4 Baseline: Enhanced Linear Regression

**What changed:**
Same linear regression algorithm, but with many more features to capture environmental context.

**V4 Additional Features:**
- **Weather features:** temperature, humidity, pressure, wind speed, cloud cover
- **Traffic features:** average volume, congestion index
- **Advanced temporal:** multiple harmonics for cyclical patterns
- **Spatial encoding:** hexagon ID for location-specific patterns
- **Total: 19 features** (vs 15 in V1)

**Why it's better:**
- Weather affects PM2.5 (rain washes particles, wind disperses them)
- Traffic is a major PM2.5 source
- Location matters (industrial vs residential areas)
- More nuanced temporal patterns

**Performance improvement:**
- **1h forecast:** MAE reduced from 2.35 → 2.08 μg/m³ (11.5% improvement)
- **24h forecast:** MAE reduced from 5.85 → 4.84 μg/m³ (17.3% improvement)

---

## Gradient Boosting Ensemble

### What is Gradient Boosting?

Gradient boosting builds many simple models (decision trees) sequentially, where each new tree corrects the errors of the previous ones.

**Conceptual Process:**
1. Start with a simple prediction (e.g., mean PM2.5)
2. Calculate errors (residuals)
3. Train a tree to predict these errors
4. Add tree's predictions to improve overall prediction
5. Repeat with new residuals
6. Final prediction = sum of all trees

### The Three Models in Our Ensemble

#### 1. LightGBM (Light Gradient Boosting Machine)
**How it works:**
- Uses leaf-wise tree growth (grows best leaf, not level-by-level)
- Histogram-based algorithm for faster training
- Handles categorical features natively

**Best for:**
- Large datasets (our 7M+ samples)
- When you need fast training
- Categorical features (like hexagon ID)

**Key parameters used:**
```python
num_leaves: 127-255 (complexity of trees)
learning_rate: 0.03-0.05 (how much each tree contributes)
feature_fraction: 0.8 (randomly sample 80% features per tree)
```

#### 2. XGBoost (Extreme Gradient Boosting)
**How it works:**
- Level-wise tree growth (grows all leaves at same depth)
- Advanced regularization to prevent overfitting
- Handles missing values automatically

**Best for:**
- Balanced performance and accuracy
- When you need robust predictions
- Datasets with missing values

**Key parameters used:**
```python
max_depth: 8-10 (tree depth)
subsample: 0.8 (use 80% of data per tree)
colsample_bytree: 0.8 (use 80% of features per tree)
```

#### 3. CatBoost (Categorical Boosting)
**How it works:**
- Ordered boosting to reduce overfitting
- Special handling of categorical features
- Symmetric trees for faster prediction

**Best for:**
- Datasets with many categorical features
- When you need stable, less-overfit models
- Production deployments (fast inference)

**Key parameters used:**
```python
depth: 8-10 (tree depth)
l2_leaf_reg: 3 (L2 regularization)
border_count: 128 (discretization bins)
```

---

## How the Ensemble Works

### Step 1: Individual Model Training
Each model is trained independently on the same data but learns different patterns:
- **LightGBM:** Captures fine-grained local patterns
- **XGBoost:** Finds robust global relationships
- **CatBoost:** Handles categorical interactions well

### Step 2: Weighted Averaging
```python
ensemble_prediction = w₁×LightGBM + w₂×XGBoost + w₃×CatBoost
```
Where weights sum to 1.0

**Our weights:** Equal (0.33, 0.33, 0.34) - optimized per horizon

### Step 3: Why Ensemble Works Better

**1. Error Reduction through Diversity:**
- Each model makes different types of errors
- Averaging cancels out individual model mistakes
- Example: If LGB overestimates and XGB underestimates, average is closer to truth

**2. Variance Reduction:**
- Single models can be unstable (high variance)
- Ensemble averages out this instability
- More consistent predictions across different data

**3. Capturing Different Patterns:**
- LightGBM might excel at capturing sudden spikes
- XGBoost might better model gradual trends
- CatBoost might handle location-specific patterns
- Together, they cover more scenarios

### Visual Example:
```
True PM2.5:          10.0 μg/m³
LightGBM predicts:   11.2 μg/m³ (error: +1.2)
XGBoost predicts:     9.5 μg/m³ (error: -0.5)
CatBoost predicts:   10.3 μg/m³ (error: +0.3)
Ensemble predicts:   10.3 μg/m³ (error: +0.3) ← Better than any single model!
```

---

## Key Differences & Improvements

### Linear Regression vs Gradient Boosting

| Aspect | Linear Regression | Gradient Boosting |
|--------|------------------|-------------------|
| **Model Type** | Single linear equation | Hundreds of decision trees |
| **Relationships** | Linear only | Non-linear, complex interactions |
| **Feature Interactions** | Manual only | Automatic discovery |
| **Training Speed** | Very fast (seconds) | Slower (minutes) |
| **Interpretability** | High (coefficients) | Low (black box) |
| **Overfitting Risk** | Low | High (needs regularization) |
| **Performance Ceiling** | Limited | Much higher |

### Feature Engineering Impact

**V4 Enhanced Features (69 total):**
1. **Lag features (11):** 1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168 hours
2. **Rolling statistics (20):** mean, std, min, max for 3, 6, 12, 24, 48h windows
3. **Exponential weighted averages (3):** α = 0.1, 0.3, 0.5
4. **Rate of change (5):** differences and rates over various periods
5. **Weather interactions (4):** temp×humidity, temp×hour, etc.
6. **Traffic interactions (2):** traffic×hour, traffic×weekend
7. **Fourier features (10):** multiple harmonics for temporal patterns
8. **Categorical encoding (1):** hexagon ID

### Why These Features Matter

**Lag Features:**
- PM2.5 has strong autocorrelation
- Recent values best predict near future
- 168h lag captures weekly patterns

**Rolling Statistics:**
- Smooth out noise
- Capture trends vs fluctuations
- Different windows for different patterns

**Interaction Features:**
- High temperature + high humidity = worse air quality
- Rush hour traffic impact varies by time
- Weekend traffic patterns differ

**Fourier Encoding:**
- Natural cycles (daily, weekly)
- Smooth representation (no jumps at midnight)
- Multiple harmonics capture complex patterns

### Performance Evolution

```
V1 Linear (PM2.5 only) → V4 Linear (+features) → V4 Ensemble
      2.35 μg/m³      →      2.08 μg/m³      →   2.00 μg/m³   (1h MAE)
      ↓                       ↓                    ↓
   Baseline            11.5% better          14.7% better than V1
```

### Key Insights

1. **Feature engineering provides biggest gain** (11.5% improvement)
2. **Ensemble adds incremental value** (additional 3.4%)
3. **Short-term predictions benefit most** (better lag correlation)
4. **Long-term R² remains challenging** (chaotic system limits)
5. **Model diversity reduces risk** (no single point of failure)

### Practical Implications

**For 1-hour predictions:**
- Error of ±2 μg/m³ is excellent for most applications
- Can reliably detect air quality changes
- Suitable for public health warnings

**For 24-hour predictions:**
- Error of ±4.7 μg/m³ still useful for planning
- Captures general trends, not precise values
- Good for next-day activity recommendations

**Future improvements should focus on:**
- Spatial features (neighboring hexagon values)
- External data (satellite, weather forecasts)
- Adaptive models (retrain on recent data)
- Uncertainty quantification (prediction intervals)