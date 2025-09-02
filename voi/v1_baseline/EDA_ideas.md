# PM2.5 Enriched Dataset - EDA Analysis Plan


## Step 1: Data Overview

**Dataset Structure**
- Dataset shape and memory usage
- Time range and temporal coverage
- Unique H3 hexagons and timestamps

**Data Completeness**
- Availability percentages for each data source
- Complete record analysis
- Missing value patterns

**Coverage Analysis**
- Missing patterns over time by feature type
- Missing data by hexagon distribution
- Geographic coverage extent

---

## Step 2: Data Quality Assessment

**Outlier Detection**
- Box plots for PM2.5, traffic volume, weather variables
- IQR-based outlier identification and Z-score analysis
- Outlier percentage by feature

**Data Consistency Checks**
- Verify reasonable values: negative PM2.5, humidity >100%, temperature ranges
- Validate timestamp continuity per hexagon
- Check for duplicate records

**Coverage Quality**
- Heatmap of data availability by hexagon and time
- Identify hexagons with consistent vs sporadic coverage
- Sensor reliability metrics

---

## Step 3: Temporal Patterns

**Time Series Decomposition**
- Hourly patterns (24-hour cycle)
- Day of week effects (weekday vs weekend)
- Monthly trends and seasonal patterns
- Year-over-year comparison (2023 vs 2024 vs 2025)

**Autocorrelation Analysis**
- ACF and PACF plots
- Lag correlation analysis: 1h, 3h, 6h, 12h, 24h, 1 week
- Identify optimal lag features for modeling

**Temporal Stability**
- Variance over time, trend analysis
- Stationarity tests

---

## Step 4: Spatial Patterns

**Geographic Clustering**
- K-means clustering by pollution levels
- Pollution hotspot identification
- Rural vs urban classification
- Coastal vs inland patterns

**Spatial Autocorrelation**
- Moran's I statistic
- Neighbor hexagon correlation analysis
- Spatial lag features and distance decay analysis

**Distance Impact Analysis**
- PM2.5 accuracy vs distance to sensors
- Optimal distance thresholds for reliability
- Traffic and weather data distance impact

**Regional Analysis**
- Prefecture-level aggregations
- Major city analysis (Tokyo, Osaka, etc.)
- Industrial area identification

---

## Step 5: Feature Relationships

**Correlation Analysis**
- Full correlation matrix heatmap
- PM2.5 correlations with all features
- Scatter plots for key relationships
- Non-linear relationship detection

**Traffic Impact Analysis**
- PM2.5 vs traffic volume relationship
- Local vs nearest traffic data comparison
- Traffic distance decay effect
- Rush hour impact analysis

**Weather Impact Analysis**
- Temperature effect on PM2.5
- Humidity influence and precipitation cleaning effect
- Combined weather factor analysis
- Seasonal weather patterns

**Feature Interactions**
- Traffic × Time of day
- Weather × Season
- Traffic × Weather conditions
- Distance × Data quality