## Unreleased

━━━━━━ PROJECT PRESENTATION SUBMISSION THU 05/09/2025 ━━━━━━

### [0.1.1] - V5 Extended Feature Enrichment (Planned)
- Additional data sources integration w/ new features
  - Explore and potentially add features from other already downloaded datasets.
  - Get more data from elsewhere. Need some outside of Japan pm2.5 measurements to be able to predict delayed transmission to Japan. Potentially data from Korea we have already or some low res satelite data paired with weather.
  - This dataset from the Climate Data Store API connected the yellow dust dataset (NetCDF zip) from the Copernicus Atmosphere Monitoring Service, including PM2.5, PM10, and wind from China. [Yellow Dust Preprocessing EDA](https://github.com/Bard89/smoglens/blob/develop/yulia/V5_yellow_dust_analysis/yellow_dust_preprocessingEDA.ipynb)

━━━━━━ WE ARE HERE 02/09/2025 ━━━━━━

### [0.1.0] - 2025-08-31 - [V4 Multi-Year Analysis (2023-2025)](voi/v4_multiyear_oAQ_oM_J) (In progress)
- Extended temporal coverage: June 2023 - June 2025
- **Key finding:** Spatial mismatch between datasets - OpenMeteo's 30 hexagons vs OpenAQ's 643 limits nearest-neighbor enrichment effectiveness
- **Processing flow:**
  - **01_raw_data_exploration/** - Individual dataset EDAs:
    - [OpenAQ PM2.5 Analysis](voi/v4_multiyear_oAQ_oM_J/01_raw_data_exploration/openaq_2023_2025_eda.ipynb) - 643 hexagons, 80% PM2.5 completeness
    - [OpenMeteo Weather Analysis](voi/v4_multiyear_oAQ_oM_J/01_raw_data_exploration/openmeteo_2023_2025_eda.ipynb) - 30 hexagons, 100% temporal coverage
    - [JARTIC Traffic Analysis](voi/v4_multiyear_oAQ_oM_J/01_raw_data_exploration/jartic_2023_2025_eda.ipynb) - 1,018 hexagons, 100% completeness
    - [NASA Weather Analysis](voi/v4_multiyear_oAQ_oM_J/01_raw_data_exploration/nasa_weather_2023_2025_eda.ipynb) - 15 hexagons, May 2024 missing -> not in enrichement.
  - **02_feature_analysis/** - Pre-enrichment feature selection:
    - [Missing Data Analysis](voi/v4_multiyear_oAQ_oM_J/02_feature_analysis/missing_data_analysis.ipynb)
    - [Feature Correlation Analysis](voi/v4_multiyear_oAQ_oM_J/02_feature_analysis/feature_correlation_analysis.ipynb)
    - [Feature Selection Summary](voi/v4_multiyear_oAQ_oM_J/02_feature_analysis/feature_selection_summary.ipynb)
  - **03_data_processing/** - [Enrichment Pipeline](voi/v4_multiyear_oAQ_oM_J/03_data_processing/pm25_enrichment_pipeline_v4.ipynb) with K-NN interpolation
  - **04_enriched_data_analysis/** - [Enriched Dataset EDA](voi/v4_multiyear_oAQ_oM_J/04_enriched_data_analysis/pm25_enriched_2023_2025_v4_eda.ipynb) - 9.4M records, 39 features
  - **05_modeling/** - [Model Plan](voi/v4_multiyear_oAQ_oM_J/05_modeling/model_plan_v4.md) - Ensemble approach (LSTM + LightGBM + GNN)
    - **V1_baseline/** - Baseline model:
      - [01_baseline_reference](voi/v4_multiyear_oAQ_oM_J/05_modeling/V1_baseline/01_baseline_reference.ipynb) - **Reference baseline** (LinearReg: MAE=2.08@1h, 4.84@24h, R²=0.845@1h, 0.201@24h) - High coverage hexagons (≥90%), simple features, used for all V2 comparisons
    - **V2_advanced/** - Gradient boosting ensemble (XGBoost + CatBoost + LightGBM) with enhanced features
      - [01_train_gradient_boosting_ensemble](voi/v4_multiyear_oAQ_oM_J/05_modeling/V2_advanced/01_ensemble_training/01_train_gradient_boosting_ensemble.ipynb) - Main ensemble training with comprehensive feature engineering, includes training scripts
      - [02_evaluate_ensemble_performance](voi/v4_multiyear_oAQ_oM_J/05_modeling/V2_advanced/02_ensemble_evaluation/02_evaluate_ensemble_performance.ipynb) - Memory-optimized evaluation with statistical tests and visualizations
      - [03_lightgbm_feature_analysis](voi/v4_multiyear_oAQ_oM_J/05_modeling/V2_advanced/03_model_analysis/03_lightgbm_feature_analysis.ipynb) - Deep dive into LightGBM feature importance and SHAP analysis
      - **Results:** MAE beats baseline on all horizons, R² challenges persist

## Released

### [0.0.4] - 2025-08-30 - [V3 fixed 2023 Data Enrichment](voi/v3_fixed_oAQ_oM_J)
- OpenAQ 2023 data incomplete (Jan-Jul missing, API broken)
- Accumulation of one dataset from mid 2023 to mid 2025.
- Comprehensive EDAs on available years ([OpenAQ 2023-2025](voi/v3_fixed_oAQ_oM_J/openaq_2023_2025_eda.ipynb), [OpenMeteo 2023-2025](voi/v3_fixed_oAQ_oM_J/openmeteo_2023_2025_eda.ipynb), [JARTIC 2023-2025](voi/v3_fixed_oAQ_oM_J/jartic_2023_2025_eda.ipynb)). Based on the results some of the OpenMeteo data need to be re-downloaded and some JARTIC needs to be reprocessed. Both in progress, should be ready on 28/8/2025.
- [OpenMeteo data consistency analysis](voi/v3_fixed_oAQ_oM_J/openmeteo_data_consistency_analysis_betwee_2_monthly_datasets.ipynb) - identified 5x spatial resolution difference between 2 monthly downloads


### [0.0.3] - 2025-08-30 - [V2 improved 2023 EDA and data Enrichment](voi/v2_improved_oAQ_oM_J)
- Improved [enrichment pipeline](voi/v2_improved_oAQ_oM_J/pm25_hexagon_enrichment_2023.ipynb) (combining PM2.5 with weather and traffic data using H3 hexagon spatial indexing)
- EDA jupyter notebooks for 3 data sources ([OpenAQ](voi/v2_improved_oAQ_oM_J/openaq_2023_eda.ipynb), [OpenMeteo](voi/v2_improved_oAQ_oM_J/openmeteo_2023_eda.ipynb), [JARTIC](voi/v2_improved_oAQ_oM_J/jartic_2023_eda.ipynb))
- Discovered: [OpenAQ 2023 data missing Jan - Jul Mid](voi/v2_improved_oAQ_oM_J/pm25_enriched_2023_EDA.ipynb)

### [0.0.2] - 2025-08-30 - [V1 basic 2023 EDA and data enrichment](voi/v1_baseline_oAQ_oM_J)
- Initial PM2.5 [enrichment pipeline](voi/v1_baseline_oAQ_oM_J/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA ([JARTIC traffic](yulia/2023smoglens_traffic_data_jartic_EDAandLinearRegression.ipynb), [OpenMeteo Weather](voi/v1_baseline_oAQ_oM_J/openmeteo_eda.ipynb)) and [baseline models](voi/v1_baseline_oAQ_oM_J/pm25_AR_analysis_modeling.ipynb) (Linear Regression, AR analysis) including hexagon H3 size selection.
- [PM2.5 Prophet model](yulia/V1_EDA_prophet/EDAonPM25.ipynb)

### [0.0.1] - 2025-08-03
- Initial project setup
