## Unreleased

━━━━━━ PROJECT PRESENTATION SUBMISSION THU 05/09/2025 ━━━━━━

### [0.1.1] - V5 Extended Feature Enrichment (Planned)
- Additional data sources integration w/ new features
  - Explore and potentially add features from other already downloaded datasets.
  - Get more data from elsewhere. Need some outside of Japan pm2.5 measurements to be able to predict delayed transmission to Japan. Potentially data from Korea we have already or some low res satelite data paired with weather.
  - This dataset from the Climate Data Store API connected the yellow dust dataset (NetCDF zip) from the Copernicus Atmosphere Monitoring Service, including PM2.5, PM10, and wind from China. [Yellow Dust Preprocessing EDA](https://github.com/Bard89/smoglens/blob/develop/yulia/V5_yellow_dust_analysis/yellow_dust_preprocessingEDA.ipynb)

━━━━━━ WE ARE HERE 31/08/2025 ━━━━━━

### [0.1.0] - 2025-08-31 - [V4 Multi-Year Analysis (2023-2025)](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC) (In progress)
- Extended temporal coverage: June 2023 - June 2025
- **Key finding:** Spatial mismatch between datasets - OpenMeteo's 30 hexagons vs OpenAQ's 643 limits nearest-neighbor enrichment effectiveness
- **Individual dataset EDAs:**
  - [OpenAQ PM2.5 Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_2025_eda.ipynb) - 643 hexagons, 80% PM2.5 completeness
  - [OpenMeteo Weather Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_2025_eda.ipynb) - 30 hexagons, 100% temporal coverage
  - [JARTIC Traffic Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_2025_eda.ipynb) - 1,018 hexagons, 100% completeness
  - [NASA Weather Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/nasa_weather_2023_2025_eda.ipynb) - 15 hexagons, May 2024 missing
- **OpenAQ + OpenMeteo + JARTIC integrated analysis:**
  - [Missing Data Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/OpenAQ_OpenMeteo_JARTIC/missing_data_analysis.ipynb)
  - [Feature Correlation Analysis](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/OpenAQ_OpenMeteo_JARTIC/feature_correlation_analysis.ipynb)
  - [Feature Selection Summary](voi/V4_enrichement_OPEANAQ_OPENMETEO_JARTIC/OpenAQ_OpenMeteo_JARTIC/feature_selection_summary.ipynb)

## Released

### [0.0.4] - 2025-08-30 - [V3 fixed 2023 Data Enrichment](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- OpenAQ 2023 data incomplete (Jan-Jul missing, API broken)
- Accumulation of one dataset from mid 2023 to mid 2025.
- Comprehensive EDAs on available years ([OpenAQ 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_2025_eda.ipynb), [OpenMeteo 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_2025_eda.ipynb), [JARTIC 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_2025_eda.ipynb)). Based on the results some of the OpenMeteo data need to be re-downloaded and some JARTIC needs to be reprocessed. Both in progress, should be ready on 28/8/2025.
- [OpenMeteo data consistency analysis](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_data_consistency_analysis_betwee_2_monthly_datasets.ipynb) - identified 5x spatial resolution difference between 2 monthly downloads


### [0.0.3] - 2025-08-30 - [V2 improved 2023 EDA and data Enrichment](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Improved [enrichment pipeline](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_2023.ipynb) (combining PM2.5 with weather and traffic data using H3 hexagon spatial indexing)
- EDA jupyter notebooks for 3 data sources ([OpenAQ](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_eda.ipynb), [OpenMeteo](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_eda.ipynb), [JARTIC](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_eda.ipynb))
- Discovered: [OpenAQ 2023 data missing Jan - Jul Mid](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_enriched_2023_EDA.ipynb)

### [0.0.2] - 2025-08-30 - [V1 basic 2023 EDA and data enrichment](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Initial PM2.5 [enrichment pipeline](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA ([JARTIC traffic](yulia/2023smoglens_traffic_data_jartic_EDAandLinearRegression.ipynb), [OpenMeteo Weather](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_eda.ipynb)) and [baseline models](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_AR_analysis_modeling.ipynb) (Linear Regression, AR analysis) including hexagon H3 size selection.
- [PM2.5 Prophet model](yulia/V1_EDA_prophet/EDAonPM25.ipynb)

### [0.0.1] - 2025-08-03
- Initial project setup
