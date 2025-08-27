## Unreleased

- [0.0.2] and [0.0.3] will be ready for release after fixing the structure in https://github.com/Bard89/smoglens/pull/24

### [0.1.1] - V5 Extended Feature Enrichment (Planned)
- Additional data sources integration w/ new features
  - Explore and potentially add features from other already downloaded datasets.
  - Get more data from elsewhere. Need some outside of Japan pm2.5 measurements to be able to predict delayed transmission to Japan. Potentially data from Korea we have already or some low res satelite data paired with weather. 

### [0.1.0] - V4 Multi-Year mid 2023 - mid 2025 analysis and advanced modeling (Planned, first version 28/8/2025)
- Non-autoregressive models with enriched dataset, deep learning models
- Multi-year validation (2023-2025)

------------------------------------ WE ARE HERE 27/08/2025-------------------------------------

### [0.0.4] - [V3 fixed 2023 Data Enrichment](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC) (In progress)
- OpenAQ 2023 data incomplete (Jan-Jul missing, API broken)
- Accumulation of one dataset from mid 2023 to mid 2025.
- Comprehensive EDAs on available years ([OpenAQ 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_2025_eda.ipynb), [OpenMeteo 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_2025_eda.ipynb), [JARTIC 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_2025_eda.ipynb)). Based on the results some of the OpenMeteo data need to be re-downloaded and some JARTIC needs to be reprocessed. Both in progress, should be ready on 28/8/2025.
- [OpenMeteo data consistency analysis](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_data_consistency_analysis_betwee_2_monthly_datasets.ipynb) - identified 5x spatial resolution difference between 2 monthly downloads


### [0.0.3] - [V2 improved 2023 EDA and data Enrichment](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Improved [enrichment pipeline](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_2023.ipynb) (combining PM2.5 with weather and traffic data using H3 hexagon spatial indexing)
- EDA jupyter notebooks for 3 data sources ([OpenAQ](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_eda.ipynb), [OpenMeteo](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_eda.ipynb), [JARTIC](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_eda.ipynb))
- Discovered: [OpenAQ 2023 data missing Jan - Jul Mid](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_enriched_2023_EDA.ipynb)

### [0.0.2] - [V1 basic 2023 EDA and data enrichment](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Initial PM2.5 [enrichment pipeline](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA ([JARTIC traffic](yulia/2023smoglens_traffic_data_jartic_EDAandLinearRegression.ipynb), [OpenMeteo Weather](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_eda.ipynb)) and [baseline models](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_AR_analysis_modeling.ipynb) (Linear Regression, AR analysis) including hexagon H3 size selection.
- [PM2.5 Prophet model](yulia/EDAonPM25.ipynb))

## Released

### [0.0.1] - 2025-01-06
- Initial project setup
