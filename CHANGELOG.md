## Unreleased

## Released

### [0.0.4] - 2025-08-30 - [V3 fixed 2023 Data Enrichment](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- OpenAQ 2023 data incomplete (Jan-Jul missing, API broken)
- Accumulation of one dataset from mid 2023 to mid 2025.
- Comprehensive EDAs on available years ([OpenAQ 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_2025_eda.ipynb), [OpenMeteo 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_2025_eda.ipynb), [JARTIC 2023-2025](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_2025_eda.ipynb))
- [OpenMeteo data consistency analysis](voi/V3_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_data_consistency_analysis_betwee_2_monthly_datasets.ipynb) - identified 5x spatial resolution difference between 2 monthly downloads

### [0.0.3] - 2025-08-30 - [V2 improved 2023 EDA and data Enrichment](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Improved [enrichment pipeline](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_2023.ipynb) (combining PM2.5 with weather and traffic data using H3 hexagon spatial indexing)
- EDA jupyter notebooks for 3 data sources ([OpenAQ](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openaq_2023_eda.ipynb), [OpenMeteo](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_2023_eda.ipynb), [JARTIC](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/jartic_2023_eda.ipynb))
- Discovered: [OpenAQ 2023 data missing Jan - Jul Mid](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_enriched_2023_EDA.ipynb)

### [0.0.2] - 2025-08-30 - [V1 basic 2023 EDA and data enrichment](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Initial PM2.5 [enrichment pipeline](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA ([JARTIC traffic](yulia/2023smoglens_traffic_data_jartic_EDAandLinearRegression.ipynb), [OpenMeteo Weather](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_eda.ipynb)) and [baseline models](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_AR_analysis_modeling.ipynb) (Linear Regression, AR analysis) including hexagon H3 size selection.
- [PM2.5 Prophet model](yulia/EDAonPM25.ipynb)

### [0.0.1] - 2025-01-06
- Initial project setup
