## Unreleased

## Released

### [0.0.2] - 2024-08-30 - [V1 basic 2023 EDA and data enrichment](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Initial PM2.5 [enrichment pipeline](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA ([JARTIC traffic](yulia/2023smoglens_traffic_data_jartic_EDAandLinearRegression.ipynb), [OpenMeteo Weather](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/openmeteo_eda.ipynb)) and [baseline models](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_AR_analysis_modeling.ipynb) (Linear Regression, AR analysis) including hexagon H3 size selection.
- [PM2.5 Prophet model](yulia/EDAonPM25.ipynb)

### [0.0.1] - 2025-01-06
- Initial project setup
