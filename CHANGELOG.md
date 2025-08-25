## Unreleased

### [0.1.2] - V5 Extended Feature Enrichment (Planned needed/possible based on the 0.1.0)
- Additional data sources integration
- New features from external datasets
- Enhanced enrichment pipeline

### [0.1.1] - V4 Multi-Year Analysis & Modeling (Planned if needed/possible based on the 0.1.0)
- Complete 2023, 2024, and partial 2025 data
- Comprehensive EDAs across all years
- Full modeling pipeline with multiyear validation

### [0.1.0] - V3 Advanced Modeling (Planned)
- Non-autoregressive models with enriched dataset, deep learning models

### [0.0.4] - V3 fixed 2023 Data Enrichment (In progress)
- Complete OpenAQ data for full year 2023 (currently missing Jan-Jul; being downloaded)

------------------------------------ WE ARE HERE 26/08/2025-------------------------------------

### [0.0.3] - [V2 improved 2023 EDA and data Enrichment](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Improved [enrichment pipeline](voi/V2_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_2023.ipynb) (combining PM2.5 with weather and traffic data using H3 hexagon spatial indexing)
- EDA jupyter notebooks for 3 data sources (OpenAQ, OpenMeteo, JARTIC)
- Discovered: OpenAQ 2023 data missing Jan 1 - Jul Mid

### [0.0.2] - [V1 basic 2023 EDA and data enrichment](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC)
- Initial PM2.5 [enrichment pipeline](voi/V1_enrichement_OPEANAQ_OPENMETEO_JARTIC/pm25_hexagon_enrichment_script.ipynb) with nearest-neighbor approximation for missing sensor data
- Enrichment process: merging PM2.5, weather (temperature, humidity, precipitation), and traffic data by location into H3 hexagons.
- Basic EDA and [baseline models](/voi/pm25_ar_analysis.ipynb) (Linear Regression... )

## Released

### [0.0.1] - 2025-01-06
- Initial project setup
