# Research Notebooks

Frozen record of the research iterations. Each version directory documents one pass over the problem; later versions supersede earlier ones. New analysis notebooks should import shared logic from the `smoglens` package instead of copying it.

| Version | Focus | Status |
|---------|-------|--------|
| [v1_baseline](v1_baseline) | 2023 data, first enrichment (nearest-neighbor), AR/LinearRegression baselines, H3 resolution selection | Superseded by v4 |
| [v2_improved](v2_improved) | 2023 re-run with improved enrichment; discovered OpenAQ Jan–Jul 2023 gap | Superseded by v4 |
| [v3_fixed](v3_fixed) | Extended to 2023–2025; OpenMeteo batch-consistency diagnostic (5x resolution difference) | Superseded by v4 |
| [v4_multiyear](v4_multiyear) | 2023–2025, 9.4M records: EDAs, K-NN enrichment pipeline, LinearReg baseline, gradient-boosting ensemble (the deployed models) | Current |

Still uniquely valuable outside v4: the H3 resolution-selection analysis ([v1_baseline/pm25_AR_analysis_modeling.ipynb](v1_baseline/pm25_AR_analysis_modeling.ipynb)), the OpenMeteo two-batch consistency diagnostic ([v3_fixed](v3_fixed)), and the EDA checklist ([v1_baseline/EDA_ideas.md](v1_baseline/EDA_ideas.md)).

The v4 pipeline notebooks read source CSVs from `PROCESSED_DATA_PATH` locations and write to `SMOGLENS_DATA_PATH` (see `.env`); paths inside the notebooks are hardcoded to their original run environment and document what was actually executed. The ensemble training scripts live in [v4_multiyear/05_modeling/02_advanced/01_ensemble_training/scripts](v4_multiyear/05_modeling/02_advanced/01_ensemble_training/scripts); `train_ensemble_granular.py` produced the deployed models.
