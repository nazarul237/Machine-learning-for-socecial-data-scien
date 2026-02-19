# Machine Learning for Social Data Science — Go-Around Prediction (Unstable Approach Proxy)

**GitHub repository (replication link):** https://github.com/nazarul237/Machine-learning-for-socecial-data-scien

## Purpose and scope
This repository contains the full reproducible workflow for my Machine Learning project. The objective is to predict whether a landing attempt results in a **go-around (GoA)**, used here as a practical proxy for an unstable approach outcome. Because go-arounds are rare events in the real test period, the evaluation emphasises **ROC–AUC** alongside **precision, recall, and F1-score**, which are more informative than accuracy under severe class imbalance.

## Research questions
**RQ1:** Using operational and approach context (e.g., airport/runway/ILS, aircraft class/segment) and 40NM crossing features, can a model predict whether a landing attempt will result in a go-around?

**RQ2 (Model comparison):** Between Logistic Regression, Random Forest, and XGBoost, which model performs best on a future holdout period (2023H1) when assessed using ROC–AUC and precision/recall/F1?

**RQ3 (Imbalance trade-off):** Under real-world class imbalance, what trade-off exists between identifying go-arounds (recall) and avoiding false alarms (precision), and how does this influence model suitability for screening versus alerting?

**RQ4–RQ5 (Robustness / operational availability and leakage risk):** How sensitive is performance to timing-derived variables (e.g., `C40_to_landing_min`) that may not be operationally available at the 40NM decision point, and does removing them materially reduce performance?

**RQ6 (Interpretability):** Which operational predictors are most strongly associated with go-arounds according to model feature importance / coefficients?

## Repository structure and key files
This repository is organised as a sequential, script-based pipeline. The scripts are named by step number to support straightforward replication.

**Scripts (run in order):**
- `step1_check_data.py` — loads data and performs basic integrity checks
- `step2_prepare_data.py` — prepares modelling datasets using a temporal split (train: 2019–2022; test: 2023H1)
- `step3_train_models.py` — trains and evaluates Logistic Regression and Random Forest baselines
- `step4_leakage_check.py` — re-evaluates performance after removing candidate timing-derived feature(s)
- `step5_feature_importance.py` — exports Logistic Regression coefficient-based feature importance
- `step6_clean_features.py` — cleans feature names and exports a readable “top features” table
- `step7_threshold_tuning.py` — tunes the classification threshold to illustrate precision/recall/F1 trade-offs
- `step8_train_xgboost.py` — trains and evaluates XGBoost for extended model comparison
- `step9_ROC_CURVE_plot` — generates comparative ROC curves on the 2023H1 test set (this file may not have a `.py` extension; it can still be executed with `python step9_ROC_CURVE_plot`)

**Outputs (created in `results/`):**
- `results/logreg_feature_importance.csv`
- `results/top20_nice_features.csv`
- `results/threshold_tuning_summary.csv`
- ROC curve figure(s), e.g. `results/roc_curves_2023H1.png`

## Data requirements and placement
To reproduce results, the scripts expect data files to be available locally. There are two supported replication routes:

### Route A (preferred for marking): reproduce directly from prepared train/test files
Place these prepared datasets in a folder named `data/` at the repository root:
- `data/train_2019_2022_sample.parquet`
- `data/test_2023H1_full.parquet`

This route allows a reproducer to run the modelling steps without needing the full raw dataset.

### Route B: reproduce from the raw merged dataset
If the raw dataset is available, place it here:
- `data/osn23_landings_merged.parquet.gzip`

Then run `step2_prepare_data.py` to generate the prepared train/test parquet files used by later steps.

> Note: If the raw dataset is not committed to GitHub due to file size or sharing restrictions, Route A is recommended for replication. In all cases, the repository is designed so the workflow runs end-to-end given the expected files in `data/`.

## Environment setup (installation and reproducibility)
The repository is intended to run on a clean machine using a standard Python environment.

### 1) Create and activate a virtual environment
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies
Install required packages using:
pip install -r requirements.txt

Important (to avoid installation failures on other machines): the requirements.txt file should contain portable package names (not local paths such as @ file:///...). If you encounter install errors, replace requirements.txt with the following minimal portable set and reinstall:
pandas
numpy
pyarrow
scikit-learn
xgboost
matplotlib

Reproducing the results (run order)

From the repository root, execute the following commands in order:
python step1_check_data.py
python step2_prepare_data.py
python step3_train_models.py
python step4_leakage_check.py
python step5_feature_importance.py
python step6_clean_features.py
python step7_threshold_tuning.py
python step8_train_xgboost.py
python step9_ROC_CURVE_plot

Expected results and artefacts

If the pipeline completes successfully, you should obtain:

Printed evaluation summaries in the terminal (ROC–AUC, precision, recall, F1; and confusion matrices where implemented).

Saved CSV outputs in results/ (feature importance, cleaned top features, threshold tuning summary).

A comparative ROC figure saved in results/ (e.g., results/roc_curves_2023H1.png), suitable for inclusion in the written report.

Methodological notes (interpretation and validity)

Temporal holdout: the 2023H1 dataset is treated as a strict future test set to better reflect deployment-style performance and reduce temporal information leakage.

Class imbalance: because GoA events are rare, accuracy is not emphasised; instead ROC–AUC and precision/recall/F1 are reported to capture both ranking performance and operational trade-offs.

Operational availability and leakage check: the workflow explicitly tests sensitivity to timing-derived predictors (e.g., C40_to_landing_min) that may embed post-event information or be unavailable at the 40NM decision point.

Code organisation and commenting

All scripts are structured as sequential steps with clear naming. Inline comments and module-level docstrings are used to document assumptions, inputs/outputs, and the rationale for key methodological choices, supporting transparent replication and assessment.

Author

SULTAN NAZARUL ISLAM — MSc Business Analytics, University of Exeter