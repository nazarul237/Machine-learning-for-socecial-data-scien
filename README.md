# Go-Around Prediction (Unstable Approach Proxy)

## Research question
Using landing and operational approach data (airport/runway/ILS, aircraft class/segment, and 40NM crossing geometry),
can a model predict whether a landing attempt will result in a go-around (proxy for an unstable approach)?

#RQ2 (Model comparison): Between Logistic Regression, Random Forest and XGBoost, which model performs best on a future time period (2023H1) using ROC-AUC and precision/recall/F1 as evaluation metrics? 

#RQ3 (Imbalance and operational trade-off): Given that go-arounds are rare in the real test period, what trade-off exists between identifying go-arounds (recall) and avoiding false alarms (precision), and how does this affect the suitability of each model for screening versus alerting? 

#RQ4 (Robustness / operational availability): How sensitive is performance to predictors that may not be operationally available at the 40NM decision point (e.g., timing-to-landing features), and does removing them reveal strong dependence consistent with potential leakage? 

#RQ5 Leakage / operational availability: How much does performance change when timing-derived information is removed (e.g., testing with vs without the timing feature), and does this suggest potential leakage or reliance on non-operational features? 

#RQ6 Interpretability / drivers: Which approach/operational features (e.g., ILS, runway, market segment, aircraft class) are most associated with go-arounds in the final model (using model feature importance / coefficients)?

## Repository contents
- step1_check_data.py
- step2_prepare_data.py
- step3_train_models.py
- step4_leakage_check.py
- step5_feature_importance.py
- step6_clean_features.py
- step7_threshold_tuning.py
- step8_train_xgboost.py
- results/ (CSV outputs)

## Data
Place the raw dataset here:
- data/osn23_landings_merged.parquet.gzip

Running step2_prepare_data.py will generate:
- data/train_2019_2022_sample.parquet
- data/test_2023H1_full.parquet

Note: data files are not included in this GitHub repo due to size.

## How to run (in order)
```bash
pip install -r requirements.txt
python step1_check_data.py
python step2_prepare_data.py
python step3_train_models.py
python step4_leakage_check.py
python step5_feature_importance.py
python step6_clean_features.py
python step7_threshold_tuning.py
python step8_train_xgboost.py
