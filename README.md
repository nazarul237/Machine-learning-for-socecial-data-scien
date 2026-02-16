# Go-Around Prediction (Unstable Approach Proxy)

## Research question
Using landing and operational approach data (airport/runway/ILS, aircraft class/segment, and 40NM crossing geometry),
can a model predict whether a landing attempt will result in a go-around (proxy for an unstable approach)?

## Repository contents
- step1_check_data.py
- step2_prepare_data.py
- step3_train_models.py
- step4_leakage_check.py
- step5_feature_importance.py
- step6_clean_features.py
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
