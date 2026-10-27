# Lab 4: Data pipeline and quality (UCI Adult)

## One command
```
pip install -r requirements.txt
python run_pipeline.py
```
Offline mode (CSV with the same columns as the OpenML frame, label column `class`):
```
python run_pipeline.py --raw-csv adult.csv
```

## What it produces
| Lab item | Output |
|---|---|
| 1 Problem definition | `problem_definition.md` |
| 2 Ingestion (rows/cols logged) | `results/pipeline.log`, `data/raw/adult_raw.csv` |
| 3 Validation | `results/validation_report.md` / `.json` |
| 4 Cleaning | `results/preprocessor.joblib` |
| 5 Derived features, accuracy/F1 with and without | `results/metrics.md` / `.json` |
| 6 Label quality | `results/label_audit.json`, `results/label_audit_sample_50.csv` |
| 7 Hash, config, data card | `results/processed_dataset.sha256`, `results/run_manifest.yaml`, `data_card.md` |

## Manual part (item 6)
Open `results/label_audit_sample_50.csv`, look at the 50 rows against the data dictionary and fill the `manual_ok` column (the `auto_issues` column is only a hint).

Note: `results/preprocessor.joblib` is a pickled `DataCleaner` from `dqpipe.py`; load it from the project root.
