# Lab 4: Data pipeline and quality

**Dataset:** UCI Adult (1994 US Census Income), loaded with `fetch_openml(name='adult', version=2, as_frame=True)`.
**Run everything with one command (from the repo root):** `python run_pipeline.py`
**Configuration:** `config.yaml` · **Full log:** `results/pipeline.log` · **Problem definition:** `problem_definition.md` · **Data card:** `data_card.md`

| Deliverable | File |
|---|---|
| Validation report | `results/validation_report.md`, `results/validation_report.json` |
| Fitted preprocessor | `results/preprocessor.joblib` |
| Processed-data hash & config | `results/processed_dataset.sha256`, `results/run_manifest.yaml` |
| Label audit | `results/label_audit.json`, `results/label_audit_sample_50.csv` |

## 1. Problem definition (summary)

The task is binary classification: predict whether a person's annual income is above $50K from 14 features (6 numeric, 8 categorical).

- **Success metrics:** F1 for the positive class `>50K` is the primary metric, and accuracy is secondary. Only ~23.9% of rows are `>50K`, so a model that always predicts `<=50K` already gets about 76% accuracy. Accuracy alone therefore says little, and F1 shows whether the model actually finds high earners.
- **Constraints:** no train/test leakage, raw data is never modified, and the whole pipeline is reproducible with one command and a fixed seed (42).

The full text is in `problem_definition.md`.

## 2. Pipeline overview

```
fetch_openml ──► 1. ingest      (snapshot + log, no modification)
             ──► 2. validate    (schema, ranges, categories, missing, duplicates → report, no drops)
             ──► 3. split 70/30 stratified   ◄── BEFORE any cleaning
                     │ train                                  │ test
                     ▼                                        │
                 drop exact duplicates (train only)           │ untouched
                     ▼                                        ▼
                 FIT: '?'→NaN → impute (mode/median) → clip P1–P99 
                     └──────────── same fitted transformer, transform() only ──────────┘
             ──► 4. derived features (capital_net, hours_per_age)
             ──► 5. LogisticRegression baseline (with / without derived features)
             ──► 6. label-quality audit (conflicts & manual sample)
             ──► 7. SHA256 of processed data + run manifest + data card
```

## 3. Ingestion

The raw data is read once and saved unchanged to `data/raw/adult_raw.csv`. Every later step works on copies, so the raw snapshot can always be evaluated.

- Logged shape: **48842 rows × 15 columns**, which matches the expected 48842 × 15 defined in `config.yaml`.

## 4. Validation

The schema is defined in `dqpipe.py` and `config.yaml`:

| Check | Rule |
|---|---|
| Types | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week = `int`; others = `category` |
| Ranges | age 17–90, hours-per-week 1–99, capital-gain 0–99999, capital-loss 0–4356 |
| Categories | every value must be in the data-dictionary list for its column |
| Null policy | `?` means missing and is an allowed category |

**Result: 0 schema violation groups.** All types, ranges, and categories are valid. Violations are reported in `results/validation_report.md`; no rows are silently dropped at this step.

**Missing values (`?`)**

Missing values are concentrated in specific columns: `workclass` (5.7%), `occupation` (5.8%), and `native-country` (1.8%). 
The missingness is **not missing at random**. For instance, people who have never worked will naturally lack an occupation. Dropping these rows would skew the dataset, so the pipeline uses **imputation** instead.

**Duplicates:** Exact duplicate rows (all 15 columns identical) exist in the raw dataset. At this validation step, they are only reported; removal happens after the split, in the train set only.

## 5. Train/test split

| Split | Rows | `>50K` rate |
|---|---|---|
| Train | 34189 | 23.9% |
| Test | 14653 | 23.9% |

The split is 70/30, stratified by the label (seed 42), so both parts have the same class balance. It is done **before any cleaning**, ensuring no statistics computed later can include test rows.

## 6. Cleaning and transformation (leakage-free)

| Step | What | Fitted on |
|---|---|---|
| 1 | `'?'` / NaN → missing | stateless |
| 2 | Drop exact duplicates: **24 rows removed** (34189 → 34165) | train only |
| 3 | Numeric imputation: median | train only |
| 4 | Categorical imputation: mode (most frequent) | train only |
| 5 | Outlier clipping to 1st / 99th percentile | train only |
| 6 | StandardScaler (numeric) & One-hot encoding | train only (inside evaluation pipeline) |

- **Duplicates:** Duplicates are only dropped from the train set. The test set must represent real incoming data, and removing rows there would invalidate the evaluation.
- **Clipping & Imputation:** Extreme values and missing data are handled using boundaries (P1/P99) and central tendencies (median/mode) calculated **only on the train set**. Test values are transformed using these exact saved boundaries.
- The fitted transformer is saved with `joblib.dump` to `results/preprocessor.joblib`.

**Leakage checklist**

| Possible leak | How it is prevented |
|---|---|
| Imputation values computed on all data | Imputers fitted on train only |
| Clip percentiles computed on all data | Clipper fitted on train only; test clipped with train bounds |
| Scaler mean/std computed on all data | Scaler fitted on train only (in `evaluate` pipeline) |
| Deduplication across train and test | Runs inside train only, after the split |
| Derived features | Row-wise formulas with no fitted state |

## 7. Derived features

- **`capital_net = capital-gain − capital-loss`**: A single signed signal for investment income.
- **`hours_per_age = hours-per-week / (age − 16)`**: Work intensity relative to the number of possible working years.

*Note: Specific Accuracy and F1 metrics for LogisticRegression are calculated dynamically during the pipeline run and logged directly to `results/metrics.md` and `results/metrics.json`.* 
Generally, non-linear derived features (like ratios) provide slight performance boosts to linear models (like Logistic Regression) because they introduce structures the model cannot build itself.

## 8. Label quality

Adult labels come from self-reported census answers. We check for label consistency and dictionary validity.

**8.1 Duplicate-conflict audit**

- **All 14 features:** 5 conflicting groups, 10 rows total. These are identical inputs with conflicting `>50K` / `<=50K` labels.
- **Without `fnlwgt`:** 1011 conflicting groups, 3807 rows. Removing the census weight reveals significant ambiguity for otherwise identical demographic profiles.
- These rows are **kept, not modified**, because there is no ground-truth way to resolve which label is correct. 

**8.2 Manual re-check of 50 random rows**

50 rows were drawn (seed 42) and saved to `results/label_audit_sample_50.csv`. Automated checks flagged **0 rows** for logical impossibilities (e.g., Husband with Female sex, age/education mismatch). 

## 9. Reproducibility

The processed dataset is concatenated, written with a deterministic format, and hashed:

| File | SHA256 |
|---|---|
| processed dataset | `0d39587e84b3a6ade9cd0ea748024bb75bdc4cd80c70520b24f0741343a95cb1` |

- The hash is stored in `results/processed_dataset.sha256` and `results/run_manifest.yaml` alongside package versions and the full configuration.
- **Verification:** Re-running `python run_pipeline.py` with the same `config.yaml` produces the exact same hash due to the fixed seed (42).

## 10. Conclusions

1. **The raw data is structurally valid:** 0 schema violations, though missing values and exact duplicates require careful handling.
2. **Strict isolation prevents leakage:** Every learned parameter (medians, modes, percentiles) is derived strictly from the training set *after* the split.
3. **Inherent dataset limitations:** Ambiguity exists (conflicting labels for identical inputs, especially when ignoring `fnlwgt`). Furthermore, the dataset suffers from historical imbalances (majority male/White) requiring awareness before using the model for decision-making (as noted in the Data Card).