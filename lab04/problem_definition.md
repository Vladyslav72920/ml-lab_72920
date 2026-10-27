# Problem definition: income prediction on UCI Adult

**Task.** Binary classification: given demographic and employment attributes of a person from the 1994 US Census, predict whether their annual income exceeds 50K USD.

**Inputs.** 14 features: 6 numeric (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). Missing values are marked `?` (or null in the OpenML frame) and appear mainly in `workclass`, `occupation` and `native-country`.

**Output.** `income` with two classes: `<=50K` (about 76% of rows) and `>50K` (about 24%, the positive class).

**Data.** `fetch_openml(name='adult', version=2, as_frame=True)`: 48,842 rows, 15 columns including the label.

**Success metrics.** F1 of the positive class `>50K` and accuracy, both measured on a held-out 30% test set. Because of the class imbalance, a model that always predicts `<=50K` already reaches about 76% accuracy, so F1 is the primary metric and accuracy is reported for reference. The baseline is `LogisticRegression`; derived features are judged by whether they change F1 and accuracy compared with the baseline.

**Constraints.**
1. *No leakage.* The stratified 70/30 split happens before any cleaning. Imputation values (mode/median), clipping bounds (1st/99th percentile), scaler and one-hot encoder are fitted on train only. Duplicates are dropped from train only and the test set keeps its original rows.
2. *Reproducible.* Fixed seed, configuration in `config.yaml`, fitted transformer saved with `joblib`, SHA256 of the processed dataset stored in the run manifest, and the whole pipeline runs with one command (`python run_pipeline.py`).
3. *Transparent quality checks.* Schema, range, missing-value and duplicate checks report violations and never silently drop rows.

**Out of scope.** Fairness mitigation and model tuning; the data card documents known biases instead.
