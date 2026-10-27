# Data card: UCI Adult (Census Income), processed

## Source
- Original: UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
- Retrieved via `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)` (OpenML).
- Task: predict whether annual income exceeds 50K USD (`income` = `>50K`) from 14 demographic and employment features.

## License
- UCI lists the Adult dataset under **CC BY 4.0** (attribution required). Check the UCI/OpenML page before redistributing.

## Size
- Raw: 48842 rows x 15 columns (14 features + 1 label).
- Train/test split: 34189 / 14653 rows (70/30, stratified, seed 42).
- Train after dropping exact duplicates: 34165 rows (24 removed). Test is untouched.
- Positive class (`>50K`) share in the raw data: 23.9%.
- Processed file: `data\processed\adult_processed.csv` (SHA256 `0d39587e84b3a6ade9cd0ea748024bb75bdc4cd80c70520b24f0741343a95cb1`).

## Processing summary
Validate schema and ranges (no rows dropped) -> split -> drop train duplicates -> fit `DataCleaner` on train only
(`?` to missing, mode/median imputation, 1st/99th percentile clipping) -> apply to test -> derived features
`capital_net`, `hours_per_age`. Fitted transformer: `results/preprocessor.joblib`.

## Label quality
- Duplicate-conflict audit (all 14 features): 5 conflicting groups, 10 rows.
- Duplicate-conflict audit (without `fnlwgt`): 1011 conflicting groups, 3807 rows.
- 50-row random sample checked against the data dictionary: 0 rows flagged by the automated check.

## Known biases and limitations
- Data is from 1994, US only (about 90% of rows have `native-country` = United-States); the 50K threshold is not inflation-adjusted.
- Class imbalance: roughly 3 of 4 rows are `<=50K`, so accuracy alone is misleading, hence F1 for the positive class.
- Sex and race are imbalanced (majority male and White) and income differs strongly by group; a model can reproduce these gaps.
- Missing values are concentrated in `workclass`, `occupation`, `native-country` (workclass (5.7%), occupation (5.8%), native-country (1.8%)) and are probably not missing at random; mode imputation hides this.
- `fnlwgt` is a census sampling weight, not a personal attribute; treating it as an ordinary feature is questionable.
- Labels come from reported income, not verified, and the dataset has no ground-truth relabeling.

## Privacy concerns
- Public and anonymized (no names or IDs), but combinations of age, sex, race, marital status, and native country are quasi-identifiers and may allow re-identification when linked with other data.
- Contains sensitive attributes (race, sex, native country, marital status). Do not use for real decisions about individuals (hiring, credit) without a fairness and legal review.
