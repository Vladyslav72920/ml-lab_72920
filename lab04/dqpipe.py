from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("lab4")

TARGET = "income"
FEATURES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
POSITIVE_LABEL = ">50K"

SCHEMA = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}

ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
                  "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
        "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
        "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
        "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
        "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": ["<=50K", ">50K"],
}
# Null policy from the lab: '?' is the missing marker and is an allowed category.
ALLOWED = {k: sorted(set(v) | {"?"}) if k != "income" else v for k, v in ALLOWED.items()}

EDU_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
    "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}


# --------------------------------------------------------------------------
# Step 2: ingestion (read only, never modifies the data)
# --------------------------------------------------------------------------
def ingest(cfg: dict, raw_csv: str | None = None) -> pd.DataFrame:
    if raw_csv:
        log.info("Ingestion: reading local CSV %s (offline mode)", raw_csv)
        df = pd.read_csv(raw_csv)
    else:
        from sklearn.datasets import fetch_openml
        d = cfg["data"]
        log.info("Ingestion: fetch_openml(name=%r, version=%r, as_frame=True)",
                 d["openml_name"], d["openml_version"])
        df = fetch_openml(name=d["openml_name"], version=d["openml_version"], as_frame=True).frame
    n_rows, n_cols = df.shape
    log.info("Ingestion: row count = %d, column count = %d", n_rows, n_cols)
    exp = (cfg["data"]["expected_rows"], cfg["data"]["expected_cols"])
    if (n_rows, n_cols) != exp:
        raise ValueError(f"Unexpected raw shape {(n_rows, n_cols)}, expected {exp}")
    return df


# --------------------------------------------------------------------------
# Step 3: validation (reports violations, never drops rows)
# --------------------------------------------------------------------------
def standardize_target_name(df: pd.DataFrame) -> pd.DataFrame:
    """OpenML calls the label 'class'; the lab calls it 'income'. Returns a copy."""
    return df.rename(columns={"class": TARGET}) if "class" in df.columns else df.copy()


def _viol(check, column, mask, detail):
    idx = [int(i) if str(i).lstrip("-").isdigit() else str(i) for i in mask[mask].index[:10]]
    return {"check": check, "column": column, "n_rows": int(mask.sum()),
            "detail": detail, "example_row_index": idx}


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    df = standardize_target_name(df)
    ranges = cfg["validation"]["ranges"]
    missing_token = cfg["cleaning"]["missing_token"]
    report = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "violations": [],
              "missing": {}, "exact_duplicate_rows": int(df.duplicated().sum())}
    V = report["violations"]

    absent = [c for c in SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in SCHEMA]
    if absent or extra:
        V.append({"check": "columns", "column": "-", "n_rows": 0,
                  "detail": f"missing columns: {absent}; unexpected columns: {extra}",
                  "example_row_index": []})

    for col, kind in SCHEMA.items():
        if col not in df.columns:
            continue
        s = df[col]
        if kind == "int":
            num = pd.to_numeric(s, errors="coerce")
            if not pd.api.types.is_numeric_dtype(s):
                V.append(_viol("type", col, pd.Series(True, index=s.index), "dtype is not numeric"))
            bad = num.isna() | (num % 1 != 0)
            if bad.any():
                V.append(_viol("type", col, bad, "not an integer (or null)"))
            if col in ranges:
                lo, hi = ranges[col]
                out = (num < lo) | (num > hi)
                if out.any():
                    V.append(_viol("range", col, out, f"outside [{lo}, {hi}]"))
        else:
            if s.dtype.name not in ("category", "object", "string", "str"):
                V.append(_viol("type", col, pd.Series(True, index=s.index),
                               f"dtype {s.dtype} is not categorical"))
            so = s.astype(object)
            bad = so.notna() & ~so.isin(ALLOWED[col])
            if bad.any():
                vals = sorted(map(str, so[bad].unique()))[:10]
                V.append(_viol("category", col, bad, f"values not in allowed set: {vals}"))
            miss = so.isna() | (so == missing_token)
            report["missing"][col] = int(miss.sum())

    report["status"] = "PASS" if not V else "VIOLATIONS FOUND (rows kept, nothing dropped)"
    return report


def write_validation_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    L = ["# Validation report", "",
         f"- Rows: **{report['n_rows']}**, columns: **{report['n_cols']}**",
         f"- Exact duplicate rows in raw data (all 15 columns): **{report['exact_duplicate_rows']}**",
         f"- Status: **{report['status']}**", "", "## Violations", ""]
    if report["violations"]:
        L += ["| check | column | rows | detail | example row index |", "|---|---|---|---|---|"]
        for v in report["violations"]:
            L.append(f"| {v['check']} | {v['column']} | {v['n_rows']} | {v['detail']} | {v['example_row_index']} |")
    else:
        L.append("None.")
    L += ["", "## Missing values ('?' or null)", "", "| column | missing | % |", "|---|---|---|"]
    for c, n in report["missing"].items():
        L.append(f"| {c} | {n} | {100 * n / report['n_rows']:.2f} |")
    (out_dir / "validation_report.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Step 4: cleaning transformer (fit on TRAIN only)
# --------------------------------------------------------------------------
class DataCleaner(BaseEstimator, TransformerMixin):
    """'?' -> NaN, impute (mode for categorical, median for numeric), clip to train 1st/99th pct."""

    def __init__(self, numeric=tuple(NUMERIC), categorical=tuple(CATEGORICAL),
                 lower_q=0.01, upper_q=0.99, missing_token="?"):
        self.numeric = numeric
        self.categorical = categorical
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.missing_token = missing_token

    def _mark_missing(self, X):
        X = X.copy()
        for c in self.categorical:
            s = X[c].astype(object)
            X[c] = s.where(s != self.missing_token, np.nan)
        return X

    def fit(self, X, y=None):
        X = self._mark_missing(X)
        num = X[list(self.numeric)].astype(float)
        self.medians_ = num.median().to_dict()
        filled = num.fillna(pd.Series(self.medians_))
        self.lower_ = filled.quantile(self.lower_q).to_dict()
        self.upper_ = filled.quantile(self.upper_q).to_dict()
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.categorical}
        return self

    def transform(self, X):
        X = self._mark_missing(X)
        for c in self.numeric:
            X[c] = X[c].astype(float).fillna(self.medians_[c]).clip(self.lower_[c], self.upper_[c])
        for c in self.categorical:
            X[c] = X[c].fillna(self.modes_[c])
        return X


# --------------------------------------------------------------------------
# Step 5: derived features + baseline model
# --------------------------------------------------------------------------
DERIVED = ["capital_net", "hours_per_age"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not (out["age"] > 16).all():
        raise ValueError("age must be > 16 for hours_per_age")
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    out["hours_per_age"] = out["hours-per-week"] / (out["age"] - 16)
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame, extra_numeric: list[str], seed: int,
             max_iter: int) -> dict:
    """Scaler / one-hot are fitted inside the pipeline on train only."""
    num = NUMERIC + extra_numeric
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=max_iter, random_state=seed))])
    cols = num + CATEGORICAL
    y_tr = (train[TARGET] == POSITIVE_LABEL).astype(int)
    y_te = (test[TARGET] == POSITIVE_LABEL).astype(int)
    clf.fit(train[cols], y_tr)
    pred = clf.predict(test[cols])
    return {"features": cols, "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred))}


# --------------------------------------------------------------------------
# Step 6: label quality
# --------------------------------------------------------------------------
def duplicate_conflict_audit(df: pd.DataFrame, features: list[str]) -> dict:
    d = df[features + [TARGET]].astype(str)
    g = d.groupby(features)[TARGET]
    n_unique, size = g.transform("nunique"), g.transform("size")
    conflict = n_unique > 1
    return {
        "features_used": len(features),
        "rows_in_duplicate_groups": int((size > 1).sum()),
        "conflicting_groups": int(d[conflict].drop_duplicates(features).shape[0]),
        "conflicting_rows": int(conflict.sum()),
    }


def dictionary_issues(row: pd.Series, ranges: dict) -> str:
    issues = []
    for col, (lo, hi) in ranges.items():
        v = pd.to_numeric(row[col], errors="coerce")
        if pd.isna(v) or v < lo or v > hi:
            issues.append(f"{col}={row[col]} outside [{lo},{hi}]")
    for col, allowed in ALLOWED.items():
        v = row[col]
        if pd.notna(v) and v not in allowed:
            issues.append(f"{col}={v} not in data dictionary")
    edu = row["education"]
    if edu in EDU_NUM and row["education-num"] != EDU_NUM[edu]:
        issues.append(f"education={edu} but education-num={row['education-num']}")
    if row["relationship"] == "Husband" and row["sex"] == "Female":
        issues.append("relationship=Husband but sex=Female")
    if row["relationship"] == "Wife" and row["sex"] == "Male":
        issues.append("relationship=Wife but sex=Male")
    if row["capital-gain"] > 0 and row["capital-loss"] > 0:
        issues.append("warning: capital-gain and capital-loss both > 0")
    return "; ".join(issues)


def sample_audit(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    n = cfg["label_audit"]["sample_size"]
    s = df.sample(n=n, random_state=seed).copy()
    s.insert(0, "orig_row_index", s.index)
    s["auto_issues"] = s.apply(lambda r: dictionary_issues(r, cfg["validation"]["ranges"]), axis=1)
    s["manual_ok"] = ""  # fill in by hand: yes / no
    return s


# --------------------------------------------------------------------------
# Step 7: hash + data card
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_deterministic(df: pd.DataFrame, path: Path) -> None:
    """Same bytes on every OS (LF line endings, fixed float format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False, float_format="%.8g").replace("\r\n", "\n")
    path.write_bytes(text.encode("utf-8"))


def write_data_card(path: Path, i: dict) -> None:
    miss = ", ".join(f"{k} ({100 * v / i['n_rows']:.1f}%)" for k, v in i["missing"].items() if v)
    txt = f"""# Data card: UCI Adult (Census Income), processed

## Source
- Original: UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
- Retrieved via `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)` (OpenML).
- Task: predict whether annual income exceeds 50K USD (`income` = `>50K`) from 14 demographic and employment features.

## License
- UCI lists the Adult dataset under **CC BY 4.0** (attribution required). Check the UCI/OpenML page before redistributing.

## Size
- Raw: {i['n_rows']} rows x {i['n_cols']} columns (14 features + 1 label).
- Train/test split: {i['n_train_raw']} / {i['n_test']} rows (70/30, stratified, seed {i['seed']}).
- Train after dropping exact duplicates: {i['n_train']} rows ({i['dup_removed']} removed). Test is untouched.
- Positive class (`>50K`) share in the raw data: {100 * i['pos_rate']:.1f}%.
- Processed file: `{i['processed_path']}` (SHA256 `{i['sha256']}`).

## Processing summary
Validate schema and ranges (no rows dropped) -> split -> drop train duplicates -> fit `DataCleaner` on train only
(`?` to missing, mode/median imputation, 1st/99th percentile clipping) -> apply to test -> derived features
`capital_net`, `hours_per_age`. Fitted transformer: `results/preprocessor.joblib`.

## Label quality
- Duplicate-conflict audit (all 14 features): {i['audit_all']['conflicting_groups']} conflicting groups, {i['audit_all']['conflicting_rows']} rows.
- Duplicate-conflict audit (without `fnlwgt`): {i['audit_nofnl']['conflicting_groups']} conflicting groups, {i['audit_nofnl']['conflicting_rows']} rows.
- 50-row random sample checked against the data dictionary: {i['sample_flagged']} rows flagged by the automated check.

## Known biases and limitations
- Data is from 1994, US only (about 90% of rows have `native-country` = United-States); the 50K threshold is not inflation-adjusted.
- Class imbalance: roughly 3 of 4 rows are `<=50K`, so accuracy alone is misleading, hence F1 for the positive class.
- Sex and race are imbalanced (majority male and White) and income differs strongly by group; a model can reproduce these gaps.
- Missing values are concentrated in `workclass`, `occupation`, `native-country` ({miss}) and are probably not missing at random; mode imputation hides this.
- `fnlwgt` is a census sampling weight, not a personal attribute; treating it as an ordinary feature is questionable.
- Labels come from reported income, not verified, and the dataset has no ground-truth relabeling.

## Privacy concerns
- Public and anonymized (no names or IDs), but combinations of age, sex, race, marital status, and native country are quasi-identifiers and may allow re-identification when linked with other data.
- Contains sensitive attributes (race, sex, native country, marital status). Do not use for real decisions about individuals (hiring, credit) without a fairness and legal review.
"""
    path.write_text(txt, encoding="utf-8")
