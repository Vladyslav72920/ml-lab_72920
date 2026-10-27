from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

import dqpipe as dq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-csv", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    np.random.seed(seed)
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(res / "pipeline.log", mode="w")])
    log = dq.log

    raw = dq.ingest(cfg, args.raw_csv)
    snap = Path(cfg["data"]["raw_snapshot"])
    snap.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(snap, index=False)
    log.info("Raw snapshot saved to %s", snap)

    report = dq.validate(raw, cfg)
    dq.write_validation_report(report, res)
    log.info("Validation: %s, %d violation entries, %d exact duplicate rows",
             report["status"], len(report["violations"]), report["exact_duplicate_rows"])
    for v in report["violations"]:
        log.warning("VIOLATION %s", v)

    df = dq.standardize_target_name(raw)
    pos_rate = float((df[dq.TARGET] == dq.POSITIVE_LABEL).mean())

    train, test = train_test_split(
        df, test_size=cfg["split"]["test_size"], random_state=seed,
        stratify=df[dq.TARGET] if cfg["split"]["stratify"] else None)
    n_train_raw = len(train)
    log.info("Split: train=%d, test=%d", len(train), len(test))

    if cfg["cleaning"]["drop_train_duplicates"]:
        train = train.drop_duplicates()
    dup_removed = n_train_raw - len(train)
    log.info("Dropped %d exact duplicate rows from train only; test untouched (%d rows)",
             dup_removed, len(test))

    cleaner = dq.DataCleaner(lower_q=cfg["cleaning"]["clip_lower_q"],
                             upper_q=cfg["cleaning"]["clip_upper_q"],
                             missing_token=cfg["cleaning"]["missing_token"])
    cleaner.fit(train[dq.FEATURES])
    joblib.dump(cleaner, cfg["paths"]["preprocessor"])
    log.info("Fitted DataCleaner on train, saved to %s", cfg["paths"]["preprocessor"])

    def apply(part: pd.DataFrame) -> pd.DataFrame:
        out = cleaner.transform(part[dq.FEATURES])
        out[dq.TARGET] = part[dq.TARGET].astype(object).values
        return out

    train_c, test_c = apply(train), apply(test)

    train_f, test_f = dq.add_derived_features(train_c), dq.add_derived_features(test_c)
    mi = cfg["model"]["max_iter"]
    variants = {
        "without_derived": [],
        "with_capital_net": ["capital_net"],
        "with_hours_per_age": ["hours_per_age"],
        "with_both": dq.DERIVED,
    }
    metrics = {}
    for name, extra in variants.items():
        metrics[name] = dq.evaluate(train_f, test_f, extra, seed, mi)
        log.info("LogReg %-18s accuracy=%.4f  F1=%.4f", name,
                 metrics[name]["accuracy"], metrics[name]["f1"])
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rows = ["| variant | accuracy | F1 |", "|---|---|---|"]
    rows += [f"| {k} | {m['accuracy']:.4f} | {m['f1']:.4f} |" for k, m in metrics.items()]
    (res / "metrics.md").write_text("\n".join(rows) + "\n")

    audit_all = dq.duplicate_conflict_audit(df, dq.FEATURES)
    audit_nofnl = dq.duplicate_conflict_audit(df, [c for c in dq.FEATURES if c != "fnlwgt"])
    log.info("Duplicate-conflict audit (14 features): %s", audit_all)
    log.info("Duplicate-conflict audit (without fnlwgt): %s", audit_nofnl)
    sample = dq.sample_audit(df, cfg, seed)
    sample.to_csv(res / "label_audit_sample_50.csv", index=False)
    flagged = int((sample["auto_issues"] != "").sum())
    log.info("50-row dictionary check: %d rows flagged (see results/label_audit_sample_50.csv)", flagged)
    (res / "label_audit.json").write_text(json.dumps(
        {"duplicate_conflicts_all_features": audit_all,
         "duplicate_conflicts_without_fnlwgt": audit_nofnl,
         "sample_rows": len(sample), "sample_flagged": flagged}, indent=2))

    processed = pd.concat([train_f.assign(split="train"), test_f.assign(split="test")],
                          ignore_index=True)
    ppath = Path(cfg["paths"]["processed"])
    dq.write_csv_deterministic(processed, ppath)
    digest = dq.sha256_file(ppath)
    log.info("SHA256(%s) = %s", ppath, digest)
    (res / "processed_dataset.sha256").write_text(f"{digest}  {ppath}\n")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed_file": str(ppath), "processed_sha256": digest,
        "rows": {"raw": len(raw), "train_raw": n_train_raw, "train": len(train), "test": len(test)},
        "metrics": metrics,
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "config": cfg,
    }
    (res / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    dq.write_data_card(Path(cfg["paths"]["data_card"]), {
        "n_rows": len(raw), "n_cols": raw.shape[1], "n_train_raw": n_train_raw,
        "n_train": len(train), "n_test": len(test), "dup_removed": dup_removed, "seed": seed,
        "pos_rate": pos_rate, "missing": report["missing"], "processed_path": str(ppath),
        "sha256": digest, "audit_all": audit_all, "audit_nofnl": audit_nofnl,
        "sample_flagged": flagged})
    log.info("Done. Outputs in %s/, data card in %s", res, cfg["paths"]["data_card"])


if __name__ == "__main__":
    main()
