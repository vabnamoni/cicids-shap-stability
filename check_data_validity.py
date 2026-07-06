"""
Data-validity diagnostics for the CIC-IDS2017 pipeline.

Checks that defend (or challenge) the near-perfect detection scores:
  1. Exact duplicate rows in the full dataset (a documented CIC-IDS2017 issue).
  2. Duplicate leakage: identical feature rows appearing in BOTH train and test
     after the split (inflates accuracy via memorisation).
  3. Per-class distribution across train/val/test (are rare classes viable?).
  4. Single-feature label leakage: any feature that near-perfectly separates a
     class (e.g. Destination Port acting as a proxy label).

Reuses the RQ1 preprocessing unchanged.

    python check_data_validity.py --data_dir ..\\data\\cicids2017 --out_dir ..\\outputs\\validity
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rq1_pipeline import (
    SEED, clean, load_data, resolve_label_column, stratified_splits,
)


def check_duplicates(X, y):
    """Exact duplicate feature rows, overall and within-class."""
    n = len(X)
    dup_mask = X.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    # Duplicates that also share the same label (true redundant samples) vs
    # duplicates with conflicting labels (ambiguous/mislabelled).
    Xy = X.copy()
    Xy["__label__"] = y.values
    dup_with_label = int(Xy.duplicated(keep="first").sum())

    return {
        "total_rows": n,
        "exact_duplicate_feature_rows": n_dup,
        "duplicate_fraction": round(n_dup / n, 4),
        "duplicate_rows_incl_label": dup_with_label,
        "duplicates_with_conflicting_label": n_dup - dup_with_label,
    }


def check_split_leakage(X, y):
    """Do identical feature rows appear in both train and test partitions?"""
    y_enc = LabelEncoder().fit_transform(y)
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)

    # Hash each row's feature tuple for fast set membership.
    def row_hashes(df):
        return pd.util.hash_pandas_object(df, index=False)

    h_tr = set(row_hashes(X_tr).values)
    h_te = row_hashes(X_te).values
    overlap = int(np.isin(h_te, list(h_tr)).sum())

    return {
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
        "test_rows_also_in_train": overlap,
        "test_leakage_fraction": round(overlap / len(X_te), 4),
    }


def check_class_split(X, y):
    """Per-class counts across train/val/test."""
    y_enc = LabelEncoder().fit_transform(y)
    le = LabelEncoder().fit(y)
    _, _, _, y_tr, y_val, y_te = stratified_splits(X, y_enc)
    names = list(le.classes_)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "class": name,
            "train": int((y_tr == i).sum()),
            "val": int((y_val == i).sum()),
            "test": int((y_te == i).sum()),
        })
    return pd.DataFrame(rows)


def check_feature_leakage(X, y, top=10):
    """Flag features that near-perfectly separate any single class.

    For each feature, we measure how well a simple threshold (the class-
    conditional median) separates that class from the rest using AUC. A feature
    with AUC ~1.0 for some class may be acting as a label proxy.
    """
    from sklearn.metrics import roc_auc_score

    y_enc = LabelEncoder().fit_transform(y)
    classes = np.unique(y_enc)
    results = []
    # Subsample for speed if huge.
    if len(X) > 200000:
        idx = np.random.RandomState(SEED).choice(len(X), 200000, replace=False)
        Xs, ys = X.iloc[idx], y_enc[idx]
    else:
        Xs, ys = X, y_enc

    for feat in Xs.columns:
        col = Xs[feat].values.astype(float)
        if np.all(col == col[0]):
            continue
        best_auc, best_cls = 0.0, None
        for c in classes:
            binary = (ys == c).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                continue
            try:
                auc = roc_auc_score(binary, col)
                auc = max(auc, 1 - auc)  # direction-agnostic
            except ValueError:
                continue
            if auc > best_auc:
                best_auc, best_cls = auc, c
        if best_cls is not None:
            results.append({"feature": feat, "max_class_auc": round(best_auc, 4),
                            "class": best_cls})
    df = pd.DataFrame(results).sort_values("max_class_auc", ascending=False)
    le = LabelEncoder().fit(y)
    df["class"] = df["class"].map(lambda i: le.classes_[int(i)])
    return df.head(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs/validity")
    ap.add_argument("--no_consolidate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + cleaning...")
    raw = load_data(args.data_dir)
    label_col = resolve_label_column(raw)
    X, y, _ = clean(raw, label_col, consolidate=not args.no_consolidate)

    print("\n[1] Duplicate rows...")
    dup = check_duplicates(X, y)
    print(json.dumps(dup, indent=2))

    print("\n[2] Train/test split leakage...")
    leak = check_split_leakage(X, y)
    print(json.dumps(leak, indent=2))

    print("\n[3] Per-class split distribution...")
    split_df = check_class_split(X, y)
    print(split_df.to_string(index=False))

    print("\n[4] Single-feature label leakage (top suspects by AUC)...")
    feat_leak = check_feature_leakage(X, y)
    print(feat_leak.to_string(index=False))

    # Save everything
    with open(os.path.join(args.out_dir, "validity_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"duplicates": dup, "split_leakage": leak}, f, indent=2)
    split_df.to_csv(os.path.join(args.out_dir, "class_split_distribution.csv"), index=False)
    feat_leak.to_csv(os.path.join(args.out_dir, "feature_leakage_suspects.csv"), index=False)

    # Interpretation hints
    print("\n=== INTERPRETATION ===")
    if dup["duplicate_fraction"] > 0.05:
        print(f"- {dup['duplicate_fraction']:.1%} of rows are exact duplicates. This is a "
              "known CIC-IDS2017 issue. Consider deduplicating and re-reporting, "
              "or explicitly justify keeping them.")
    else:
        print(f"- Duplicate rows are low ({dup['duplicate_fraction']:.1%}); not a major concern.")
    if leak["test_leakage_fraction"] > 0.01:
        print(f"- {leak['test_leakage_fraction']:.1%} of test rows are identical to a "
              "train row. This inflates accuracy via memorisation. Deduplicate "
              "BEFORE splitting, then re-run RQ1.")
    else:
        print(f"- Test/train row overlap is low ({leak['test_leakage_fraction']:.1%}); "
              "leakage via duplicates is not a major concern.")
    print("- Any feature with max_class_auc > 0.99 may be a label proxy; inspect "
          "whether it is legitimately predictive or leaks the label.")
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()