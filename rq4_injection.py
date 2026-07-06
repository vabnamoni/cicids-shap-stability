"""
RQ4 (controlled injection variant): artifact-induced explanation distortion
as a function of corruption prevalence.

Because the natural artifact is rare in CIC-IDS2017 (~115 rows), this variant
deliberately injects negative Flow Duration into a controlled fraction of rows
and measures how detection and the SHAP importance of Flow Duration change as
the injection rate rises. This isolates the mechanism and must be reported as
SYNTHETIC injection, not naturally-occurring corruption.

    python rq4_injection.py --data_dir ..\\data\\cicids2017 --out_dir ..\\outputs\\rq4_inj ^
        --rates 0.0 0.01 0.05 0.10 0.20
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rq1_pipeline import (
    SEED, class_weight_dict, clean, load_data, resolve_label_column,
    stratified_splits,
)
from rq2_shap import _stratified_sample_idx, tree_shap_importance

ARTIFACT_FEATURE = "Flow Duration"
DEFAULT_RARE = ["Web Attack", "Infiltration", "Heartbleed", "Bot"]


def inject(X, feature, rate, rng, target_rows=None):
    """Flip `feature` to negative on a `rate` fraction of rows.

    If target_rows is given (boolean mask), injection is confined to those rows
    (e.g. a specific class); otherwise it is spread across all rows.
    """
    X = X.copy()
    pool = np.where(target_rows)[0] if target_rows is not None else np.arange(len(X))
    n_flip = int(round(rate * len(pool)))
    if n_flip == 0:
        return X, 0
    flip_idx = rng.choice(pool, size=n_flip, replace=False)
    X.iloc[flip_idx, X.columns.get_loc(feature)] = -np.abs(
        X.iloc[flip_idx, X.columns.get_loc(feature)]
    )
    return X, n_flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs/rq4_inj")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--no_consolidate", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="Remove exact duplicate rows before splitting.")
    ap.add_argument("--shap_sample", type=int, default=5000)
    ap.add_argument("--inject_class", default=None,
                    help="Confine injection to this class (e.g. 'Web Attack'). "
                         "Default spreads across all rows.")
    ap.add_argument("--rare_classes", nargs="+", default=DEFAULT_RARE)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)

    print("Loading + cleaning (clean baseline, artifact removed)...")
    raw = load_data(args.data_dir)
    label_col = resolve_label_column(raw)
    X, y, _ = clean(raw, label_col, consolidate=not args.no_consolidate, dedup=args.dedup)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_names = list(le.classes_)
    n_classes = len(label_names)
    feature_names = list(X.columns)

    if ARTIFACT_FEATURE not in feature_names:
        raise SystemExit(f"'{ARTIFACT_FEATURE}' not in features; cannot inject.")

    # Fixed split so every rate sees the same train/test partition.
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)
    cw = class_weight_dict(y_tr)

    from xgboost import XGBClassifier

    # Train ONCE on clean data. The realistic threat is corruption appearing at
    # inference time on a model trained on clean data: the model has learned to
    # trust Flow Duration and is then fed physically-impossible values. (Injecting
    # into training instead teaches the model to distrust the feature, which
    # masks the effect, so we corrupt the test set.)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    sample_w = np.array([cw[c] for c in y_tr])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, objective="multi:softprob", tree_method="hist",
        random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sample_w)

    # Baseline SHAP importance of the feature on clean test data.
    base_idx = _stratified_sample_idx(y_te, min(args.shap_sample, len(X_te)), rng)

    rows = []
    for rate in args.rates:
        # Corrupt the TEST features at this rate (all rows, or one class).
        te_mask = None
        if args.inject_class and args.inject_class in label_names:
            te_mask = (y_te == label_names.index(args.inject_class))
        X_te_inj, n_flip = inject(X_te, ARTIFACT_FEATURE, rate, rng, te_mask)
        X_te_s = scaler.transform(X_te_inj)
        pred = xgb.predict(X_te_s)

        macro_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
        p, r, f, s = precision_recall_fscore_support(
            y_te, pred, labels=range(n_classes), zero_division=0
        )
        per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=label_names)

        g, _ = tree_shap_importance(xgb, X_te_s[base_idx], feature_names, n_classes)
        ranking = list(g.index)
        fd_rank = ranking.index(ARTIFACT_FEATURE) + 1

        # Track the injected class's OWN precision, plus the rare-class mean.
        inj_class_prec = (float(per_class.loc[args.inject_class, "precision"])
                          if args.inject_class in label_names else np.nan)
        rare = [c for c in args.rare_classes if c in label_names]
        rare_prec = float(np.mean([per_class.loc[c, "precision"] for c in rare])) if rare else np.nan

        rows.append({
            "injection_rate": rate,
            "rows_flipped": n_flip,
            "macro_f1": round(float(macro_f1), 4),
            "injected_class_precision": round(inj_class_prec, 4),
            "rare_mean_precision": round(rare_prec, 4),
            "flow_duration_shap_rank": fd_rank,
            "flow_duration_shap_value": round(float(g[ARTIFACT_FEATURE]), 6),
        })
        print(f"rate={rate:<5} flipped={n_flip:<7} macroF1={macro_f1:.4f} "
              f"injClassP={inj_class_prec:.4f} FD_rank={fd_rank}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "injection_sweep.csv"), index=False)
    with open(os.path.join(args.out_dir, "injection_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rates": args.rates, "inject_class": args.inject_class,
                   "note": "SYNTHETIC injection of negative Flow Duration"}, f, indent=2)

    print("\n=== Injection sweep: distortion vs corruption rate ===")
    print(res.to_string(index=False))
    print(f"\nArtifacts written to {args.out_dir}/")
    print("NOTE: report these as controlled synthetic injections, not natural corruption.")


if __name__ == "__main__":
    main()