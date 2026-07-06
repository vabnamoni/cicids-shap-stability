"""
RQ4: Artifact-induced explanation distortion.

Runs the full classify-then-explain pipeline twice on CIC-IDS2017:
  - CORRUPTED: negative Flow Duration rows (CICFlowMeter clock-desync artifact)
    left in place.
  - CLEANED: those rows removed by the same rule used elsewhere.

Reports, for each condition:
  - rare-class precision/recall (does the artifact degrade detection?)
  - the SHAP global-importance rank of Flow Duration (does the artifact distort
    the explanation?)

The claim under test: a silent data-quality artifact corrupts the analyst-facing
explanation as well as the prediction.

Run locally (XGBoost is the primary model here; --with_rf adds Random Forest):
    python rq4_artifact.py --data_dir ..\\data\\cicids2017 --out_dir ..\\outputs\\rq4
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rq1_pipeline import (
    PHYSICAL_TIME_COLS, SEED, class_weight_dict, clean, load_data,
    resolve_label_column, stratified_splits,
)
from rq2_shap import _stratified_sample_idx, tree_shap_importance

# Classes most likely to be affected by the artifact (rare + timing-sensitive).
DEFAULT_RARE = ["Web Attack", "Infiltration", "Heartbleed", "Bot"]
ARTIFACT_FEATURE = "Flow Duration"


def run_condition(raw, label_col, keep_artifact, consolidate, shap_sample, rng, dedup=False):
    """Train XGBoost and compute per-class metrics + SHAP ranking for one condition."""
    X, y, rep = clean(raw, label_col, keep_artifact=keep_artifact,
                      consolidate=consolidate, dedup=dedup)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_names = list(le.classes_)
    n_classes = len(label_names)
    feature_names = list(X.columns)

    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    cw = class_weight_dict(y_tr)

    from xgboost import XGBClassifier
    sample_w = np.array([cw[c] for c in y_tr])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, objective="multi:softprob", tree_method="hist",
        random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sample_w)
    pred = xgb.predict(X_te_s)

    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        y_te, pred, labels=range(n_classes), zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=label_names
    )

    # SHAP global importance and the rank of the artifact feature
    idx = _stratified_sample_idx(y_te, min(shap_sample, len(X_te_s)), rng)
    g, _ = tree_shap_importance(xgb, X_te_s[idx], feature_names, n_classes)
    ranking = list(g.index)
    artifact_rank = ranking.index(ARTIFACT_FEATURE) + 1 if ARTIFACT_FEATURE in ranking else None

    return {
        "per_class": per_class,
        "global_shap": g,
        "artifact_rank": artifact_rank,
        "n_features": len(feature_names),
        "clean_report": rep,
        "label_names": label_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs/rq4")
    ap.add_argument("--no_consolidate", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="Remove exact duplicate rows before splitting.")
    ap.add_argument("--shap_sample", type=int, default=5000)
    ap.add_argument("--rare_classes", nargs="+", default=DEFAULT_RARE)
    ap.add_argument("--sample_frac", type=float, default=None,
                    help="Keep this fraction of rows while streaming (for huge "
                         "datasets like 2018). Artifact counts scale accordingly.")
    ap.add_argument("--downcast", action="store_true",
                    help="Load as float32/int32 to roughly halve memory use.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    consolidate = not args.no_consolidate
    rng = np.random.RandomState(SEED)

    print("Loading data once...")
    raw = load_data(args.data_dir, downcast=args.downcast, sample_frac=args.sample_frac)
    label_col = resolve_label_column(raw)
    if args.sample_frac:
        print(f"  NOTE: loaded a {args.sample_frac:.0%} sample "
              f"({len(raw):,} rows); artifact counts are proportional.")

    print("\n[1/2] CORRUPTED condition (artifact left in)...")
    corrupted = run_condition(raw, label_col, True, consolidate, args.shap_sample, rng, dedup=args.dedup)
    print(f"  artifact rows kept; Flow Duration SHAP rank = {corrupted['artifact_rank']}")

    print("\n[2/2] CLEANED condition (artifact removed)...")
    cleaned = run_condition(raw, label_col, False, consolidate, args.shap_sample, rng, dedup=args.dedup)
    print(f"  artifact rows dropped: "
          f"{cleaned['clean_report']['rows_dropped_negative_time_artifact']}; "
          f"Flow Duration SHAP rank = {cleaned['artifact_rank']}")

    # --- Detection change on rare classes (Table 7) ---
    rare = [c for c in args.rare_classes if c in corrupted["label_names"]]
    det_rows = []
    for c in rare:
        pc_corr = corrupted["per_class"].loc[c]
        pc_clean = cleaned["per_class"].loc[c]
        det_rows.append({
            "class": c,
            "precision_corrupted": round(float(pc_corr["precision"]), 4),
            "precision_cleaned": round(float(pc_clean["precision"]), 4),
            "delta_precision": round(float(pc_clean["precision"] - pc_corr["precision"]), 4),
            "recall_corrupted": round(float(pc_corr["recall"]), 4),
            "recall_cleaned": round(float(pc_clean["recall"]), 4),
        })
    det = pd.DataFrame(det_rows)
    det.to_csv(os.path.join(args.out_dir, "rare_class_detection.csv"), index=False)

    # --- SHAP ranking shift (Table 8) ---
    shift = pd.DataFrame([{
        "feature": ARTIFACT_FEATURE,
        "rank_corrupted": corrupted["artifact_rank"],
        "rank_cleaned": cleaned["artifact_rank"],
        "rank_change": (None if corrupted["artifact_rank"] is None
                        or cleaned["artifact_rank"] is None
                        else corrupted["artifact_rank"] - cleaned["artifact_rank"]),
    }])
    shift.to_csv(os.path.join(args.out_dir, "artifact_shap_rank_shift.csv"), index=False)

    # Also save the two full rankings so shifts in correlated features are visible.
    pd.DataFrame({
        "corrupted": corrupted["global_shap"],
        "cleaned": cleaned["global_shap"].reindex(corrupted["global_shap"].index),
    }).to_csv(os.path.join(args.out_dir, "global_shap_both_conditions.csv"))

    with open(os.path.join(args.out_dir, "rq4_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "artifact_rows_dropped": cleaned["clean_report"]["rows_dropped_negative_time_artifact"],
            "flow_duration_rank_corrupted": corrupted["artifact_rank"],
            "flow_duration_rank_cleaned": cleaned["artifact_rank"],
            "physical_time_cols": PHYSICAL_TIME_COLS,
        }, f, indent=2)

    print("\n=== Rare-class detection: corrupted vs cleaned (Table 7) ===")
    print(det.to_string(index=False))
    print("\n=== Flow Duration SHAP rank shift (Table 8) ===")
    print(shift.to_string(index=False))
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()