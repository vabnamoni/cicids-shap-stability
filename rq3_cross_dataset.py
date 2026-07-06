"""
RQ3: Cross-dataset SHAP stability (transfer framing).

Models are trained on CIC-IDS2017 and then explained on the independent
CSE-CIC-IDS2018 dataset using only the features common to both schemas. The
SHAP feature rankings from 2017 and 2018 are compared with rank correlation to
test whether intrusion-detection explanations generalise across capture
environments.

Reuses RQ1 preprocessing and RQ2 SHAP routines unchanged (imported).

Run locally:
    python rq3_cross_dataset.py ^
        --train_dir ..\\data\\cicids2017 ^
        --test_dir  ..\\data\\cicids2018 ^
        --out_dir   ..\\outputs\\rq3
    # add --skip_cnn for tree models only
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rq1_pipeline import (
    SEED,
    build_cnn,
    class_weight_dict,
    clean,
    load_data,
    normalize_labels,
    resolve_label_column,
    stratified_splits,
)
from rq2_shap import (
    CNN_BACKGROUND,
    TOP_K,
    _stratified_sample_idx,
    deep_shap_importance,
    tree_shap_importance,
)

SHAP_SAMPLE = 5000


def normalize_feature_name(name):
    """Normalise a CIC feature name so 2017 and 2018 schemas align.

    2018 systematically abbreviates 2017's names (Cnt/Len/Tot/Pkt/Byts/Var/Seg
    Size Avg, and underscores vs spaces). We first apply token-level expansion
    rules that cover those patterns wholesale, then a few explicit special cases
    that the rules don't capture.
    """
    n = str(name).strip().lower()
    n = n.replace("_", " ").replace(".", " ")
    n = " ".join(n.split())

    # Normalise "/s" spacing early so "pkts/s" and "packets s" collapse
    # consistently before token expansion.
    n = n.replace("/", " ")
    n = " ".join(n.split())

    # Token-level abbreviation expansion (order-independent word replacement).
    token_map = {
        "cnt": "count",
        "len": "length",
        "tot": "total",
        "pkts": "packet",
        "pkt": "packet",
        "byts": "bytes",
        "var": "variance",
        "seg": "segment",
        "forward": "fwd",
        "backward": "bwd",
        "packets": "packet",
    }
    tokens = [token_map.get(t, t) for t in n.split()]
    n = " ".join(tokens)

    # Word-order-insensitive canonicalisation for stat-qualifier features where
    # 2017 leads with the qualifier (Avg/Max/Min) and 2018 trails with it, e.g.
    # "avg bwd segment size" vs "bwd segment size avg", "max packet length" vs
    # "packet length max". Move a trailing/leading qualifier to a canonical slot.
    QUALIFIERS = {"avg", "average", "max", "min", "mean", "std"}
    parts = n.split()
    quals = [p for p in parts if p in QUALIFIERS]
    rest = [p for p in parts if p not in QUALIFIERS]
    # Canonical form: sorted qualifiers first, then the sorted remaining tokens.
    # Sorting the rest makes "fwd segment size" == "segment size fwd" etc.
    qn = "avg" if any(q in ("avg", "average") for q in quals) else (quals[0] if quals else "")
    canon = " ".join(([qn] if qn else []) + sorted(rest))

    # Explicit whole-string special cases where the above isn't enough.
    specials = {
        "dst port": "destination port",
        "src port": "source port",
        "dst ip": "destination ip",
        "src ip": "source ip",
        "init bytes fwd win": "init win bytes forward",
        "init bytes bwd win": "init win bytes backward",
    }
    return specials.get(canon, canon)


def harmonize(X_train_df, test_raw, test_label_col, consolidate, dedup=False):
    """Return (X_test_aligned_df, y_test, common_features, report).

    Aligns the 2018 feature columns to the 2017 feature set used in training,
    by normalised name, keeping only the intersection.
    """
    report = {}

    # Clean 2018 with the same rules (artifact handling, NaN/Inf, zero-var).
    Xt, yt, clean_rep = clean(test_raw, test_label_col, consolidate=consolidate, dedup=dedup)
    report["test_clean"] = clean_rep

    # Map normalised names -> actual column names on each side.
    train_norm = {normalize_feature_name(c): c for c in X_train_df.columns}
    test_norm = {normalize_feature_name(c): c for c in Xt.columns}

    common_norm = [n for n in train_norm if n in test_norm]
    report["n_train_features"] = len(train_norm)
    report["n_test_features"] = len(test_norm)
    report["n_common_features"] = len(common_norm)
    report["dropped_train_only"] = sorted(
        train_norm[n] for n in train_norm if n not in test_norm
    )
    report["dropped_test_only"] = sorted(
        test_norm[n] for n in test_norm if n not in train_norm
    )

    # Build aligned test matrix in the TRAIN column order (restricted to common).
    train_cols_common = [train_norm[n] for n in common_norm]
    test_cols_common = [test_norm[n] for n in common_norm]
    X_test_aligned = Xt[test_cols_common].copy()
    X_test_aligned.columns = train_cols_common  # rename to train names
    return X_test_aligned, yt, train_cols_common, report


def compare_rankings(imp_2017, imp_2018, top_k=TOP_K):
    feats = list(imp_2017.index)
    a = imp_2017.reindex(feats).values
    b = imp_2018.reindex(feats).values
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    top_a = set(imp_2017.head(top_k).index)
    top_b = set(imp_2018.head(top_k).index)
    overlap = len(top_a & top_b) / top_k
    return {
        "spearman_rho": round(float(rho), 4),
        "kendall_tau": round(float(tau), 4),
        f"top{top_k}_overlap": round(overlap, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="CIC-IDS2017 CSV directory")
    ap.add_argument("--test_dir", required=True, help="CSE-CIC-IDS2018 CSV directory")
    ap.add_argument("--out_dir", default="./outputs/rq3")
    ap.add_argument("--skip_cnn", action="store_true")
    ap.add_argument("--cnn_epochs", type=int, default=30)
    ap.add_argument("--no_consolidate", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="Remove exact duplicate rows before splitting.")
    ap.add_argument("--shap_sample", type=int, default=SHAP_SAMPLE)
    ap.add_argument("--test_sample_frac", type=float, default=None,
                    help="Keep this fraction of the 2018 test rows while streaming "
                         "(for the large ~8M-row file). e.g. 0.10 for 10%.")
    ap.add_argument("--downcast", action="store_true",
                    help="Load as float32/int32 to roughly halve memory use.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    consolidate = not args.no_consolidate

    # --- Train side (2017) ---
    print("Loading + cleaning 2017 (train)...")
    raw_tr = load_data(args.train_dir, downcast=args.downcast)
    tr_label = resolve_label_column(raw_tr)
    X, y, _ = clean(raw_tr, tr_label, consolidate=consolidate, dedup=args.dedup)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)

    # --- Test side (2018), harmonised to the 2017 feature set ---
    print("Loading + harmonising 2018 (test)...")
    raw_te2018 = load_data(args.test_dir, downcast=args.downcast,
                           sample_frac=args.test_sample_frac)
    if args.test_sample_frac:
        print(f"  NOTE: loaded a {args.test_sample_frac:.0%} sample of 2018 "
              f"({len(raw_te2018):,} rows).")
    te_label = resolve_label_column(raw_te2018)
    X18, y18, common_cols, harm_report = harmonize(X, raw_te2018, te_label, consolidate, dedup=args.dedup)
    print(json.dumps({k: v for k, v in harm_report.items()
                      if k != "test_clean"}, indent=2))
    with open(os.path.join(args.out_dir, "harmonization_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(harm_report, f, indent=2)

    # Restrict BOTH sides to the common features, in the same order.
    X_tr_common = X_tr[common_cols]
    X_te2017_common = X_te[common_cols]

    # Scaler fit on 2017 train (common features), applied to 2017-test and 2018.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_common)
    X_2017_s = scaler.transform(X_te2017_common)
    X_2018_s = scaler.transform(X18[common_cols])
    cw = class_weight_dict(y_tr)

    rng = np.random.RandomState(SEED)
    idx17 = _stratified_sample_idx(y_te, min(args.shap_sample, len(X_2017_s)), rng)
    idx18 = _stratified_sample_idx(
        LabelEncoder().fit_transform(y18), min(args.shap_sample, len(X_2018_s)), rng
    )
    X17_shap = X_2017_s[idx17]
    X18_shap = X_2018_s[idx18]
    feats = common_cols
    print(f"Common features: {len(feats)}. Explaining "
          f"{len(idx17)} (2017) and {len(idx18)} (2018) instances.")

    stability_rows = []

    def record(model_name, imp17, imp18):
        imp17.to_frame("importance").to_csv(
            os.path.join(args.out_dir, f"shap_2017_{model_name}.csv"))
        imp18.to_frame("importance").to_csv(
            os.path.join(args.out_dir, f"shap_2018_{model_name}.csv"))
        res = compare_rankings(imp17, imp18)
        res["model"] = model_name
        stability_rows.append(res)
        print(f"  {model_name}: spearman={res['spearman_rho']} "
              f"kendall={res['kendall_tau']} top{TOP_K}={res[f'top{TOP_K}_overlap']}")

    # XGBoost
    print("Training XGBoost on 2017, explaining on both...")
    from xgboost import XGBClassifier
    sample_w = np.array([cw[c] for c in y_tr])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, objective="multi:softprob", tree_method="hist",
        random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sample_w)
    g17, _ = tree_shap_importance(xgb, X17_shap, feats, n_classes)
    g18, _ = tree_shap_importance(xgb, X18_shap, feats, n_classes)
    record("xgboost", g17, g18)

    # Random Forest
    print("Training Random Forest on 2017, explaining on both...")
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)
    g17, _ = tree_shap_importance(rf, X17_shap, feats, n_classes)
    g18, _ = tree_shap_importance(rf, X18_shap, feats, n_classes)
    record("random_forest", g17, g18)

    # 1D-CNN
    if not args.skip_cnn:
        print("Training 1D-CNN on 2017, explaining on both (DeepSHAP)...")
        import tensorflow as tf
        cnn = build_cnn(X_tr_s.shape[1], n_classes)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                              restore_best_weights=True)
        X_val_common = scaler.transform(X_val[common_cols])
        cnn.fit(X_tr_s[..., np.newaxis], y_tr,
                validation_data=(X_val_common[..., np.newaxis], y_val),
                epochs=args.cnn_epochs, batch_size=512, class_weight=cw,
                callbacks=[es], verbose=2)
        bg_idx = rng.choice(len(X_tr_s), size=min(CNN_BACKGROUND, len(X_tr_s)),
                            replace=False)
        g17, _ = deep_shap_importance(cnn, X_tr_s[bg_idx], X17_shap, feats, n_classes)
        g18, _ = deep_shap_importance(cnn, X_tr_s[bg_idx], X18_shap, feats, n_classes)
        record("cnn", g17, g18)

    # --- Stability table (RQ3 / Table 6) ---
    stab = pd.DataFrame(stability_rows).set_index("model")
    stab.to_csv(os.path.join(args.out_dir, "cross_dataset_stability.csv"))
    print("\n=== Cross-dataset SHAP stability (RQ3, Table 6) ===")
    print(stab.to_string())
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()