"""
RQ2: SHAP-based explainability and cross-model agreement on CIC-IDS2017.

Reuses the RQ1 preprocessing unchanged (imported), retrains the three models,
then computes:
  - global SHAP importance per model (TreeSHAP for RF/XGBoost, DeepSHAP for CNN)
  - per-class SHAP importance
  - cross-model rank agreement (Spearman, Kendall, top-k overlap)

Outputs feed paper Table 5 and the SHAP summary figures.

Run locally (after RQ1 works):
    python rq2_shap.py --data_dir ..\\data\\cicids2017 --out_dir ..\\outputs\\rq2
    # add --skip_cnn to run tree models only (RQ2 agreement then covers 2 models)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reuse RQ1 components verbatim so preprocessing cannot diverge from the paper.
from rq1_pipeline import (
    SEED,
    build_cnn,
    class_weight_dict,
    clean,
    load_data,
    resolve_label_column,
    stratified_splits,
)

# SHAP can be heavy on memory; we subsample the data used for explanation.
SHAP_SAMPLE = 5000          # instances explained for global/per-class importance
CNN_BACKGROUND = 200        # background set size for DeepSHAP
TOP_K = 15                  # top-k features for agreement metrics


def tree_shap_importance(model, X_sample, feature_names, n_classes):
    """Return (global_importance Series, per_class_importance DataFrame).

    Uses fast TreeSHAP. If the installed shap is too old to parse a modern
    multiclass XGBoost model (it stores base_score as a per-class vector that
    older shap cannot read), raises a clear, actionable error rather than
    silently producing wrong numbers.
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
    except ValueError as e:
        if "could not convert string to float" in str(e) and _is_xgb(model):
            raise RuntimeError(
                "shap cannot parse this XGBoost model: your shap and xgboost "
                "versions are incompatible (modern XGBoost stores a per-class "
                "base_score vector). Fix by aligning versions, e.g.\n"
                "    pip install \"xgboost==2.0.3\"\n"
                "(keeps your current shap), or\n"
                "    pip install -U shap\n"
                "(keeps your current xgboost). Verified working pair: "
                "shap 0.49.1 + xgboost 2.0.3, or shap >= 0.52 + xgboost 3.x."
            ) from e
        raise

    per_class = _to_per_class_list(sv, n_classes, X_sample.shape[1])
    global_imp = np.mean([np.abs(c).mean(axis=0) for c in per_class], axis=0)
    global_series = pd.Series(global_imp, index=feature_names).sort_values(ascending=False)
    per_class_df = pd.DataFrame(
        {f"class_{i}": np.abs(per_class[i]).mean(axis=0) for i in range(len(per_class))},
        index=feature_names,
    )
    return global_series, per_class_df


def _is_xgb(model):
    return type(model).__module__.startswith("xgboost")


def deep_shap_importance(model, X_background, X_sample, feature_names, n_classes):
    import shap

    bg = X_background[..., np.newaxis]
    xs = X_sample[..., np.newaxis]
    explainer = shap.DeepExplainer(model, bg)
    sv = explainer.shap_values(xs)

    per_class = _to_per_class_list(sv, n_classes, X_sample.shape[1], squeeze_last=True)

    global_imp = np.mean([np.abs(c).mean(axis=0) for c in per_class], axis=0)
    global_series = pd.Series(global_imp, index=feature_names).sort_values(ascending=False)
    per_class_df = pd.DataFrame(
        {f"class_{i}": np.abs(per_class[i]).mean(axis=0) for i in range(len(per_class))},
        index=feature_names,
    )
    return global_series, per_class_df


def _to_per_class_list(sv, n_classes, n_features, squeeze_last=False):
    """Coerce assorted SHAP return formats into a list of (n_samples, n_features).

    Handles: list-per-class (older TreeSHAP), stacked (n_samples, n_features,
    n_classes) (newer TreeSHAP), and CNN/DeepSHAP outputs that carry a trailing
    channel axis, e.g. (n_samples, n_features, 1, n_classes) or
    (n_samples, n_features, 1).
    """
    if isinstance(sv, list):
        arrs = sv
    else:
        sv = np.asarray(sv)
        # Drop any singleton channel axis (the CNN's last input dim of size 1).
        # Squeeze only axes that are size 1 and are NOT the sample axis (0) and
        # NOT a class axis equal to n_classes.
        while sv.ndim > 3:
            squeezed = False
            for ax in range(1, sv.ndim):
                if sv.shape[ax] == 1:
                    sv = np.squeeze(sv, axis=ax)
                    squeezed = True
                    break
            if not squeezed:
                break
        if sv.ndim == 3:
            # Decide which trailing axis is the class axis.
            if sv.shape[-1] == n_classes:
                arrs = [sv[:, :, i] for i in range(sv.shape[-1])]
            elif sv.shape[1] == n_classes:
                arrs = [sv[:, i, :] for i in range(sv.shape[1])]
            else:
                arrs = [sv[:, :, i] for i in range(sv.shape[-1])]
        else:
            arrs = [sv]

    cleaned = []
    for a in arrs:
        a = np.asarray(a)
        # Remove any remaining singleton axes beyond (n_samples, n_features).
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
            if a.shape[1] != n_features and a.shape[1] % n_features == 0:
                a = a[:, :n_features]
        cleaned.append(a)
    return cleaned


def agreement(rankings, names, top_k=TOP_K):
    """Pairwise Spearman, Kendall, and top-k overlap between feature rankings.

    rankings: dict model_name -> importance Series (index = feature names).
    """
    models = list(rankings.keys())
    feats = list(next(iter(rankings.values())).index)  # common feature universe
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            ra = rankings[a].reindex(feats)
            rb = rankings[b].reindex(feats)
            rho, _ = spearmanr(ra.values, rb.values)
            tau, _ = kendalltau(ra.values, rb.values)
            top_a = set(rankings[a].head(top_k).index)
            top_b = set(rankings[b].head(top_k).index)
            overlap = len(top_a & top_b) / top_k
            rows.append({
                "pair": f"{a} vs {b}",
                "spearman_rho": round(float(rho), 4),
                "kendall_tau": round(float(tau), 4),
                f"top{top_k}_overlap": round(overlap, 4),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs/rq2")
    ap.add_argument("--skip_cnn", action="store_true")
    ap.add_argument("--cnn_epochs", type=int, default=30)
    ap.add_argument("--cnn_weight_cap", type=float, default=50.0,
                    help="Cap on CNN class weights; prevents huge gradients from "
                         "ultra-rare classes destabilising training.")
    ap.add_argument("--no_consolidate", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="Remove exact duplicate rows before splitting.")
    ap.add_argument("--shap_sample", type=int, default=SHAP_SAMPLE)
    ap.add_argument("--rf_max_depth", type=int, default=None,
                    help="Cap RF depth to speed up TreeSHAP (None matches RQ1). "
                         "Try 20-30 for a large speedup with near-identical rankings.")
    ap.add_argument("--rf_estimators", type=int, default=300,
                    help="RF tree count for SHAP (fewer = faster TreeSHAP).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading and cleaning (reusing RQ1 pipeline)...")
    raw = load_data(args.data_dir)
    label_col = resolve_label_column(raw)
    X, y, _ = clean(raw, label_col, consolidate=not args.no_consolidate, dedup=args.dedup)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_names = list(le.classes_)
    n_classes = len(label_names)
    feature_names = list(X.columns)

    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)
    cw = class_weight_dict(y_tr)

    # Subsample the test set for SHAP (stratified by label to keep rare classes).
    rng = np.random.RandomState(SEED)
    n_take = min(args.shap_sample, len(X_te_s))
    idx = _stratified_sample_idx(y_te, n_take, rng)
    X_shap = X_te_s[idx]
    print(f"Explaining {len(idx)} instances; {len(feature_names)} features; {n_classes} classes.")

    rankings = {}
    per_class_store = {}

    # XGBoost
    print("Training + explaining XGBoost...")
    from xgboost import XGBClassifier
    sample_w = np.array([cw[c] for c in y_tr])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, objective="multi:softprob", tree_method="hist",
        random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sample_w)
    g, pc = tree_shap_importance(xgb, X_shap, feature_names, n_classes)
    rankings["xgboost"], per_class_store["xgboost"] = g, pc

    # Random Forest
    print("Training + explaining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=args.rf_estimators, max_depth=args.rf_max_depth,
        max_features="sqrt", class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)
    g, pc = tree_shap_importance(rf, X_shap, feature_names, n_classes)
    rankings["random_forest"], per_class_store["random_forest"] = g, pc

    # 1D-CNN
    if not args.skip_cnn:
        print("Training + explaining 1D-CNN (DeepSHAP)...")
        import tensorflow as tf
        from rq1_pipeline import capped_class_weight_dict
        cnn = build_cnn(X_tr_s.shape[1], n_classes)
        cnn_cw = capped_class_weight_dict(y_tr, cap=args.cnn_weight_cap)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                              restore_best_weights=True)
        cnn.fit(X_tr_s[..., np.newaxis], y_tr,
                validation_data=(X_val_s[..., np.newaxis], y_val),
                epochs=args.cnn_epochs, batch_size=512, class_weight=cnn_cw,
                callbacks=[es], verbose=2)
        # Report CNN test accuracy so its training health is on the record; a
        # CNN that failed to learn must not be used for the SHAP comparison.
        cnn_pred = cnn.predict(X_te_s[..., np.newaxis], verbose=0).argmax(axis=1)
        cnn_acc = float((cnn_pred == y_te).mean())
        cnn_macro_f1 = f1_score(y_te, cnn_pred, average="macro")
        print(f"CNN test accuracy={cnn_acc:.4f}  macro_f1={cnn_macro_f1:.4f}")
        if cnn_acc < 0.90:
            print("WARNING: CNN accuracy is low; it may not have converged. "
                  "The CNN SHAP comparison below may be unreliable. Consider "
                  "raising --cnn_epochs or adjusting --cnn_weight_cap.")
        bg_idx = rng.choice(len(X_tr_s), size=min(CNN_BACKGROUND, len(X_tr_s)), replace=False)
        g, pc = deep_shap_importance(cnn, X_tr_s[bg_idx], X_shap, feature_names, n_classes)
        rankings["cnn"], per_class_store["cnn"] = g, pc

    # Save global rankings
    global_df = pd.DataFrame({m: r for m, r in rankings.items()})
    global_df.to_csv(os.path.join(args.out_dir, "global_shap_importance.csv"))

    # Save per-class with readable class names
    for m, pc in per_class_store.items():
        pc.columns = [f"{label_names[int(c.split('_')[1])]}" for c in pc.columns]
        pc.to_csv(os.path.join(args.out_dir, f"per_class_shap_{m}.csv"))

    # Cross-model agreement (RQ2 / Table 5)
    if len(rankings) >= 2:
        agr = agreement(rankings, feature_names)
        agr.to_csv(os.path.join(args.out_dir, "cross_model_agreement.csv"), index=False)
        print("\n=== Cross-model SHAP agreement (RQ2, Table 5) ===")
        print(agr.to_string(index=False))

    # Top features summary
    print("\n=== Top 15 features by mean SHAP (per model) ===")
    print(global_df.head(15).round(5).to_string())
    with open(os.path.join(args.out_dir, "top_features.json"), "w", encoding="utf-8") as f:
        json.dump({m: list(r.head(TOP_K).index) for m, r in rankings.items()}, f, indent=2)

    print(f"\nArtifacts written to {args.out_dir}/")


def _stratified_sample_idx(y, n_take, rng):
    """Indices for an approximately stratified subsample, keeping rare classes."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    idx_all = []
    per_class_quota = max(1, n_take // len(classes))
    for c in classes:
        c_idx = np.where(y == c)[0]
        take = min(per_class_quota, len(c_idx))
        idx_all.extend(rng.choice(c_idx, size=take, replace=False))
    idx_all = np.array(idx_all)
    if len(idx_all) > n_take:
        idx_all = rng.choice(idx_all, size=n_take, replace=False)
    return idx_all


if __name__ == "__main__":
    main()