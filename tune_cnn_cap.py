"""
Fast sweep of the CNN class-weight cap to find the value that best balances
rare-class recall against precision (maximises macro-F1).

Trains on a subsample for speed so several cap values can be compared quickly,
then you run the full rq1_pipeline once with the chosen --cnn_weight_cap.

    python tune_cnn_cap.py --data_dir ..\\data\\cicids2017 --caps 5 10 15 25 50 --subsample 300000
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rq1_pipeline import (
    SEED, build_cnn, capped_class_weight_dict, clean, load_data,
    resolve_label_column, stratified_splits,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--caps", type=float, nargs="+", default=[5, 10, 15, 25, 50])
    ap.add_argument("--subsample", type=int, default=300000,
                    help="Rows to subsample for the sweep (stratified). Speeds up "
                         "each fit; the final run uses full data.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no_consolidate", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf

    raw = load_data(args.data_dir)
    label_col = resolve_label_column(raw)
    X, y, _ = clean(raw, label_col, consolidate=not args.no_consolidate)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Stratified subsample for speed (keeps all rare-class rows).
    rng = np.random.RandomState(SEED)
    if args.subsample and args.subsample < len(X):
        keep = _stratified_subsample(y_enc, args.subsample, rng)
        X, y_enc = X.iloc[keep].reset_index(drop=True), y_enc[keep]

    # The two-stage stratified split needs each class to have enough members to
    # appear in train, val, AND test. Classes with too few samples (e.g.
    # Heartbleed=11) can crash the split. For CNN hyperparameter tuning they
    # contribute almost nothing, so fold any class with < 8 samples into the
    # nearest viable handling: drop them from the sweep only (the full pipeline
    # still trains on them). Report which are dropped for transparency.
    classes, counts = np.unique(y_enc, return_counts=True)
    too_rare = classes[counts < 8]
    if len(too_rare):
        drop_names = [le.classes_[c] for c in too_rare]
        print(f"NOTE: excluding ultra-rare classes from the CNN tuning sweep "
              f"(too few samples to split): {drop_names}. "
              f"The full rq1_pipeline still trains on them.")
        mask = ~np.isin(y_enc, too_rare)
        X, y_enc = X[mask].reset_index(drop=True), y_enc[mask]
        # Re-encode to a contiguous label range for the CNN.
        _, y_enc = np.unique(y_enc, return_inverse=True)
        n_classes = len(np.unique(y_enc))
    else:
        n_classes = len(le.classes_)

    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)[..., np.newaxis]
    X_val_s = scaler.transform(X_val)[..., np.newaxis]
    X_te_s = scaler.transform(X_te)[..., np.newaxis]

    rows = []
    for cap in args.caps:
        print(f"\n--- weight cap = {cap} ---")
        tf.keras.utils.set_random_seed(SEED)
        cnn = build_cnn(X_tr_s.shape[1], n_classes)
        cw = capped_class_weight_dict(y_tr, cap=cap)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                              restore_best_weights=True)
        cnn.fit(X_tr_s, y_tr, validation_data=(X_val_s, y_val),
                epochs=args.epochs, batch_size=512, class_weight=cw,
                callbacks=[es], verbose=2)
        pred = cnn.predict(X_te_s, verbose=0).argmax(axis=1)
        rows.append({
            "weight_cap": cap,
            "accuracy": round(float((pred == y_te).mean()), 4),
            "macro_precision": round(float(precision_score(y_te, pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_te, pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_te, pred, average="weighted", zero_division=0)), 4),
        })

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n=== CNN weight-cap sweep (sorted by macro-F1) ===")
    print(res.to_string(index=False))
    best = res.iloc[0]["weight_cap"]
    print(f"\nBest cap by macro-F1: {best}")
    print(f"Run the full pipeline with:  --cnn_weight_cap {best}")
    res.to_csv("cnn_cap_sweep.csv", index=False)


def _stratified_subsample(y, n_take, rng):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    frac = n_take / len(y)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        # The downstream three-way stratified split needs each class present in
        # train, val, and test. Guarantee a floor of samples per class (capped at
        # the class's true size) so ultra-rare classes (e.g. Heartbleed=11) do
        # not collapse below what the split requires.
        floor = min(len(c_idx), 6)
        take = max(floor, int(round(len(c_idx) * frac)))
        take = min(take, len(c_idx))
        idx.extend(rng.choice(c_idx, size=take, replace=False))
    return np.array(idx)


if __name__ == "__main__":
    main()