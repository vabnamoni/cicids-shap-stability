"""
RQ1 pipeline: multiclass intrusion detection on CIC-IDS2017.
Loads and cleans flow data, trains 1D-CNN, Random Forest, and XGBoost under
cost-sensitive learning, and reports imbalance-aware evaluation metrics.

Designed so RQ2 (SHAP), RQ3 (cross-dataset stability), and RQ4 (artifact
distortion) attach to the artifacts produced here without restructuring.

Run locally:
    python rq1_pipeline.py --data_dir /path/to/cicids2017_csvs --out_dir ./outputs
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

# Time-derived columns where a negative value is physically impossible and
# signals the CICFlowMeter clock-desync artifact. Rows with negatives here are
# dropped. Columns NOT listed here may carry semantically valid negatives
# (e.g. directional or flag encodings) and are preserved.
PHYSICAL_TIME_COLS = [
    "Flow Duration",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Flow IAT Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
]


def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def load_data(data_dir, max_rows=None, downcast=False, seed=SEED, sample_frac=None):
    """Load and concatenate all CSVs in a directory.

    For very large datasets (e.g. CSE-CIC-IDS2018 at ~8M rows/file):
      - sample_frac: keep this fraction of each chunk while streaming
        (e.g. 0.2 keeps 20%), the simplest way to fit a huge file in memory.
      - downcast: store floats as float32 / ints as int32 (about half the RAM).
    Without these, behaviour is unchanged (full load).
    max_rows caps the final row count as a safety net.
    """
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if sample_frac is None and max_rows is None and not downcast:
        frames = [normalize_columns(pd.read_csv(p, low_memory=False)) for p in paths]
        return pd.concat(frames, ignore_index=True)

    rng = np.random.RandomState(seed)
    kept = []
    for path in paths:
        for chunk in pd.read_csv(path, low_memory=False, chunksize=200_000):
            chunk = normalize_columns(chunk)
            if downcast:
                fc = chunk.select_dtypes(include=["float64"]).columns
                chunk[fc] = chunk[fc].astype("float32")
                ic = chunk.select_dtypes(include=["int64"]).columns
                chunk[ic] = chunk[ic].astype("int32")
            if sample_frac is not None and sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=rng.randint(1 << 30))
            kept.append(chunk)
    data = pd.concat(kept, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return data


def resolve_label_column(df):
    for cand in ["Label", "label", " Label"]:
        if cand in df.columns:
            return cand
    raise KeyError("No label column found (expected 'Label').")


# Maps the granular CIC-IDS2017 labels onto consolidated classes used in the
# paper. The keys are matched after fixing the corrupted en-dash and stripping
# whitespace. Set --no_consolidate to keep the original 15 classes instead.
LABEL_CONSOLIDATION = {
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
}


def normalize_labels(y, consolidate=True):
    # Repair the corrupted byte that appears as U+FFFD in the CIC-IDS2017
    # web-attack labels, and normalise dash variants to a plain ASCII hyphen
    # with single surrounding spaces, matching the consolidation keys above.
    y = (
        y.astype(str)
        .str.replace("\ufffd", "-", regex=False)
        .str.replace("\u2013", "-", regex=False)  # en-dash
        .str.replace("\u2014", "-", regex=False)  # em-dash
        .str.replace(r"\s*-\s*", " - ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    if consolidate:
        # Build a lookup whose keys are normalised the same way as y so the
        # hyphen spacing in e.g. "FTP-Patator" matches "FTP - Patator".
        def norm_key(k):
            import re
            k = re.sub(r"\s*-\s*", " - ", k)
            return re.sub(r"\s+", " ", k).strip()
        lookup = {norm_key(k): v for k, v in LABEL_CONSOLIDATION.items()}
        y = y.map(lambda v: lookup.get(v, v))
    return y


def clean(df, label_col, keep_artifact=False, consolidate=True, dedup=False):
    """Clean flow data.

    keep_artifact=True preserves the negative-Flow-Duration rows so the same
    function can produce the corrupted snapshot used in RQ4.
    consolidate=True maps granular labels onto the paper's class set.
    dedup=True removes exact duplicate feature rows (a documented CIC-IDS2017
    issue; ~21% of rows). Deduplicating here, before any split, prevents
    identical rows leaking across train/test and inflating accuracy.
    Returns (X_df, y_series, report_dict).
    """
    report = {}
    n0 = len(df)

    # Separate label, repair/normalise, coerce features to numeric
    y = normalize_labels(df[label_col], consolidate=consolidate)
    X = df.drop(columns=[label_col])

    # Drop obviously non-feature columns if present
    drop_candidates = [
        "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
        "Source Port", "Src Port", "Destination Port", "Dst Port",
        "Timestamp", "Protocol",
    ]
    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    # Inf -> NaN, then drop NaN rows
    X = X.replace([np.inf, -np.inf], np.nan)
    nan_mask = X.isna().any(axis=1)
    report["rows_dropped_nan"] = int(nan_mask.sum())
    X, y = X[~nan_mask], y[~nan_mask]

    # Drop zero-variance features. Use per-column nunique (a constant column has
    # exactly one unique value) rather than X.var() over the whole matrix, which
    # allocates a large float64 intermediate and OOMs on multi-million-row data.
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    report["zero_variance_features_dropped"] = zero_var
    X = X.drop(columns=zero_var)

    # Artifact handling: drop rows with physically impossible negative times
    present_time_cols = [c for c in PHYSICAL_TIME_COLS if c in X.columns]
    if not keep_artifact and present_time_cols:
        neg_mask = (X[present_time_cols] < 0).any(axis=1)
        report["rows_dropped_negative_time_artifact"] = int(neg_mask.sum())
        X, y = X[~neg_mask], y[~neg_mask]
    else:
        report["rows_dropped_negative_time_artifact"] = 0

    # Deduplicate exact feature rows (before any split) to prevent leakage.
    if dedup:
        before = len(X)
        dup_mask = X.duplicated(keep="first")
        X, y = X[~dup_mask], y[~dup_mask]
        report["rows_dropped_duplicates"] = int(before - len(X))
    else:
        report["rows_dropped_duplicates"] = 0

    report["rows_in"] = n0
    report["rows_out"] = len(X)
    report["n_features"] = X.shape[1]
    return X.reset_index(drop=True), y.reset_index(drop=True), report


def stratified_splits(X, y, seed=SEED):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def class_weight_dict(y_encoded):
    classes = np.unique(y_encoded)
    weights = compute_class_weight("balanced", classes=classes, y=y_encoded)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def capped_class_weight_dict(y_encoded, cap=50.0):
    """Class weights with an upper cap.

    Full inverse-frequency weighting gives the rarest classes enormous weights
    (e.g. ~250000x for an 11-sample class in 2.8M rows), which produces huge,
    unstable gradients for a neural net. Trees tolerate this; CNNs do not.
    Capping keeps the minority emphasis without destabilising training.
    """
    base = class_weight_dict(y_encoded)
    return {c: min(w, cap) for c, w in base.items()}


def build_cnn(n_features, n_classes):
    # Imported here so the script still loads for users without TF installed,
    # as long as they only run the tree models.
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(SEED)
    model = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(name, y_true, y_pred, label_names, out_dir):
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    acc = (y_true == y_pred).mean()

    summary = {
        "model": name,
        "accuracy": float(acc),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
    }

    report_txt = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(out_dir, f"{name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
    np.savetxt(os.path.join(out_dir, f"{name}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    print(f"\n=== {name} ===")
    print(f"accuracy={acc:.4f}  macro_f1={macro[2]:.4f}  weighted_f1={weighted[2]:.4f}")
    print(report_txt)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory of CIC-IDS2017 CSVs")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--skip_cnn", action="store_true", help="Skip the 1D-CNN (no TensorFlow)")
    ap.add_argument("--cnn_epochs", type=int, default=30)
    ap.add_argument("--cnn_weight_cap", type=float, default=15.0,
                    help="Cap on CNN class weights. Lower (5-15) reduces rare-class "
                         "over-prediction and raises precision/macro-F1; higher "
                         "raises recall at the cost of precision.")
    ap.add_argument("--keep_artifact", action="store_true",
                    help="Keep negative-Flow-Duration rows (corrupted snapshot for RQ4)")
    ap.add_argument("--no_consolidate", action="store_true",
                    help="Keep original 15 labels instead of consolidating to paper classes")
    ap.add_argument("--dedup", action="store_true",
                    help="Remove exact duplicate rows before splitting (prevents "
                         "train/test leakage; recommended for CIC-IDS2017).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    raw = load_data(args.data_dir)
    label_col = resolve_label_column(raw)
    print(f"Loaded {len(raw):,} rows, {raw.shape[1]} columns")

    print("Cleaning...")
    X, y, clean_report = clean(
        raw, label_col, keep_artifact=args.keep_artifact,
        consolidate=not args.no_consolidate, dedup=args.dedup,
    )
    print(json.dumps(clean_report, indent=2))
    with open(os.path.join(args.out_dir, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_names = list(le.classes_)
    n_classes = len(label_names)
    print(f"\nClasses ({n_classes}): {label_names}")
    print("Class distribution:")
    print(pd.Series(y).value_counts().to_string())

    # Split (stratified)
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_splits(X, y_enc)

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    feature_names = list(X.columns)
    with open(os.path.join(args.out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    cw = class_weight_dict(y_tr)
    summaries = []

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    from xgboost import XGBClassifier
    sample_w = np.array([cw[c] for c in y_tr])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
        tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
    )
    xgb.fit(X_tr_s, y_tr, sample_weight=sample_w)
    summaries.append(evaluate("xgboost", y_te, xgb.predict(X_te_s), label_names, args.out_dir))

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)
    summaries.append(evaluate("random_forest", y_te, rf.predict(X_te_s), label_names, args.out_dir))

    # --- 1D-CNN ---
    if not args.skip_cnn:
        print("\nTraining 1D-CNN...")
        import tensorflow as tf
        cnn = build_cnn(X_tr_s.shape[1], n_classes)
        X_tr_c = X_tr_s[..., np.newaxis]
        X_val_c = X_val_s[..., np.newaxis]
        X_te_c = X_te_s[..., np.newaxis]
        cnn_cw = capped_class_weight_dict(y_tr, cap=args.cnn_weight_cap)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
        cnn.fit(
            X_tr_c, y_tr, validation_data=(X_val_c, y_val),
            epochs=args.cnn_epochs, batch_size=512, class_weight=cnn_cw,
            callbacks=[es], verbose=2,
        )
        cnn_pred = cnn.predict(X_te_c, verbose=0).argmax(axis=1)
        summaries.append(evaluate("cnn", y_te, cnn_pred, label_names, args.out_dir))

    # --- Comparison table ---
    comp = pd.DataFrame(summaries).set_index("model")
    comp = comp[["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
    comp.to_csv(os.path.join(args.out_dir, "model_comparison.csv"))
    print("\n=== Model comparison (RQ1) ===")
    print(comp.round(4).to_string())
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()