# xai-nids-shap

Explainable AI for network intrusion detection: a leakage-controlled study of
SHAP explanation **trustworthiness** across model families, across datasets, and
under data-quality corruption, on CIC-IDS2017 and CSE-CIC-IDS2018.

This repository contains the complete, reproducible pipeline for the paper
*"A Hybrid Machine Learning Framework with SHAP-Based Interpretability for
Network Intrusion Detection: Cross-Model Agreement, Cross-Dataset Stability, and
Sensitivity to Data-Quality Corruption."*

> **Note on anonymity:** if the associated paper is under double-blind review,
> keep author-identifying information out of this repository (commit history,
> README, and file headers) until acceptance.

---

## What this project does

The pipeline trains three intrusion-detection models with distinct inductive
biases — a 1D-CNN, a Random Forest, and XGBoost — on CIC-IDS2017 and interprets
all of them with SHAP. It then answers four research questions:

- **RQ1** — How do the three models compare on multiclass detection, especially
  on rare classes?
- **RQ2** — Do SHAP feature-importance rankings agree across model families?
- **RQ3** — Are those rankings stable when transferred to an independent dataset
  (CSE-CIC-IDS2018)?
- **RQ4** — How does controlled corruption of a timing feature affect detection
  and its SHAP importance?

A key methodological step is **deduplication before splitting**: CIC-IDS2017
contains ~21% exact duplicate flows, and a naive split leaks ~23% of test rows
into training, inflating apparent performance. All results are reported on
deduplicated, leakage-controlled data.

---

## Key findings

- Tree ensembles substantially outperform the 1D-CNN on rare-class detection
  (macro-F1 0.94 and 0.91 vs 0.54).
- SHAP rankings are only **moderately** consistent across model families
  (Spearman rho ~ 0.5) but **highly** stable across datasets for a fixed model
  (rho ~ 0.97): model choice affects the explanation more than dataset choice.
- Controlled corruption of Flow Duration raises its SHAP importance monotonically
  while detection degrades — explanations inherit data-quality vulnerabilities.

---

## Repository structure

```
xai-nids-shap/
├── README.md
├── requirements.txt
├── .gitignore
├── rq1_pipeline.py          # RQ1: load, clean, dedup, train CNN/RF/XGBoost, evaluate
├── rq2_shap.py              # RQ2: SHAP global/per-class importance + cross-model agreement
├── rq3_cross_dataset.py     # RQ3: cross-dataset SHAP stability (2017 -> 2018)
├── rq4_artifact.py          # RQ4: natural artifact check (corrupted vs cleaned)
├── rq4_injection.py         # RQ4: controlled injection dose-response
├── check_data_validity.py   # duplicate / leakage / feature-leakage audit
├── tune_cnn_cap.py          # CNN class-weight-cap selection sweep
├── make_figures.py          # generates the six publication figures
└── data/                    # (not committed) place CIC CSVs here
    ├── cicids2017/
    └── cicids2018/
```

`rq1_pipeline.py` is the core module; the other scripts import their shared
preprocessing (`clean`, `load_data`, `stratified_splits`, etc.) from it, so all
experiments use an identical, single-source cleaning pipeline.

---

## Datasets

The datasets are **not** included in this repository (they are large and
licensed). Download them from the Canadian Institute for Cybersecurity:

- **CIC-IDS2017** — https://www.unb.ca/cic/datasets/ids-2017.html
- **CSE-CIC-IDS2018** — https://www.unb.ca/cic/datasets/ids-2018.html

Place the CSV files under `data/cicids2017/` and `data/cicids2018/` respectively
(or point the scripts elsewhere with `--data_dir`).

---

## Installation

Requires Python 3.10+. On Windows, GPU TensorFlow is unavailable natively
(TF >= 2.11); the CNN runs on CPU, or use WSL2 for GPU.

```bash
python -m venv env
# Windows:  env\Scripts\activate
# Linux/Mac: source env/bin/activate
pip install -r requirements.txt
```

**Version note (important):** SHAP is sensitive to the XGBoost version. The
verified working pair is `shap==0.49.1` with `xgboost==2.0.3`, or `shap>=0.52`
with `xgboost>=3.x`. A mismatch causes a TreeExplainer parse error on multiclass
models. See `requirements.txt`.

---

## Usage

Run the experiments in order. Paths below assume the CSVs are in `data/`.

**1. Data-validity audit (run this first):**
```bash
python check_data_validity.py --data_dir data/cicids2017 --out_dir outputs/validity
```
Reports duplicate fraction, train/test leakage, per-class split, and any
single-feature label leakage.

**2. RQ1 — detection (with deduplication):**
```bash
python rq1_pipeline.py --data_dir data/cicids2017 --out_dir outputs/dedup \
    --cnn_epochs 30 --cnn_weight_cap 25 --dedup
```
Add `--skip_cnn` for a fast tree-only run.

**3. (optional) Select the CNN weight cap:**
```bash
python tune_cnn_cap.py --data_dir data/cicids2017 --caps 5 10 15 25 --subsample 300000
```

**4. RQ2 — SHAP agreement:**
```bash
python rq2_shap.py --data_dir data/cicids2017 --out_dir outputs/dedup/rq2 \
    --rf_max_depth 25 --shap_sample 5000 --cnn_epochs 30 --cnn_weight_cap 25 --dedup
```

**5. RQ3 — cross-dataset stability** (2018 is large; sample + downcast for memory):
```bash
python rq3_cross_dataset.py --train_dir data/cicids2017 --test_dir data/cicids2018 \
    --out_dir outputs/dedup/rq3 --skip_cnn --shap_sample 5000 \
    --test_sample_frac 0.10 --downcast --dedup
```

**6. RQ4 — corruption sensitivity:**
```bash
# natural-artifact check
python rq4_artifact.py --data_dir data/cicids2017 --out_dir outputs/rq4 --dedup
# controlled injection dose-response (all rows)
python rq4_injection.py --data_dir data/cicids2017 --out_dir outputs/rq4_inj_all \
    --rates 0.0 0.1 0.25 0.5 0.75 1.0
```

**7. Figures:**
```bash
python make_figures.py
```
Writes six publication figures (PNG 300 dpi + PDF) to `outputs/figures/`.

---

## Reproducibility

- All stochastic operations use a fixed seed (`SEED = 42`).
- Every transformation (scaler, resampling) is fitted on the training partition
  only; deduplication happens before splitting.
- Cleaning rules distinguish artifact-induced negative values (dropped) from
  semantically valid negatives (preserved).
- Record your exact `shap` / `xgboost` / `tensorflow` versions when reporting
  results, as SHAP output is version-sensitive.

---

## Common issues

- **`TreeExplainer` ValueError on XGBoost** — shap/xgboost version mismatch; see
  the version note above.
- **`MemoryError` loading CSE-CIC-IDS2018** — use `--test_sample_frac 0.10
  --downcast`; the file is ~8M rows.
- **CNN training unstable / macro-F1 near zero** — lower `--cnn_weight_cap`; the
  cap prevents ultra-rare classes from destabilising gradients.
- **`UnicodeEncodeError` on Windows** — handled in-script via UTF-8 writes; the
  CIC-IDS2017 web-attack labels contain a non-ASCII dash that the pipeline
  repairs automatically.

---

## Citation

If you use this code, please cite the associated paper:

```
[Citation to be added on acceptance / de-anonymisation.]
```

---

## License

[Choose a license — MIT is a common, permissive default for research code.
Add a LICENSE file. If you want others to reuse freely, MIT; if you want
attribution and share-alike, consider Apache-2.0 or GPL-3.0.]