# Radar_ml

**Radar_ml** — simplified demonstration of a radar-based object detection & localization pipeline.

> This repository contains a privacy‑preserving, runnable skeleton of a model that classifies **object type**, predicts **object number**, and estimates **distance (R)** and **angle** at each location using aggregated radar burst features. The original training dataset and proprietary project details are confidential and **not** included here. The code shows the **data aggregation**, **feature selection**, **model training**, **evaluation**, and **a prediction wrapper** using XGBoost.

---

## Key points (high level)

- Input: per-burst radar features in a CSV (`sample_data.csv` format used). Bursts are grouped into `location_id` buckets.
- Aggregation: each location's bursts are aggregated (mean, std, min, max, median) per feature to build one row per location.
- Targets: `Object Type` (classification), `Object Number` (classification), `R` (regression — distance), `Angle` (classification).
- Models: XGBoost classifiers/regressors. Separate feature selection for each target (via XGBoost feature importance).
- Output: trained models + encoders saved to `radar_xgb_models.pkl`. A helper `predict_on_new_location()` reads a new CSV and prints predictions for a single location.

---

## Files in this repo

- `main.py` — full training + evaluation + save pipeline (the script you provided).
- `sample_data.csv` — example CSV with the column layout expected by the script (***this file is a placeholder / synthetic*** in the public repo; original dataset is confidential).
- `test.csv` — sample file used by the prediction helper (also placeholder/synthetic).
- `radar_xgb_models.pkl` — generated after running the script (not included by default).

---

## How to run (local)

1. Create a Python environment (recommended: `venv` or `conda`) with Python 3.8+.

2. Install dependencies (recommended to pin versions in `requirements.txt`):

```bash
pip install pandas numpy scikit-learn xgboost
```

3. Ensure `sample_data.csv` (or your non-confidential synthetic data) is present in the working directory and matches the expected column layout. Then run:

```bash
python main.py
```

This will:
- read `sample_data.csv`,
- aggregate bursts per location,
- select top features per target,
- train XGBoost models,
- evaluate on a holdout split (prints accuracy / RMSE / classification reports),
- save models and encoders to `radar_xgb_models.pkl`,
- call `predict_on_new_location('test.csv')` and print a prediction for the provided test file.

---

## Expected console output (conceptually)

The script prints progress messages such as:

```
Selecting best features with XGBoost...
Top features for Object Type: [ ... ]
Top features for Object Number: [ ... ]
Top features for R: [ ... ]
Top features for Angle: [ ... ]
Training XGBoost models...
--- VALIDATION RESULTS (Holdout set) ---
Object Type accuracy: X.XX
Confusion matrix:
...
R (distance) RMSE: X.XX
Models and encoders saved as radar_xgb_models.pkl
--- Prediction for this location ---
Object Type: <label>
Object Number: <label>
R: <float>
Angle: <label>
```

Actual numeric results depend on the data used for training and evaluation.

---

## Data format & expectations

**Input CSV columns (required)** — the script expects columns at least similar to the following (example names):
- `burstid` — per-burst identifier (optional but dropped in aggregation)
- per-burst numeric features: e.g. `feat1`, `feat2`, `feat3`, ... (any number of numeric columns)
- targets (per-burst): `Object Type`, `Object Number`, `R`, `Angle` — the script takes the first value per location after grouping

Important notes:
- The code groups every `BURSTS_PER_LOCATION` rows into one `location_id` by using `df.index // BURSTS_PER_LOCATION`.
- Aggregation produces columns like `feat1_mean`, `feat1_std`, `feat1_min`, `feat1_max`, `feat1_median`.
- The `predict_on_new_location()` helper assumes the test CSV has the same per-burst feature columns and performs the same aggregation (it sets `location_id = 0` internally to build a single aggregated example).

---

## Confidentiality-safe recommendations (keeps your IP safe)

- **Do not** include raw/confidential radar captures in the public repo. Keep original training data private.
- Provide a **synthetic sample** (randomized numeric features with the same column names) so others can run the code end‑to‑end without exposing sensitive data (current `sample_data.csv` can be such a placeholder).
- In the README and comments, avoid disclosing device IDs / sensor-specific parameters / any project code names.
- If you want to showcase performance, show **aggregated metrics** (e.g., overall accuracy, mean RMSE) on anonymized or synthetic validation sets rather than raw confidential results.

---

## Pitfalls & suggestions (improvements you might want to add)

- Add `requirements.txt` for reproducibility; pin `xgboost` and `scikit-learn` versions.
- Add small utility scripts:
  - `make_synthetic_data.py` — to generate a synthetic `sample_data.csv` and `test.csv` with the expected schema.
  - `evaluate_saved_models.py` — to load `radar_xgb_models.pkl` and compute metrics on a held-out (synthetic) set.
- Move heavy logic into modules (`data.py`, `features.py`, `train.py`, `predict.py`) to improve readability and testability.
- Add CLI flags (via `argparse`) to control `CSV_PATH`, `BURSTS_PER_LOCATION`, `TOP_FEATURES`, and the output model path.
- Persist model training metrics (JSON summary) and example plots (confusion matrix, predicted vs true R scatter) to a `reports/` folder.

---

## Example `requirements.txt` (suggested)

```
pandas>=1.2
numpy>=1.19
scikit-learn>=1.0
xgboost>=1.6
```

---

## License

Add a license file if you intend to open source this repo. A permissive option is the MIT license. If this project stays proprietary, remove the license file and add a `NOTICE.md` describing usage constraints.

---

If you want, I can:
- generate a polished `README.md` file and commit it for you, or
- create a small `make_synthetic_data.py` script and add a `requirements.txt` and `run.sh` to make the repo runnable without exposing confidential data.

Tell me which of those you'd like and I will create the files.