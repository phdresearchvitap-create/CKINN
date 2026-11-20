# CKINN versus M1 and M2 on real data, version 2

Re-analysis of the version 1 real-data study (CKINN against the plain networks M1 and M2 on NAGALA, Rich Healthcare Group and NHANES) after an audit. Version 1 and its results are kept as the record; every change is an amendment in `protocol_v2.md`.

## Contents

| File | What it is |
| --- | --- |
| `CKINN_vs_MLP_real_data_final.ipynb` | Clean notebook (no outputs). Self-contained: it writes the `ckv2` library to `ckv2_lib/` and runs everything. |
| `CKINN_vs_MLP_real_data_final_EXECUTED.ipynb` | The same notebook executed in full mode. |
| `AUDIT.md`, `audit/` | Findings on version 1 with evidence; `audit/audit_checks.py` prints every number quoted. |
| `protocol_v2.md` | Frozen protocol with amendment log, labels (P1, P2, E, PH) and the configuration hash. |
| `CHANGELOG.md` | What changed from version 1, in engineering terms. |
| `references.md` | References and their verification status. |
| `outputs_v2_full/` | Tables (CSV), figures (PDF and 300 dpi PNG), LaTeX tables, `report/report.md`, `report/methods_results_draft.md`, `report/TRIPOD_AI_checklist.md`, manifest and hashes, and the archive `ckinn_v2_outputs_full.zip`. No row-level data. |
| `ckv2/`, `tests/` | Library and unit tests (the notebook carries identical copies). |
| `notebook_cells.py`, `build_notebook.py` | Source of the notebook; `python build_notebook.py` rebuilds it. |

## Run on Google Colab

1. Put the three files in one Drive folder, for example `MyDrive/CKINN_Real/data/`: `DatasetIntJObesity.xlsx`, `RCHealthCareData-20180820.xlsx`, `nhanes_t2d_extract.csv` (names may carry prefixes; the sha256 of each is checked in cell 17).
2. Upload `CKINN_vs_MLP_real_data_final.ipynb` to Colab. In a first cell set:
   ```python
   import os
   os.environ["CKINN_REAL_MODE"] = "smoke"          # then "full"
   os.environ["CKINN_DATA_DIR"] = "/content/drive/MyDrive/CKINN_Real/data"
   os.environ["CKINN_OUT_DIR"] = "/content/drive/MyDrive/CKINN_Real/outputs_v2_smoke"   # outputs_v2_full for full mode
   ```
3. Run all cells. Colab already has PyTorch, XGBoost, LightGBM, scikit-learn and pytest; lifelines and scikit-survival are optional (two unit tests use them as references and are skipped if absent).
4. If the runtime disconnects, run all cells again: every model fit is checkpointed in `<out>/cache/tasks/` and is not repeated.

## Run locally

```bash
pip install torch numpy pandas scipy scikit-learn xgboost lightgbm matplotlib openpyxl pytest nbclient ipykernel
export CKINN_REAL_MODE=full CKINN_DATA_DIR=/path/to/data CKINN_OUT_DIR=$PWD/outputs_v2_full CKINN_WORKERS=4
jupyter nbconvert --to notebook --execute CKINN_vs_MLP_real_data_final.ipynb --output CKINN_vs_MLP_real_data_final_EXECUTED.ipynb --ExecutePreprocessor.timeout=-1
python -m pytest -q tests        # library tests on toy arrays only
```

## Modes and runtime

`smoke` runs the whole pipeline on stratified subsamples with tiny settings (about 4 minutes on four cores); its numbers test the code and must not be interpreted. `full` is the reported analysis; the full-mode configuration must match the hash in `protocol_v2.md`, otherwise cell 16 stops. Measured wall time of the full run on four CPU cores is in `outputs_v2_full/tables/table35_runtime.csv` and `report/manifest.json`. On a two-core Colab machine expect roughly twice as long. Each model fit runs single-threaded in its own worker process (`CKINN_WORKERS`, default: all cores), which also makes the results independent of the number of cores.

## Integrity rules built into the code

- Every number in the tables, figures, `report.md` and `methods_results_draft.md` is computed from stored predictions; cell 41 fails if a decimal number in the report or the draft does not match a stored table or the configuration, or if a markdown cell contains a result-like number.
- Development rows are used only for tuning; evaluation rows only by cross-validation; thresholds come from inner validation splits.
- Row-level data and predictions stay in `<out>/cache/` and are never archived or committed.
- Outputs are research risk estimates, not diagnoses.
