# CKINN Objective 1: code and run guide for reviewers

## Contents

| File | Purpose |
| --- | --- |
| `CKINN_Objective1_Final.ipynb` | Self-contained notebook (Google Colab or Jupyter). It writes the analysis package `ckinn_obj1/`, verifies the inputs and runs every analysis. Code cells hold code only; text cells explain each step. |
| `CKINN_Objective1_Final_cells.txt` | The same notebook as plain text, cell by cell. |
| `src/` | The analysis package as files (identical to the `%%writefile` cells) and `config.json`. |
| `docs/CKINN_Objective1_Protocol.md` | Operational analysis protocol, written before the full development run (unchanged). |
| `docs/EXT_PRESPECIFICATION.md` | Rules of the exploratory extension analyses, written and hashed before each was computed (`.sha256` holds the hashes and UTC times). |
| `CHANGES.md` | Differences from the code version that produced the first full development run. |
| `build_final.py`, `notebook_text.py` | Regenerate the notebook from the source package. |

## Data

The inputs are identified by SHA-256, not by file name.

| Input | Source | SHA-256 |
| --- | --- | --- |
| RICH workbook | Chen et al., BMJ Open 2018;8:e021768; Dryad 10.5061/dryad.ft8750v | `ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6` |
| NAGALA workbook | Okamura et al., Int J Obes 2019;43:139–48; Dryad 10.5061/dryad.8q0p192 | `671359aea2019c848f969395f88e57f888eaba8463ea0351bcda8f1c947f5523` |
| NHANES extract (ZIP with two period CSVs, provenance JSON files and a consistency report) | Derived from public CDC NHANES 2017–March 2020 and August 2021–August 2023 files | `3af63d0d53d8f88c38a7f54d80e6e448014322f7c8c3fe115ba9b3e47c2c6acf` |

Put the three files in `<project>/data/raw/`. In Colab the project folder is `MyDrive/CKINN_OBJ1`. Elsewhere, set the environment variable `CKINN_ROOT`.

## Running

1. Open the notebook. In Colab, select a CPU runtime and run all cells.
2. `RUN_MODE` (cell 01 or environment variable `CKINN_RUN_MODE`) chooses the profile:
   * `SMOKE` is a workflow check on small samples; its numbers are not results.
   * `STANDARD` is the reported analysis.
   * `FULL` adds three network seeds and larger searches.
3. All fits, splits, plans and predictions are checkpointed. After an interruption, run the notebook again with the same settings. `RUN_STAGE` (or `CKINN_RUN_STAGE`) can restrict a session to one stage group.
4. Compute: STANDARD needs roughly 20 CPU hours. The leave-one-site-out sensitivity analysis (stage 29) takes about half of that.

Outputs are written to `results/tables/<mode>/<run>/`, `results/figures/...` and `results/reports/...`. Stage 39 writes `CKINN_<mode>_aggregate.zip`, which contains no participant-level data. Models, predictions and individual explanations stay in `checkpoints/`.

## Analysis status

* **Pre-specified confirmatory contrast (H1):** Brier(M3) − Brier(M1) in NAGALA, with a paired bootstrap interval.
* **Pre-specified but descriptive:** every other analysis in stages 21 to 35.
* **Exploratory (EXT):** analyses specified after the first full development run, with rules fixed before each was computed.
* **Generated map:** stage 37 maps every research question and hypothesis of the proposal to the computed estimates (`A/EXT/proposal_hypotheses_evidence.csv`). It lists what these data cannot address: progression through prediabetes, physiological dynamics, clinician agreement, a knowledge graph and clinical benefit.

## Software environment of the development run

The development run used Python 3.13 with numpy 2.4.6, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1, torch 2.14.1 (CPU), lifelines 0.30.3, LightGBM 4.6.0, XGBoost 3.4.1, openpyxl 3.1.5, matplotlib 3.10.0 and shap 0.54.1. Each run records its own versions in `results/reports/<mode>/<run>/run_manifest.json`.

Neural networks, logistic regression, gradient boosting and Cox models reproduced the development run to < 10⁻⁶ in Brier score and AUROC under a different environment. Random forest and SVM differed slightly between scikit-learn 1.6 and 1.9.

## Integrity

* Only the original cohort data are analysed.
* Artificial arrays appear only inside the software checks (stages 14, 17 and 20).
* No result is entered by hand. Null and unfavourable results are reported as computed.
