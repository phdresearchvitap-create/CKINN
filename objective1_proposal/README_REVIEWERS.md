# CKINN Objective 1: code and run guide for reviewers (protocol v4)

## Contents

| File | Purpose |
| --- | --- |
| `CKINN_Objective1_Proposal.ipynb` | Self-contained notebook (Google Colab or Jupyter). It writes the analysis package `ckinn_obj1/`, verifies the inputs and runs every analysis. Code cells hold code only; text cells explain each step and the proposal section it implements. |
| `CKINN_Objective1_Proposal_cells.txt` | The same notebook as plain text, cell by cell. |
| `src/` | The analysis package as files (identical to the `%%writefile` cells) and `config.json`. |
| `docs/PROTOCOL_v4_proposal_aligned.md` | Analysis protocol, committed and hashed before any v4 result (`PROTOCOL_v4.sha256`). |
| `build_notebook.py`, `notebook_text.py` | Regenerate the notebook from `src/`. |

## Data

| Input | Source | SHA-256 |
| --- | --- | --- |
| RICH workbook | Chen et al., BMJ Open 2018;8:e021768; Dryad 10.5061/dryad.ft8750v | `ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6` |
| NAGALA workbook | Okamura et al., Int J Obes 2019;43:139–48; Dryad 10.5061/dryad.8q0p192 | `671359aea2019c848f969395f88e57f888eaba8463ea0351bcda8f1c947f5523` |
| NHANES extract (ZIP) | Derived from public CDC NHANES 2017–March 2020 and August 2021–August 2023 files | `3af63d0d53d8f88c38a7f54d80e6e448014322f7c8c3fe115ba9b3e47c2c6acf` |

Put the three files in `<project>/data/raw/`. In Colab the project folder is `MyDrive/CKINN_OBJ1`. Elsewhere, set the environment variable `CKINN_ROOT`.

## Running

* **Profiles** (`RUN_MODE`):
  * `SMOKE` checks the workflow on 6,000-row samples in about 2 minutes; its numbers are not results.
  * `STANDARD` is the reported analysis.
  * `FULL` uses larger searches and five seeds.
* **Resuming:** all fits, splits, plans, predictions and the expensive reports are checkpointed. After an interruption, run the notebook again with the same settings.
* **Stages:** `RUN_STAGE` restricts a session to one stage group, in the order `AUDIT`, `CHECKS`, `INTERNAL`, `FINAL`, `EXTERNAL`, `REVERSE`, `LOSO`, `SENSITIVITY`, `FAIRNESS`, `EXPLAIN`, `SCREEN`, `EXPLORATORY`, `REPORT`.
* **Outputs:** tables go to `results/tables/<mode>/<run>/{A,R,B_as_coded}/v4/` and figures to `results/figures/<mode>/<run>/{A,R}/v4/` (PNG at 300 dpi and PDF). The aggregate ZIP is written to `results/reports/<mode>/<run>/` and contains no participant-level data.

## Analysis status

* **Pre-specified confirmatory contrast (H1):** Brier(CKINN) − Brier(M1) in NAGALA after freezing in RICH, with a paired bootstrap interval.
* **Descriptive:** all other hypotheses (H2 to H5, RQ7).
* **Exploratory:** the training-size curve and the sensitivity analyses.
* **History:** protocol v4 restores the clinical sign table of the research proposal after two earlier runs, in which a two-factor penalty was inactive. The protocol describes this history; the paper must report it.

## Integrity

* Only the original cohort data are analysed.
* Artificial arrays appear only inside the software checks.
* No result is entered by hand. Null and unfavourable results are reported as computed.
* Outputs are research risk estimates, not diagnoses or medical advice.
