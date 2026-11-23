INTRO = r"""
# Clinical knowledge-informed neural networks (CKINN) for type 2 diabetes risk: development in RICH, frozen external evaluation in NAGALA, and NHANES screening

This notebook reproduces every analysis of the study from the original data files. It is self-contained: the cells below write the analysis package (`ckinn_obj1/`), verify the inputs by SHA-256, and run the complete pipeline. Code cells contain code only. The text cells explain each step.

## Study design

| Element | Definition |
|---|---|
| Task A (primary) | Recorded incident type 2 diabetes within three years of a baseline health examination |
| Development cohort | Rich Healthcare Group, China (RICH; Chen et al., BMJ Open 2018; Dryad 10.5061/dryad.ft8750v) |
| External cohort | NAGALA, Japan (Okamura et al., Int J Obes 2019; Dryad 10.5061/dryad.8q0p192), scored only with models frozen in RICH |
| Eligibility | Age ≥ 20 years, positive follow-up, binary event, baseline FPG < 6.1 mmol/L (NAGALA additionally HbA1c < 6.5%) |
| Predictors | Age, sex, BMI, systolic blood pressure, ALT, fasting plasma glucose, triglycerides, HDL cholesterol (plus missingness indicators) |
| Censoring | Inverse probability of censoring weights from a reverse Kaplan–Meier estimate; cases 1/G(T−), controls 1/G(3), unknown status 0 |
| Models | M1 unconstrained branched network; M2 = M1 with explanations; M3 = CKINN (M1 plus a soft penalty on wrong-signed probability gradients, FPG+ and BMI+, λ tuned in {0, 1, 10, 100}); M4 site-balanced; M5 sex loss-gap penalty; logistic regression, random forest, XGBoost, LightGBM, linear SVM, ridge Cox, intercept-only reference |
| Primary contrast | Brier(M3) − Brier(M1) in NAGALA with a paired participant bootstrap (censoring weights re-estimated per draw) |
| Secondary contrast | Matched positive versus zero penalty with identical architecture, preprocessing, splits, seeds and epochs |
| Task B (separate) | Elevated HbA1c (≥ 6.5%) at the examination among NHANES participants without a reported diagnosis; trained on 2017–2020, frozen and evaluated on 2021–2023, four non-glycaemic inputs |

Tuning, early stopping, penalty selection and thresholds use RICH development partitions only. NAGALA and the later NHANES period are never used for fitting.

## Proposal coverage (CKINN_OBJ, sections 22–24)

| Proposal item | Where it is answered | Output |
|---|---|---|
| RQ1 Predict future type 2 diabetes from baseline variables | Stages 21–25 | `pooled_oof.csv`, `external_metrics.csv` |
| RQ2 / H1 Clinical knowledge improves prediction over an unconstrained network | Stage 26 (primary contrast), stage 28, EXT E9–E10 | `paired_brier.csv`, `matched_positive_paired_brier.csv`, `ablations.csv` |
| RQ3 / H3 Clinical knowledge improves cross-population generalisation | Stages 24–26 (RICH → NAGALA), stage 29 (held-out RICH sites), EXT E6, E9 | `external_metrics.csv`, `A_RICH_site.csv`, `EXT/*` |
| RQ4 / H2 Calibration is maintained externally and improved by constraints | Stages 25–27, EXT E12 | `calibration_bins.csv`, `calibration_roc_pr.png`, `EXT/E12_calibration_contrast.csv` |
| RQ5 / H4 Explanations are clinically consistent | Stage 30 (integrated gradients, random-forest SHAP, finite-step direction checks, ICE), EXT E2, E8 | `directions.csv`, `attributions.csv`, `rf_shap.csv`, `EXT/E2`, `EXT/E8` |
| RQ6 / H5 Performance is consistent across demographic groups | Stages 28, 31 (sex, age, site, missingness; M1, M3, M4, M5) | `subgroups.csv`, `sex_gaps.csv` |
| RQ7 Elevated risk before diagnostic thresholds | Stage 31 (baseline FPG < 5.6 mmol/L) | `subgroups.csv` |
| Section 24 evaluation framework | AUROC, average precision, sensitivity, specificity, PPV, NPV, F1, FPR, FNR, Brier, calibration intercept and slope, calibration plot, external and cross-population validation, group metrics, attribution and direction checks, net benefit and IPA (EXT) | All tables |
| Section 3A, current status | Task B, stages 33–35 | `B_as_coded/*` |
| Sections 3C, 11, 26 and clinical expert agreement | Not measurable with these data (no repeated dated states, glucose–insulin dynamics, curated knowledge graph or clinician panel) | Listed in `proposal_hypotheses_evidence.csv` |

H1 is the single pre-specified confirmatory contrast. All other comparisons are descriptive. Analyses labelled EXT were specified after the first full development run and are exploratory. Their rules are in `EXT_PRESPECIFICATION.md`.

## How to run

1. Place the original files in one folder (`data/raw` under the project folder): `RCHealthCareData-20180820.xlsx` (RICH), `DatasetIntJObesity.xlsx` (NAGALA) and `Diabetes_Health_Indicators_NHANES_2017-2023.zip` (NHANES extract). File names may differ; the files are identified by their SHA-256.
2. In Colab, choose a CPU runtime and run all cells. Outside Colab, set `CKINN_ROOT` to the project folder.
3. `RUN_MODE`: `SMOKE` checks the workflow on small samples in minutes and its numbers must not be interpreted. `STANDARD` is the reported analysis (5 × 2 nested cross-validation, 6 network trials, 100 epochs with early stopping, 200 bootstrap draws for the v2 contrasts and 1,000 for EXT). `FULL` adds three network seeds, 12 trials and 1,000 bootstrap draws.
4. Every fit is checkpointed. After an interruption, rerun the notebook. To run one part, set `RUN_STAGE` to `AUDIT`, `CHECKS`, `INTERNAL`, `FINAL`, `EXTERNAL`, `ABLATIONS`, `SENSITIVITY`, `EXPLAIN`, `SUBGROUPS`, `UPDATE`, `SCREEN`, `EXT` or `REPORT`.
5. STANDARD needs about 20 CPU hours in total (stage 29 is about half). Results are written to `results/`. Stage 39 writes an aggregate ZIP without participant-level data.

## Integrity

Only the original cohort data are analysed; no synthetic, simulated or augmented records enter any analysis. Small artificial arrays are used only inside the software checks (stages 14, 17 and 20). Every number in the output tables is computed by the code from the data. Null and unfavourable results are reported as found. Participant-level data, models and predictions stay in the private `checkpoints/` folder and are never exported.
"""

SETTINGS_MD = r"""
## 01 · Paths and run settings

The project folder holds `data/raw` (inputs), `checkpoints/` (private models and predictions), `results/` (aggregate tables, figures and reports) and `logs/`. The run identity binds the input bytes, the package code, the configuration, the device and the software versions. Changing any of them starts a new run. Changing `RUN_STAGE` does not.
"""

SETTINGS_CODE = r"""
import os
import sys
import json
from pathlib import Path

try:
    from google.colab import drive

    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/CKINN_OBJ1")
except ImportError:
    ROOT = Path(os.environ.get("CKINN_ROOT", "CKINN_OBJ1")).resolve()

RAW_DIR = ROOT / "data" / "raw"
PACKAGE = Path("ckinn_obj1").resolve()
PACKAGE.mkdir(parents=True, exist_ok=True)
RUN_MODE = os.environ.get("CKINN_RUN_MODE", "STANDARD")
RUN_STAGE = os.environ.get("CKINN_RUN_STAGE", "ALL")
DEVICE = "cpu"
OPTION_OVERRIDES = {"broader_rich": True, "site_holdout": True, "alt_zero": True, "tree_shap": True}
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
print("Project folder:", ROOT)
print("Input folder:", RAW_DIR, "| exists:", RAW_DIR.is_dir())
print("Mode:", RUN_MODE, "| stage:", RUN_STAGE, "| device:", DEVICE)
"""

MODULES = {
    "data.py": r"""
## 02 · Package module `data.py`: inputs, harmonisation and eligibility

Identifies the input files by SHA-256 and reads the workbooks, including both cached values and formula text. It harmonises names, units and sex codes (male = 1) and sets non-positive physical measurements to missing. It applies the eligibility rules step by step and audits the NHANES extract. The three-year status function codes a case (event by year 3), a control (follow-up beyond 3 years) or unknown status. Every predictor passes through a task-specific allowlist.
""",
    "dictionary.py": r"""
## 03 · Package module `dictionary.py`: field dictionary

Builds a field dictionary from the raw-feature audit: type, unit, missingness and range of every source column.
""",
    "stats.py": r"""
## 04 · Package module `stats.py`: fixed-horizon estimands

Reverse Kaplan–Meier censoring survival (detections precede withdrawals at tied times), IPCW weights, the IPCW Brier score over all N (Graf et al., 1999), the weighted cumulative/dynamic AUROC, average precision, calibration intercept (logit offset) and slope, decile calibration, threshold metrics and the paired participant bootstrap with censoring weights re-estimated in every draw.
""",
    "models.py": r"""
## 05 · Package module `models.py`: preprocessing, networks, penalty and comparators

Preprocessing is fitted on training rows only: median imputation, standardisation and one missingness indicator per input. The branched network has demographic, body/pressure and laboratory branches feeding a fusion block that outputs one logit. The clinical penalty averages the squared wrong-signed probability derivative over observed, in-support participant–feature pairs, and is differentiated during training. The module also covers training with early stopping and epoch checkpoints, participant-disjoint stratified splits, IPCW-weighted comparators, the SVM with cross-fitted Platt calibration, and ridge Cox truncated at three years.
""",
    "architecture.py": r"""
## 06 · Package module `architecture.py`: architecture figure

Draws the default eight-input architecture with its two training terms.
""",
    "runner.py": r"""
## 07 · Package module `runner.py`: nested validation, selection, freezing and scoring

`Study` binds the run identity and directories. Its main methods are:

* `internal`: nested cross-validation in RICH, plus the held-out-site, broader-eligibility and ALT-zero variants;
* `plan`: inner-fold tuning, including the matched-penalty and component variants;
* `freeze`: final RICH fits and the freeze manifest;
* `score`: frozen scoring of the external cohort.

Candidate schedules for M1 and M3 share seeds, batch order and stopping rules. M3 additionally searches λ.
""",
    "analysis.py": r"""
## 08 · Package module `analysis.py`: post-freeze analyses

Paired external contrasts, calibration/ROC/PR figures, pooled out-of-fold summaries, subgroups with the 20-case support rule, integrated gradients with completeness checks, finite-step and analytical direction checks, ICE curves, optional random-forest SHAP, optional target intercept updating, the claim-to-evidence register and the aggregate export.
""",
    "stages.py": r"""
## 09 · Package module `stages.py`: stage groups and display helpers

Defines the stage groups used by `RUN_STAGE`, the IPCW support table and small display helpers.
""",
    "verify.py": r"""
## 10 · Package module `verify.py`: software checks

Unit checks on small artificial arrays only (no cohort rows):

* hand-computed weights with ties;
* the full-N Brier denominator;
* penalty gradients and mask exclusion;
* raw-unit finite differences;
* integrated-gradient completeness;
* training-only preprocessing and forbidden predictors;
* unit conversion and eligibility flow;
* ZIP safety;
* the matched zero-penalty identity;
* exact epoch resumption;
* the estimator weight response;
* site weights.

The integration routine runs the whole pipeline on artificial records.
""",
    "extension.py": r"""
## 11 · Package module `extension.py`: extension analyses (EXT, exploratory)

These analyses were specified after the first full development run; their rules are in `EXT_PRESPECIFICATION.md`.

* E2: penalty activity (wrong-way gradient shares and penalty versus data loss).
* E3 to E5: paired Brier differences against every baseline, the index of prediction accuracy, and IPCW net benefit.
* E6: NAGALA restricted to at least two years of follow-up, as in the RICH source design.
* E7: NAGALA with HDL treated as missing.
* E8: six-direction finite-step checks.
* E9, E9b: a penalty-strength path for six clinical directions, with paired contrasts.
* E10: a training-size curve on RICH subsamples.
* E11: RICH without sites lacking three-year support.
* E12: a paired calibration contrast.
* A generated table mapping the proposal's research questions and hypotheses to the computed estimates.
""",
    "verify_extension.py": r"""
## 12 · Package module `verify_extension.py`: checks of the extension

Hand-calculated net benefit and index of prediction accuracy, plan reconstruction, and an end-to-end run of every extension analysis on artificial records (including resumption from checkpoints).
""",
}

CONFIG_MD = r"""
## 13 · Configuration

Profiles (budgets), the search space, the model list, the options and the support rules. The protocol fixes τ = 3 years and seed 1729.
"""

STAGES = [
    (r"""
## 14 · Environment check

Checks the installed versions against the declared ranges and installs only missing packages.
""", r"""
from importlib import metadata
from packaging.requirements import Requirement
import subprocess

REQUIREMENTS = [
    "numpy>=1.26,<3",
    "pandas>=2.1,<3.1",
    "scipy>=1.11,<2",
    "scikit-learn>=1.4,<2",
    "torch>=2.2,<3",
    "lifelines>=0.30,<0.31",
    "lightgbm>=4.1,<5",
    "xgboost>=2,<4",
    "openpyxl>=3.1,<4",
    "matplotlib>=3.7,<4",
    "nbformat>=5.9,<6",
    "shap>=0.45,<1",
]
missing = []
for spec in map(Requirement, REQUIREMENTS):
    try:
        installed = metadata.version(spec.name)
        if installed not in spec.specifier:
            raise RuntimeError(f"{spec.name} {installed} is outside {spec.specifier}")
    except metadata.PackageNotFoundError:
        missing.append(str(spec))
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *missing])
print({spec.name: metadata.version(spec.name) for spec in map(Requirement, REQUIREMENTS)})
"""),
    (r"""
## 15 · Imports, seeds and device

Imports the package written above, fixes the seed and enables deterministic PyTorch. LightGBM 4.6 attaches placeholder column names to NumPy inputs and scikit-learn then warns about missing feature names. The warning is suppressed because predictions are identical with and without names.
""", r"""
import inspect
import warnings

sys.path.insert(0, str(PACKAGE))
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
from IPython.display import display, Image
import numpy as np
import pandas as pd
from data import discover, js, csv, R, J, CORE, NONLAB, PLUS, UNITS, allowed, digest
from stats import weights, metrics
from models import Prep, clinical, Net, seed_all
from runner import Study, VARIANTS, env_info
from analysis import external_contrast, figures, subgroups, explain, updating, evidence, export, internal_summary
from stages import GROUPS, TRAIN_STAGES, SELECTION_COLS, METRIC_COLS, support_table, show, test_cases
from verify import run_checks
import architecture
import extension
import verify_extension

config = json.loads((PACKAGE / "config.json").read_text())
seed_all(config["seed"])
print(json.dumps(env_info(), indent=2))
"""),
    (r"""
## 16 · Run identity, budget and options

Builds the `Study` object, prints the run identifier, budget and options. `active` decides which stages run under `RUN_STAGE`.
""", r"""
config.update(profile=RUN_MODE, stage=RUN_STAGE, device=DEVICE)
if RUN_MODE not in config["profiles"] or (RUN_STAGE not in ("ALL", "EXT") and RUN_STAGE not in GROUPS):
    raise ValueError("Unknown RUN_MODE or RUN_STAGE")
if set(OPTION_OVERRIDES) - set(config["options"]):
    raise ValueError("Unknown option in OPTION_OVERRIDES")
config["options"].update(OPTION_OVERRIDES)
opt = config["options"]
st = Study(ROOT, config, paths=discover(RAW_DIR))


def active(number):
    selected = RUN_STAGE == "ALL" or (RUN_STAGE in GROUPS and number in GROUPS[RUN_STAGE])
    if selected and RUN_MODE == "AUDIT" and number in TRAIN_STAGES:
        print("Skipped in AUDIT mode.")
        return False
    return selected


def active_ext():
    return RUN_STAGE in ("ALL", "EXT") and RUN_MODE != "AUDIT"


print("Run", st.run, "·", RUN_MODE, "·", st.device)
print(json.dumps({"budget": st.b, "options": opt, "search": config["search"]}, indent=2))
"""),
    (r"""
## 17 · Stage 05 · Verify and audit the inputs

Identifies each input by SHA-256, inspects every sheet and ZIP member, and caches the harmonised and eligible tables privately. The first pass takes several minutes.
""", r"""
if active(5):
    print(pd.Series(st.identity["data"]).to_string())
    st.audit()
"""),
    (r"""
## 18 · Stage 06 · Schemas and provenance

Workbook sheets and headers, NHANES archive members, provenance files and the cross-cycle consistency report.
""", r"""
if active(6):
    info = json.loads((st.auditpath / "inventory.json").read_text(encoding="utf8"))
    for sheet in info["workbooks"]:
        print(sheet)
    print(info["zip_members"], info["consistency_report"])
"""),
    (r"""
## 19 · Stage 07 · Completeness, coded values and unit checks

Dimensions, identifiers, duplicate rows, missing and zero values, formula caches and mass-to-molar conversion checks (NAGALA).
""", r"""
if active(7):
    for name in ["structure", "raw_features", "formula_cache", "unit_checks"]:
        show(st.auditpath / (name + ".csv"))
"""),
    (r"""
## 20 · Stage 08 · Harmonised predictor registry

Source headers in both cohorts and the analytical unit of every shared predictor.
""", r"""
if active(8):
    display(pd.DataFrame([dict(input=k, RICH=R[k], NAGALA=J[k], unit=UNITS[k]) for k in PLUS]))
"""),
    (r"""
## 21 · Stage 09 · Eligibility flow

Participants and events after each rule. The FPG < 6.1 mmol/L rule aligns RICH with NAGALA, whose source study excluded higher baseline glucose. It removes 3.4% of RICH participants but more than half of the RICH three-year cases, so it defines the target population as adults with normal fasting glucose. The broader RICH sensitivity analysis keeps 6.1 to < 7.0 mmol/L.
""", r"""
if active(9):
    show(st.auditpath / "cohort_flow.csv")
"""),
    (r"""
## 22 · Stage 10 · Three-year support and censoring

A person with an event after year 3 is a known control. An event-free person last seen at or before year 3 has unknown status. The table reports G(3), weight quantiles, effective sample size and sites without follow-up beyond three years.

Note that RICH participants all have at least two years of follow-up (a source exclusion), so RICH three-year cases are detections in years 2 to 3. EXT E6 evaluates NAGALA under the same condition.
""", r"""
if active(10):
    display(support_table(st))
"""),
    (r"""
## 23 · Stage 11 · Predictor allowlist and development partitions

RICH outer folds are participant-disjoint and stratified by three-year status and site, and are stored before any fit.
""", r"""
if active(11):
    print("Primary inputs", allowed(CORE, "A"), "Screening inputs", allowed(NONLAB, "B"))
    if st.mode != "AUDIT":
        st.split(st.data("RICH"), st.b["outer"], "A_RICH_nested", st.seed)
"""),
    (r"""
## 24 · Stage 12 · Preprocessing learned on training rows

Training medians, missingness indicators, training mean and positive scale, and support limits (1st–99th training percentiles) for the penalty.
""", r"""
if active(12):
    print(inspect.getsource(Prep))
"""),
    (r"""
## 25 · Stage 13 · Censoring weights and known status

A case exactly at year 3 is included. A censoring exactly at year 3 has unknown status.
""", r"""
if active(13):
    print(inspect.getsource(weights))
"""),
    (r"""
## 26 · Stage 14 · Independent checks of the weights and the loss

Seven artificial participants with a tied detection and withdrawal, a detection and a censoring at year 3, and a later detection. The expected non-zero weights are 1, 1.5, 2.25 and 2.25. A separate fixture checks the full-N denominator.
""", r"""
if active(14):
    test_cases(["test_boundaries_ties_hand_weights", "test_loss_denominator_and_penalty"])
"""),
    (r"""
## 27 · Stage 15 · Metrics and calibration

The Brier score is the weighted mean squared error over all eligible participants. Rank measures compare known cases with known controls.
""", r"""
if active(15):
    print(inspect.getsource(metrics))
"""),
    (r"""
## 28 · Stage 16 · Network architecture

Three branches (demographic; body and pressure; laboratory) produce learned features, and a fusion block produces one logit. M1 and M3 share the architecture. M3 adds the clinical penalty during training only.
""", r"""
if active(16):
    architecture.draw(st.fig / "architecture.png")
    display(Image(filename=str(st.fig / "architecture.png")))
"""),
    (r"""
## 29 · Stage 17 · The soft clinical penalty

The penalty uses the local probability derivative with respect to each constrained input. Increasing FPG and BMI have positive priors in task A. Only observed, in-support pairs contribute.
""", r"""
if active(17):
    test_cases(["test_loss_denominator_and_penalty", "test_derivative_sign_masks_and_raw_finite_difference"])
    print(inspect.getsource(clinical))
"""),
    (r"""
## 30 · Stage 18 · Statistical comparators

Logistic regression, random forest, XGBoost, LightGBM and a linear SVM are fitted with IPCW sample weights, with the SVM calibrated on training-only cross-fitted scores. Ridge Cox is administratively truncated at three years. The reference model predicts the weighted training event fraction.
""", r"""
if active(18):
    print(st.cfg["models"], "Comparator candidate budget", st.b["comparator_trials"])
"""),
    (r"""
## 31 · Stage 19 · Tuning, stopping and checkpoints

Each inner fold learns its own preprocessing and censoring weights. Candidates are selected by validation Brier score, and models are refitted on the outer training partition for the median selected epoch count.
""", r"""
if active(19):
    print(st.b, "Checkpoint directory", st.private)
"""),
    (r"""
## 32 · Stage 20 · Software checks

Runs the unit checks of the package and of the extension, on artificial arrays only.
""", r"""
if active(20):
    js(st.report / "software_checks.json", {"package": run_checks(), "extension": verify_extension.run_checks()})
"""),
    (r"""
## 33 · Stage 21 · Nested internal validation in RICH

Within each outer training fold, every model is tuned in inner folds. The held-out outer participants are scored once. A site-specific censoring model is fitted as a sensitivity analysis.
""", r"""
if active(21):
    display(st.internal().head())
"""),
    (r"""
## 34 · Stage 22 · Internal results

Fold-level results and pooled out-of-fold metrics. Folds share training participants and are not independent replicates.
""", r"""
if active(22):
    show(st.table / "A" / "A_RICH_nested.csv")
    display(internal_summary(st))
"""),
    (r"""
## 35 · Stage 23 · Final configurations selected in RICH

The inner procedure is repeated on all eligible RICH participants. The matched pair is anchored on the M3 settings and epoch count; its positive λ (1, 10 or 100) is chosen on the RICH inner folds only.
""", r"""
if active(23):
    display(st.selection_table("A")[SELECTION_COLS])
"""),
    (r"""
## 36 · Stage 24 · Final fits and freeze manifest

The selected configurations are fitted on RICH. Models, preprocessing and hashes enter the freeze manifest before any external scoring.
""", r"""
if active(24):
    print(st.freeze("A"))
"""),
    (r"""
## 37 · Stage 25 · Frozen scoring in NAGALA

The frozen pipelines are applied unchanged to NAGALA.
""", r"""
if active(25):
    display(st.score("A")[METRIC_COLS])
"""),
    (r"""
## 38 · Stage 26 · Paired external contrasts (H1)

Brier(M3) − Brier(M1) and Brier(matched positive) − Brier(matched zero) in the same NAGALA participants. Negative values favour the constrained model. A zero-crossing interval is a valid result.
""", r"""
if active(26):
    display(pd.Series(external_contrast(st)))
    if opt["ablations"]:
        display(pd.Series(external_contrast(st, matched=True)))
"""),
    (r"""
## 39 · Stage 27 · Calibration, ROC and precision–recall in NAGALA

Decile calibration against the IPCW event fraction, and ROC and precision–recall curves, with the threshold chosen on RICH validation predictions.
""", r"""
if active(27):
    figures(st, "A")
    show(st.fig / "A" / "calibration_roc_pr.png")
"""),
    (r"""
## 40 · Stage 28 · M1–M5 and component variants

M2 reuses M1 predictions. M4 adds training site weights, and M5 a sex loss-gap penalty. The FPG-only, BMI-only and expanded-prior variants use the matched positive λ. Also included: no masks, single stream, expanded inputs and the four-input non-laboratory network.
""", r"""
if active(28):
    if opt["ablations"]:
        table = pd.read_csv(st.table / "A" / "external_metrics.csv")
        if not set(VARIANTS).issubset(table.model):
            raise ValueError("Component fits missing from the frozen manifest")
        csv(st.table / "A" / "ablations.csv", table[table.model.isin(VARIANTS + ["M1", "M2", "M3"])])
        display(pd.read_csv(st.table / "A" / "ablations.csv")[METRIC_COLS])
"""),
    (r"""
## 41 · Stage 29 · Sensitivity analyses (H3)

* Leave-one-site-out validation in RICH (sites with fewer than 20 cases or controls are reported as unsupported).
* Broader RICH eligibility (FPG < 7.0 mmol/L).
* ALT zeros retained.

This is the longest stage, and it resumes after interruption.
""", r"""
if active(29):
    for option, source, site in [("site_holdout", "RICH", True), ("broader_rich", "RICH_broad", False), ("alt_zero", "RICH_alt0", False)]:
        if opt[option]:
            display(st.internal(source=source, site=site).head())
        else:
            print("Disabled", option)
    show(st.auditpath / "feature_set_availability.csv")
"""),
    (r"""
## 42 · Stage 30 · Explanations and clinical direction (H4)

* Integrated gradients from training medians, with a completeness check.
* Finite raw-unit steps and analytical gradients for FPG and BMI (all six expanded directions for the expanded-prior model).
* ICE curves.
* Random-forest probability SHAP.

These describe model behaviour, not causal effects.
""", r"""
if active(30):
    if opt["explain"]:
        explain(st, "A")
        show(st.table / "A" / "directions.csv")
        show(st.table / "A" / "attributions.csv")
"""),
    (r"""
## 43 · Stage 31 · Subgroups (H5, RQ7)

Sex, age bands, baseline FPG < 5.6 mmol/L, predictor availability and (in RICH out-of-fold predictions) site, for M1, M3, M4 and M5. Groups need at least 20 known cases and 20 known controls for estimates.
""", r"""
if active(31):
    if opt["subgroups"]:
        display(subgroups(st, "A").head())
"""),
    (r"""
## 44 · Stage 32 · Optional target intercept update

Disabled by default. If enabled, 1,000 NAGALA participants update the intercept and the remaining participants evaluate it, after the frozen results have been reported.
""", r"""
if active(32):
    if opt["target_update"]:
        display(pd.DataFrame(updating(st)))
    else:
        print("Target updating disabled.")
"""),
    (r"""
## 45 · Stage 33 · NHANES screening eligibility (task B)

Adults with measured HbA1c and the extract's no-diagnosis code. A screen is positive at HbA1c ≥ 6.5%. The extract's recoding of borderline and unknown responses is undocumented, so results are labelled as coded.
""", r"""
if active(33):
    show(st.auditpath / "cohort_flow.csv")
    print("NHANES analysis label", st.tag("B"), "| producer recode evidence:", opt["producer_recode_evidence"] or "unavailable")
"""),
    (r"""
## 46 · Stage 34 · Development and freeze in NHANES 2017–2020

Inputs are age, sex, BMI and systolic blood pressure, with a BMI-only positive prior. Glycaemic measurements and self-reported diabetes never enter the models.
""", r"""
if active(34):
    if opt["screening"] != "off":
        print(st.freeze("B"))
        display(st.selection_table("B", frozen=True)[SELECTION_COLS])
"""),
    (r"""
## 47 · Stage 35 · Frozen evaluation in NHANES 2021–2023

Screening metrics, the paired contrast, calibration, subgroups and direction checks in the later period. The extract has no survey design variables, so estimates apply to the extract only.
""", r"""
if active(35):
    if opt["screening"] != "off":
        display(st.score("B")[METRIC_COLS])
        display(pd.Series(external_contrast(st, "B")))
        figures(st, "B")
        if opt["subgroups"]:
            subgroups(st, "B")
        if opt["explain"]:
            explain(st, "B")
"""),
    (r"""
## 48 · EXT · Extension analyses (exploratory)

The rules for E2 to E12 are in `EXT_PRESPECIFICATION.md`. New fits (E9, E10, E11) are checkpointed like all other fits. Outputs are in `results/tables/<mode>/<run>/A/EXT/`.
""", r"""
if active_ext():
    print(extension.run_all(st))
    for name in ["E2_penalty_diagnostics.csv", "E3_paired_brier_vs_baselines.csv", "E9_lambda_path_expanded.csv",
                 "E9b_lambda_path_paired_brier.csv", "E10_training_size_summary.csv", "E12_calibration_contrast.csv"]:
        show(st.table / "A" / "EXT" / name)
"""),
    (r"""
## 49 · Stage 36 · Inventory of tables and fit budget

Lists every aggregate table and summarises the recorded fits and run times.
""", r"""
if active(36):
    for path in sorted(st.table.rglob("*.csv")):
        print(path.relative_to(st.table))
    if (st.log / "fits.jsonl").exists():
        fits = pd.read_json(st.log / "fits.jsonl", lines=True)
        csv(st.table / "fit_budget.csv",
            fits.groupby(["task", "model", "status"], dropna=False).agg(fits=("model", "size"), seconds=("seconds", "sum")).reset_index())
"""),
    (r"""
## 50 · Stage 37 · Proposal questions, hypotheses and the claim register

Maps RQ1–RQ7 and H1–H5 of the proposal to the computed estimates. Only H1 has a decision rule. It also writes the claim-to-evidence register and a narrative generated from the saved tables.
""", r"""
if active(37):
    display(extension.proposal_evidence(st))
    evidence(st)
    show(st.report / "summary.md")
"""),
    (r"""
## 51 · Stage 38 · Reproducibility manifest

Input and code identities, environment, settings, budgets, available tables and frozen-model hashes.
""", r"""
if active(38):
    evidence(st)
    show(st.report / "run_manifest.json")
"""),
    (r"""
## 52 · Stage 39 · Aggregate export

Writes a ZIP of aggregate tables, figures, reports, the package code and logs. Models, participant-level predictions, individual explanations and source data stay in `checkpoints/`.
""", r"""
if active(39):
    print(export(st, PACKAGE))
"""),
]
