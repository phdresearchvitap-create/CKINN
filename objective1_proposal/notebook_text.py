INTRO = r"""
# Clinical knowledge-informed neural networks (CKINN) for explainable, fair and cross-population type 2 diabetes risk prediction

Central question (research proposal, section 28): can clinical knowledge constraints improve the accuracy, calibration, fairness, explainability and cross-population generalisation of neural networks for early type 2 diabetes risk prediction?

This notebook reproduces every analysis from the original data files. The cells below write the analysis package (`ckinn_obj1/`), verify the inputs by SHA-256 and run the full pipeline. Code cells contain code only. The text cells explain each step and the proposal section it implements.

## Design

| Element | Implementation |
|---|---|
| Prediction target (§3B) | Recorded incident type 2 diabetes within three years of a baseline examination |
| Populations | RICH, China (15 health-check sites; Chen et al., BMJ Open 2018) and NAGALA, Japan (Okamura et al., Int J Obes 2019) |
| Current-status task (§3A) | NHANES: HbA1c ≥ 6.5% among participants without a reported diagnosis, 2017–2020 to 2021–2023 |
| Inputs (§5) | Age, sex, BMI, systolic blood pressure, ALT, fasting glucose, triglycerides, HDL cholesterol, with missingness indicators |
| Clinical knowledge layer (§6) | Age ↑, BMI ↑, systolic blood pressure ↑ (hypertension), fasting glucose ↑, triglycerides ↑, HDL ↓ |
| Architecture (§7) | Demographic, body/pressure and laboratory branches, fusion network, risk prediction head |
| Knowledge loss (§8–10) | IPCW binary cross-entropy + λ Σⱼ mean max(0, −sⱼ ∂p̂/∂xⱼ)², over observed, in-support values (soft constraints) |
| λ selection | Largest λ in {0, 10, 10², 10³, 10⁴, 10⁵} whose inner-validation Brier is within one paired standard error of the best (the model follows clinical knowledge unless the data clearly disagree) |
| Comparators (§19) | Logistic regression, random forest, XGBoost, LightGBM, SVM, MLP (M1), ridge Cox; equal tuning budgets |
| Ablation (§20) | M1 MLP; M2 MLP + XAI; M3 CKINN; CKINN with FPG and BMI only; M4 + cross-population site balancing; M5 + fairness |
| Cross-population (§12, §18) | RICH → NAGALA (primary), NAGALA → RICH, and leave-one-site-out across RICH sites |
| Calibration (§17) | Calibration curves with intervals, intercept, slope, ECE, Brier, index of prediction accuracy |
| Fairness (§16) | Sensitivity, specificity, FNR, FPR, AUROC, AUPRC and calibration by sex, age group, glycaemic stratum and site |
| Explainability (§14–15) | SHAP, integrated gradients, monotonicity (gradient and finite-step) checks, feature stability across seeds, counterfactuals, clinical-reasoning text |

## Hypotheses (§23) and where they are answered

| Hypothesis | Estimate | Output |
|---|---|---|
| H1 CKINN > MLP (primary) | Brier(CKINN) − Brier(M1) in NAGALA with paired 95% interval; AUROC and AUPRC differences | `A/v4/contrast_ckinn_vs_m1.csv` |
| H2 Constraints improve calibration | Paired differences in calibration intercept, slope and ECE | `A/v4/contrast_ckinn_vs_m1.csv`, `calibration.png` |
| H3 Constraints improve generalisation | NAGALA → RICH contrast; pooled leave-one-site-out difference | `R/v4/contrast_ckinn_vs_m1.csv`, `A/v4/loso_pooled.csv` |
| H4 Explanations more clinically consistent | Wrong-way gradient shares, SHAP direction agreement, seed stability, counterfactual consistency | `A/v4/directions_*.csv`, `shap_*.csv`, `counterfactual_summary.csv` |
| H5 More consistent across groups | Group gaps, CKINN against M1 | `A/v4/fairness_gaps.csv` |
| RQ7 Risk before diagnostic thresholds | Baseline FPG < 5.6 mmol/L | `A/v4/fairness_groups.csv` |

Stage REPORT writes `proposal_questions_evidence.csv`, which fills this table with the computed estimates.

## Analysis history

This notebook implements protocol v4 (`PROTOCOL_v4_proposal_aligned.md`), written and committed before any v4 result.

Two earlier development runs used an operational protocol that constrained only fasting glucose and BMI, with λ ≤ 100. In those runs the unconstrained network already respected both directions, so the penalty was inactive and CKINN equalled the MLP. Protocol v4 restores the proposal's full sign table and a λ range under which the penalty can act. The earlier results were seen before v4 was written. Not examined before v4: the reverse direction, leave-one-site-out validation of the six-direction CKINN, the three-seed models and their explanations.

## How to run

1. Place the original files in `<project>/data/raw/`: the RICH workbook, the NAGALA workbook and the NHANES extract ZIP, identified by SHA-256.
2. In Colab, choose a CPU runtime and run all cells. Elsewhere, set `CKINN_ROOT`.
3. `RUN_MODE`: `SMOKE` is a workflow check with small samples. `STANDARD` is the reported analysis. `FULL` uses larger searches and five seeds.
4. All fits and expensive results are checkpointed. After an interruption, run the notebook again. `RUN_STAGE` can restrict a session to one stage group.

## Integrity

* Only the original cohort data are analysed.
* Artificial arrays appear only inside the software checks.
* Every reported number is computed from the data by this code. Null and unfavourable results are reported as computed.
* Participant-level data, models, predictions and individual explanations stay in `checkpoints/` and are never exported.
* Outputs are research risk estimates, not diagnoses or advice.
"""

SETTINGS_MD = r"""
## 01 · Paths and run settings

`RUN_STAGE` takes `ALL` or one of `AUDIT`, `CHECKS`, `INTERNAL`, `FINAL`, `EXTERNAL`, `REVERSE`, `LOSO`, `SENSITIVITY`, `FAIRNESS`, `EXPLAIN`, `SCREEN`, `EXPLORATORY` and `REPORT`. The run identity binds the inputs, the package code, the configuration, the device and the software versions.
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
print("Project folder:", ROOT)
print("Input folder:", RAW_DIR, "| exists:", RAW_DIR.is_dir())
print("Mode:", RUN_MODE, "| stage:", RUN_STAGE, "| device:", DEVICE)
"""

MODULES = {
    "data.py": "## 02 · `data.py`: input identity, harmonisation and eligibility\n\nSHA-256 identification, workbook audit (values and formulas), unit and code harmonisation, eligibility flow, three-year status (case, control or unknown) and a task-specific predictor allowlist.",
    "dictionary.py": "## 03 · `dictionary.py`: field dictionary\n\nType, unit, missingness and range of every source column.",
    "stats.py": "## 04 · `stats.py`: fixed-horizon estimands\n\nReverse Kaplan–Meier censoring weights, IPCW Brier score (Graf et al., 1999), weighted AUROC and AUPRC, calibration intercept and slope, decile calibration, threshold metrics and the participant bootstrap with censoring weights re-estimated in every draw.",
    "models.py": "## 05 · `models.py`: preprocessing, branched network, knowledge loss and comparators\n\nTraining-only imputation and scaling with missingness indicators; the branched network (§7); the clinical knowledge loss on wrong-signed probability derivatives (§8–9), differentiated during training; early stopping with epoch checkpoints; participant-disjoint stratified splits; IPCW-weighted comparators; SVM with cross-fitted Platt calibration; ridge Cox truncated at three years.",
    "architecture.py": "## 06 · `architecture.py`: architecture figure",
    "runner.py": "## 07 · `runner.py`: run identity, tuning, nested validation, freezing and scoring\n\nTask A trains in RICH and scores NAGALA; task R trains in NAGALA and scores RICH; task B trains on NHANES 2017–2020 and scores 2021–2023. All selection uses development rows only.",
    "analysis.py": "## 08 · `analysis.py`: shared post-freeze utilities\n\nOut-of-fold predictions, integrated gradients and probability averaging across seeds.",
    "stages.py": "## 09 · `stages.py`: display helpers and the IPCW support table",
    "verify.py": "## 10 · `verify.py`: software checks of the core package (artificial arrays only)",
    "proposal.py": "## 11 · `proposal.py`: protocol v4 analyses\n\n* CKINN λ selection by the paired one-standard-error rule, anchored on the M1 architecture.\n* Nested validation, freezing and external reports with bootstrap intervals.\n* Paired CKINN − M1 contrasts (H1, H2), the index of prediction accuracy and net benefit.\n* Leave-one-site-out validation with random-effects pooling (H3).\n* Sensitivity analyses and fairness (H5).\n* The training-size curve.\n* The proposal question table and the aggregate export.",
    "xai.py": "## 12 · `xai.py`: explainability (§14–15, H4)\n\nWrong-way gradient shares and finite-step checks for the six clinical directions, permutation SHAP (seed-averaged) with direction agreement and seed stability, integrated gradients with completeness checks, counterfactual risk changes, and clinical-reasoning text, stored privately.",
    "figures.py": "## 13 · `figures.py`: publication figures\n\nCalibration with 95% intervals and distinct markers and line styles (so that overlapping curves stay visible), ROC and precision–recall curves, decision curves, clinical-direction bars, SHAP importance, λ selection, a leave-one-site-out forest plot and the training-size curve. Saved as 300 dpi PNG and vector PDF.",
    "verify_proposal.py": "## 14 · `verify_proposal.py`: checks of the protocol v4 code\n\nThe one-standard-error rule, DerSimonian–Laird pooling and net benefit against hand calculations, and an end-to-end run of every v4 analysis on artificial records.",
}

CONFIG_MD = "## 15 · Configuration\n\nProfiles, search space, comparator list, knowledge layer (λ grid and directions), validation settings and support rules (at least 20 known cases and 20 known controls)."

STAGES = [
    ("## 16 · Environment check", r"""
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
    ("## 17 · Imports, seeds and run identity\n\nLightGBM 4.6 attaches placeholder column names to NumPy inputs, and scikit-learn then warns about missing feature names. The warning is suppressed because predictions are identical with and without names.", r"""
import inspect
import warnings

sys.path.insert(0, str(PACKAGE))
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
from IPython.display import display, Image
import numpy as np
import pandas as pd
from data import discover, js, csv, R, J, CORE, NONLAB, PLUS, UNITS, allowed
from stats import weights, metrics
from models import Prep, clinical, seed_all
from runner import Study, env_info
from stages import support_table, show
import architecture
import proposal as P
import xai
import figures
import verify
import verify_proposal

config = json.loads((PACKAGE / "config.json").read_text())
seed_all(config["seed"])
config.update(profile=RUN_MODE, stage=RUN_STAGE, device=DEVICE)
GROUPS = ["AUDIT", "CHECKS", "INTERNAL", "FINAL", "EXTERNAL", "REVERSE", "LOSO", "SENSITIVITY", "FAIRNESS", "EXPLAIN", "SCREEN", "EXPLORATORY", "REPORT"]
if RUN_MODE not in config["profiles"] or (RUN_STAGE != "ALL" and RUN_STAGE not in GROUPS):
    raise ValueError("Unknown RUN_MODE or RUN_STAGE")
st = Study(ROOT, config, paths=discover(RAW_DIR))


def active(group):
    selected = RUN_STAGE in ("ALL", group)
    if selected and RUN_MODE == "AUDIT" and group not in ("AUDIT", "CHECKS"):
        print("Skipped in AUDIT mode.")
        return False
    return selected


print("Run", st.run, "·", RUN_MODE, "·", st.device)
print(json.dumps({"budget": st.b, "knowledge layer": config["ckinn"], "validation": config["validation"]}, indent=2))
print(json.dumps(env_info(), indent=2))
"""),
    ("## 18 · AUDIT · Inputs, schemas and completeness\n\nInputs identified by SHA-256; workbook sheets and NHANES archive members; dimensions, identifiers, duplicates, missing and zero values; formula caches and unit conversions. The harmonised and eligible tables are cached privately.", r"""
if active("AUDIT"):
    print(pd.Series(st.identity["data"]).to_string())
    st.audit()
    info = json.loads((st.auditpath / "inventory.json").read_text(encoding="utf8"))
    print(info["zip_members"])
    for name in ["structure", "raw_features", "unit_checks"]:
        show(st.auditpath / (name + ".csv"))
"""),
    ("## 19 · AUDIT · Harmonised predictors, eligibility and three-year support\n\nThe source headers in both cohorts map to the analytical units. The FPG < 6.1 mmol/L rule aligns RICH with NAGALA, whose source study excluded higher baseline glucose. RICH participants all have at least two years of follow-up (a source exclusion), so RICH three-year cases are detections in years 2 to 3. The support table reports censoring survival at three years, the weights and sites without follow-up beyond three years.", r"""
if active("AUDIT"):
    display(pd.DataFrame([dict(input=k, RICH=R[k], NAGALA=J[k], unit=UNITS[k]) for k in PLUS]))
    show(st.auditpath / "cohort_flow.csv")
    display(support_table(st))
"""),
    ("## 20 · CHECKS · Software checks and the knowledge loss\n\nUnit checks on artificial arrays: censoring weights with ties, the full-N Brier denominator, penalty gradients and mask exclusion, finite differences, integrated-gradient completeness, the one-standard-error rule and random-effects pooling.", r"""
if active("CHECKS"):
    js(st.report / "software_checks.json", {"core": verify.run_checks(), "protocol_v4": verify_proposal.run_checks()})
    print(inspect.getsource(clinical))
    architecture.draw(st.fig / "architecture.png")
    display(Image(filename=str(st.fig / "architecture.png")))
"""),
    ("## 21 · INTERNAL · Nested validation in RICH (§19, §20)\n\nThe procedure for each outer fold (5 folds):\n\n1. Tune M1 and every comparator on inner splits (equal budgets).\n2. Select λ for CKINN and CKINN-FPG-BMI by the one-standard-error rule, using M1's architecture and epochs.\n3. Fit M4 and M5 with the CKINN λ.\n4. Refit on the outer training partition and score the held-out participants once.\n\nThe pooled out-of-fold metrics answer RQ1 within RICH.", r"""
if active("INTERNAL"):
    display(P.nested(st, "A")[["model", "n", "events", "brier", "auroc", "average_precision", "cal_intercept", "cal_slope"]])
"""),
    ("## 22 · FINAL · Selection on all RICH participants and freezing\n\nThe inner procedure is repeated on all eligible RICH participants. Final networks average three seeds. All models, preprocessing and hashes enter the freeze manifest before any external scoring.", r"""
if active("FINAL"):
    P.final_plans(st, "A")
    display(st.selection_table("A")[["model", "lambda_value", "epochs", "source_cv_brier", "threshold"]])
    show(P.table_dir(st, "A") / "lambda_path_final.csv")
    print(st.freeze("A"))
"""),
    ("## 23 · EXTERNAL · Frozen evaluation in NAGALA (RQ1, H1, H2)\n\nThe frozen pipelines are applied unchanged to NAGALA. The stage reports every metric with a 95% bootstrap interval, the paired CKINN − M1 contrasts (Brier score as the primary H1 contrast; AUROC, AUPRC and calibration as secondary), CKINN against every comparator, the index of prediction accuracy and net benefit.", r"""
if active("EXTERNAL"):
    st.score("A")
    table = P.external_report(st, "A")
    display(table[["model", "brier", "brier_lo", "brier_hi", "auroc", "auroc_lo", "auroc_hi", "average_precision", "cal_intercept", "cal_slope", "sensitivity", "specificity"]])
    show(P.table_dir(st, "A") / "contrast_ckinn_vs_m1.csv")
    show(P.table_dir(st, "A") / "contrast_ckinn_vs_all.csv")
    show(P.table_dir(st, "A") / "ipa.csv")
"""),
    ("## 24 · EXTERNAL · Calibration, discrimination and decision curves\n\nIn the calibration panels, decile estimates have 95% bootstrap intervals and each model has its own colour, marker and line style, so overlapping curves stay visible. Panel c shows the distribution of predicted risk.", r"""
if active("EXTERNAL"):
    for fn in (figures.calibration, figures.roc_pr, figures.decision_curve, figures.lambda_path):
        fn(st, "A")
    for name in ["calibration", "roc_pr", "decision_curve", "lambda_selection"]:
        show(st.fig / "A" / "v4" / f"{name}.png")
"""),
    ("## 25 · REVERSE · NAGALA → RICH (§12, H3)\n\nThe same procedure, run in the opposite direction:\n\n1. Nested 5 × 2 validation in NAGALA.\n2. λ selection, freezing and three-seed final networks.\n3. Frozen evaluation in all eligible RICH participants.\n\nNAGALA has far fewer events than RICH, so this direction also tests the constraints when development data are scarce.", r"""
if active("REVERSE"):
    display(P.nested(st, "R")[["model", "n", "events", "brier", "auroc", "average_precision"]])
    P.final_plans(st, "R")
    print(st.freeze("R"))
    st.score("R")
    table = P.external_report(st, "R")
    display(table[["model", "brier", "brier_lo", "brier_hi", "auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope"]])
    show(P.table_dir(st, "R") / "contrast_ckinn_vs_m1.csv")
    for fn in (figures.calibration, figures.roc_pr, figures.decision_curve):
        fn(st, "R")
    show(st.fig / "R" / "v4" / "calibration.png")
"""),
    ("## 26 · LOSO · Leave-one-site-out across RICH sites (§12 \"hospitals A, B, C → D\", H3)\n\nFor each RICH site with at least 20 known cases and 20 known controls, M1, CKINN, logistic regression and XGBoost are refitted on the other sites, using the configurations selected on all RICH participants. Per-site paired CKINN − M1 differences are pooled with a DerSimonian–Laird random-effects model.", r"""
if active("LOSO"):
    display(P.loso(st))
    show(P.table_dir(st, "A") / "loso_ckinn_vs_m1.csv")
    figures.loso(st)
    show(st.fig / "A" / "v4" / "leave_one_site_out.png")
"""),
    ("## 27 · SENSITIVITY · Eligibility, outcome window and HDL availability\n\n* Frozen specifications refitted with broader RICH eligibility (FPG < 7.0 mmol/L).\n* Frozen specifications refitted with ALT zeros retained.\n* NAGALA restricted to at least two years of follow-up (the RICH source design).\n* NAGALA with HDL treated as missing. HDL is missing for 45% of RICH participants.", r"""
if active("SENSITIVITY"):
    display(P.sensitivity(st)[["analysis", "model", "brier", "auroc", "cal_intercept", "cal_slope"]])
"""),
    ("## 28 · FAIRNESS · Group performance (§16, H5, RQ7)\n\nGroups: sex, age 20–39, 40–59 and 60+, and baseline FPG < 5.6 versus ≥ 5.6 mmol/L. They are evaluated in NAGALA and in the RICH out-of-fold predictions. Groups below the support rule are reported with their counts only. Gaps are differences between the best and worst supported groups.", r"""
if active("FAIRNESS"):
    groups = P.fairness(st, "A")
    display(groups[["stage", "model", "group", "n", "cases", "status", "sensitivity", "specificity", "fnr", "auroc", "cal_intercept"]])
    show(P.table_dir(st, "A") / "fairness_gaps.csv")
"""),
    ("## 29 · EXPLAIN · Clinical consistency and explanations (§14–15, H4)\n\nThe explanation outputs are:\n\n* wrong-way gradient shares for the six clinical directions in both cohorts, with finite-step checks;\n* permutation SHAP for M1, CKINN, logistic regression and XGBoost (500 NAGALA participants, RICH training background), averaged over seeds;\n* SHAP direction agreement and seed stability;\n* integrated gradients with completeness checks;\n* counterfactual risk changes when modifiable factors move to the development median;\n* clinical-reasoning text for the five highest-risk participants, stored privately.\n\nThese describe model behaviour, not causal effects or advice.", r"""
if active("EXPLAIN"):
    xai.directions(st, "A")
    xai.directions(st, "R")
    xai.explain(st, "A")
    for name in ["directions_gradient", "shap_direction", "shap_stability", "counterfactual_summary", "reasoning_summary"]:
        show(P.table_dir(st, "A") / f"{name}.csv")
    figures.directions(st, "A")
    figures.shap_importance(st, "A")
    show(st.fig / "A" / "v4" / "clinical_directions.png")
    show(st.fig / "A" / "v4" / "shap_importance.png")
"""),
    ("## 30 · SCREEN · Current status in NHANES (§3A)\n\nTraining on 2017–2020 and frozen evaluation on 2021–2023 use age, sex, BMI and systolic blood pressure, with CKINN constraints for age ↑, BMI ↑ and blood pressure ↑. Glycaemic measurements and self-reported diabetes never enter the models. The extract has no survey design variables, and its recoding of borderline answers is undocumented, so results are labelled as coded.", r"""
if active("SCREEN"):
    P.final_plans(st, "B")
    print(st.freeze("B"))
    st.score("B")
    table = P.external_report(st, "B")
    display(table[["model", "brier", "auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope"]])
    show(P.table_dir(st, "B") / "contrast_ckinn_vs_m1.csv")
"""),
    ("## 31 · EXPLORATORY · When do constraints help? Training-size curve\n\nThe frozen M1 architecture and the CKINN selection rule are applied to RICH subsamples of 1,000, 3,000, 10,000 and 30,000 participants (5 draws each), and each fit is scored in NAGALA. The stage reports paired differences per draw and their mean.", r"""
if active("EXPLORATORY"):
    display(P.size_curve(st))
    figures.size_curve(st)
    show(st.fig / "A" / "v4" / "training_size.png")
"""),
    ("## 32 · REPORT · Proposal questions, figures, manifest and export\n\nThe table maps RQ1–RQ7 and H1–H5 of the proposal to the computed estimates. Only H1 has a decision rule. The export ZIP holds aggregate tables, figures, reports, the package code and logs, and no participant-level data.", r"""
if active("REPORT"):
    figures.all_figures(st)
    display(P.proposal_table(st))
    print(P.export(st, PACKAGE))
"""),
]
