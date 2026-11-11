# CKINN notebook master plan (44 cells)

Prepared 2026-11-11 from three sources, in this order of authority: the locked analysis plan (`protocol/protocol_v1.0_LOCKED.md`, with `protocol/deviations.md`), the 26-page proposal (`CKINN_Proposal.pdf`), and the two raw workbooks. The 73-page `CKINN_All_in_One_Research_PDF.pdf` was treated as a document under review, not as a source of truth; every figure it states was recomputed from the workbooks.

No model was trained to write this plan. The counts below are descriptive checks on the raw files.

Assumption: the notebook runs in ordinary Jupyter with PyTorch available, and needs only the two workbooks beside it. If it still has to run on the iPad (no PyTorch), cells 17 and 21 use the NumPy backend already in `src/ckinn/nn_backend.py`; nothing else changes.

## 1. Review of the All-in-One PDF

### 1.1 What it states correctly (verified against the workbooks)

Rows and columns (NAGALA 15,464 by 31; RICH 211,833 by 25); events (373 and 4,174; 2.41% and 1.97%); follow-up ranges (164 to 4,732 days; 2.00 to 7.56 years); participants followed for at least 3, 4 and 5 years (RICH 104,014, 45,133, 8,912; NAGALA 10,998, 9,659, 8,345); events by 3 and 5 years (RICH 1,764 and 3,928; NAGALA 101 and 161); RICH missingness (HDL 44.64%, AST 58.20%, smoking and drinking 71.57%); the RICH event flag (4,174) against the diagnosis column (1,304); the NAGALA sex column labelled 1/2 but coded 1/0; NAGALA obesity-phenotype codes outside 1 to 8; every Excel column letter in its crosswalk. Its three methodological warnings are also right: binary cross-entropy at 3 years is invalid with censored follow-up, "MLP + XAI" is not a distinct predictive model, and cross-population testing is an evaluation design rather than a network layer.

### 1.2 Errors, omissions and conflicts

| # | Where | Problem | What the notebook does instead |
| --- | --- | --- | --- |
| 1 | Crosswalk; cells 025 to 027 | Maps NAGALA glucose, cholesterol, triglycerides and HDL to the mg/dL columns (AA, S, U, Q) and converts RICH into mg/dL. NAGALA already carries mmol/L columns (AB, T, V, R) that agree with the mg/dL ones to machine precision. The conversion is unnecessary, and "TC/TG likewise" hides three different factors (18.015, 38.67, 88.57). | Use the mmol/L columns; convert nothing. |
| 2 | Whole document | Family history of diabetes is never mentioned, although the proposal lists it as an input and as a clinical prior (pp. 5 to 6) and RICH records it for every participant (4,344 positive). | Family history enters the RICH extras analysis S4 with its "up" prior (cell 25). |
| 3 | Sections 2 and 3 | The baseline glucose ceilings differ and this is not noticed: NAGALA stops at 6.05 mmol/L, RICH at 6.99. The 7,193 RICH participants at or above 6.1 mmol/L (3.4%) carry 1,936 of the 4,174 events (46%). | Reported in cell 5; an eligibility-aligned sensitivity analysis is added (cell 39, exploratory). |
| 4 | "Nine-variable core model" | Treats the set that includes fasting glucose as the single primary model. In RICH, 3,404 of 4,174 events are defined by final-visit glucose of at least 7.0 mmol/L, so baseline glucose is close to a repeated measurement of the outcome. | Headline tier stays T2 (no glucose), as locked; T3 is reported and supports no claim alone. |
| 5 | Cells 043, 076, 078 | One development/tuning/holdout split with hyperparameter search. The locked plan uses repeated 5-fold cross-validation, leave-one-site-out and fixed hyperparameters, with lambda chosen inside the training data only. | Locked design (cells 20 to 28). |
| 6 | Sections 1 and 3; cell 085 | Makes Japan to China "secondary". The locked decision rule for C2 and C3 requires both directions at tier T2. | Both directions are primary (cell 30). |
| 7 | Cell 083 | Forbids any calibration with target data. The locked plan pre-specifies intercept-and-slope recalibration on 1,000 random target participants, evaluated on the rest. | Raw and recalibrated transport are both reported (cells 30 to 32). |
| 8 | Cells 047 to 052 | Fits imputation and scaling once and freezes them. Inside cross-validation this lets test folds inform preprocessing. | Preprocessing is refitted inside every training fold (cell 13). |
| 9 | Cells 054, 055, 058, 064, 081 | Requires tuned penalized Cox, random survival forest, IPCW classifiers, a discrete-time head and integrated Brier. Protocol section 13 dropped these before any run. | Not run; stated as limitations. |
| 10 | Cells 061, 068 | The estimand is cut off ("Define R(t") and "normalized components" is never defined; neither loss is written out. | Estimand and loss are those of protocol section 6 (cells 14 and 17). |
| 11 | Cell 087 | Asks for subgroup sensitivity and specificity without any threshold. PPV, NPV and F1 (proposal p. 22) are absent. | Thresholds are fixed in advance (cell 34, exploratory). |
| 12 | Cell 039 | Title reads "Quantify India-free China↔Japan covariate shift". Neither cohort is Indian; no result here extends to Indian populations. | Stated as a limitation (cell 44). |
| 13 | Cell 018; first table | Plans to look for zero sentinels but reports the laboratory values simply as "available". Exact zeros exist: RICH triglycerides 860 (637 at one site), HDL 102, ALT 2, AST 1; NAGALA HDL 11 (the mg/dL column is blank in those rows). | Zeros in these four variables become missing (deviation D-3; cells 6 and 9). |
| 14 | Cells 041, 087 to 089 | Proposes age by sex by site fairness audits. NAGALA has 101 events by 3 years, 17 of them in women. | Subgroups limited to sex and three age bands, with intervals; reported as underpowered. |
| 15 | 100-cell blueprint | Every code cell is a placeholder ("not implemented"); the companion `.ipynb` it names was not supplied. Several cells are optional or out of scope (058, 098). | 44 cells, each with executable content. |

One point the All-in-One PDF raises that the locked plan lacks, and which is adopted: RICH has no participant with under 2 years of follow-up and no event before 2.0 years, whereas NAGALA has 61 events before 2 years. A RICH-derived 3-year risk is therefore the risk of an event between years 2 and 3 among people event-free at 2 years. Cell 38 adds a two-year landmark sensitivity analysis in NAGALA (12,513 at risk at 2 years; 40 events between 2 and 3 years), labelled exploratory.

### 1.3 Corrections to the repository found during the same check

`data_dictionary.md` gives NAGALA HDL as 0% missing (11 are zero-coded) and RICH SBP and DBP as 0% missing (23 and 24 are blank). `scripts/00_harmonize.py` keeps zero-coded laboratory values as real values, so the existing `results/parts/cv_Cox_*` and `cv_XGB_*` files were produced with them; under deviation D-3 they are superseded and re-run by cell 20. The independent check of NAGALA sex coding now has a second line of evidence: height derived from weight and BMI averages 170.8 cm for code 1 and 158.3 cm for code 0.

## 2. How the proposal maps to the study

| Proposal item | In the notebook | Status |
| --- | --- | --- |
| RQ1 future T2D risk (pp. 3, 20) | R1 internal validation, cells 20 to 23 | Pre-specified |
| RQ2, H1 value of clinical knowledge | Claim C1 (non-inferiority, margin 0.01); CKINN minus MLP reported, cell 22 | Pre-specified |
| RQ3, H3 cross-population generalization | Claim C2: leave-one-site-out, covariate shift, both cohort directions, cells 26 to 32 | Pre-specified |
| RQ4, H2 calibration under transport | Claim C3, cells 30 to 32 | Pre-specified |
| RQ5, H4 clinically consistent explanation | Claim C4 (violation rate), cell 22; curves, attributions, what-if, cells 35 and 36 | C4 pre-specified; the rest exploratory |
| RQ6, H5 group disparities | Sex and age-band metrics, cells 26 and 31; group sensitivity and false-negative rate, cell 34 | Exploratory |
| RQ7 risk before diagnostic thresholds | Tiers T1 and T2 without glucose; participants below 5.6 mmol/L, cell 40 | Tiers pre-specified; subgroup exploratory |
| Loss, p. 9 | Same hinge-squared penalty, on the derivative of the log hazard; 3-year risk increases with the log hazard, so the sign is that of the proposal's derivative of risk | Pre-specified |
| Soft constraints, p. 10; priors table, pp. 6 to 7 | Protocol section 7; family history, HbA1c and exercise priors only within their own cohort (S4) | Pre-specified |
| Baselines, p. 17 | Cox (the censored-data counterpart of logistic regression), boosted Cox with and without monotone constraints, MLP | Random forest, LightGBM and SVM not run (protocol section 13) |
| Ablation, p. 18 | MLP against CKINN on identical folds; lambda path and leave-one-prior-out, cell 37 | Models 2, 4 and 5 are not separate trained models |
| Metrics, p. 22 | Protocol section 9; sensitivity, specificity, PPV, NPV, F1 in cell 34 | Threshold metrics exploratory |

Not possible with these two cohorts, and not attempted: HbA1c, waist, activity or family history across cohorts; prediabetes-to-diabetes progression (no intermediate state is recorded); fairness by ethnicity or socio-economic group; a third external cohort; knowledge graphs; a physiological differential-equation model (no insulin measurements); clinical expert agreement.

## 3. The cells

Type M is markdown, C is code. Every code cell prints the checks named in the last column; the next cell is issued only after those checks agree with the shared output. Cells 20 to 31 are resumable: each unit writes one file to `results/parts` and is skipped when the file exists.

### A. Set-up (cells 1 to 3)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 1 | M | Title, cohorts, outcome wording ("incident diabetes, diagnosis- or biomarker-defined"), claims C1 to C4, headline tier T2, 3-year horizon, the mapping in section 2. | Fixes what may be claimed before any number is seen. | None |
| 2 | C | Records Python and package versions, sets one thread and all seeds, creates the output folders, loads the protocol constants (tiers, priors, hyperparameters, seeds), verifies the protocol lock hash, starts the run manifest. | Reproducibility; every later cell reads its settings from one place. | Versions printed; lock hash matches |
| 3 | C | Verifies SHA-256 of both workbooks, reads each once, asserts shape and exact header list, caches unchanged copies. | Proves the analysis starts from the published files; later cells load in seconds. | 15,464 by 31; 211,833 by 25; both hashes match |

### B. Data forensics (cells 4 to 8)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 4 | C | Identifier uniqueness, duplicate rows, data types, missingness per column, code tables for sex, smoking, drinking, family history, site and NAGALA phenotype. | Establishes one row per participant and exposes coding problems. | No duplicate identifiers; HDL 44.64%, AST 58.20% missing in RICH |
| 5 | C | Outcome audit. NAGALA: `Incident DM`. RICH: event flag against the diagnosis column and final-visit glucose. Event-time ranges; events before 2 years; baseline glucose and HbA1c ceilings. | Defines the label and shows the two structural differences between the cohorts (rows 3 and 4 of section 1.2 and the landmark issue). | 373; 4,174 = 770 diagnosis only + 2,870 glucose only + 534 both; none unexplained; 0 and 61 events before 2 years |
| 6 | C | Counts exact zeros and implausible values in every laboratory and anthropometric variable, by site. States the D-3 rule. Deletes nothing. | A zero triglyceride or HDL is a code for "not measured"; left in, it is read as the lowest value in the cohort. | RICH zeros: TG 860, HDL 102, ALT 2, AST 1; NAGALA HDL 11 |
| 7 | C | NAGALA sex coding (counts against the source paper, derived height, event rate by code); agreement of each mg/dL and mmol/L pair. | Sex is a predictor in every tier; a reversed code would corrupt transport. Confirms no unit conversion is needed. | 8,430 and 7,034; 170.8 and 158.3 cm; ratios constant |
| 8 | C | At 2, 3, 4 and 5 years: participants followed that long, events by then, censored before then; reverse Kaplan-Meier follow-up; maximum follow-up by RICH site. | Confirms the 3-year horizon, shows why a binary 3-year label is invalid (50.1% of RICH censored earlier), reproduces deviation D-1. | Figures of section 1.1; sites 14 and 16 below 3 years |

### C. Harmonization and description (cells 9 to 12)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 9 | C | Builds one schema for both cohorts (mmol/L, male = 1, time in years, event, diagnosis-only event), applies D-3, defines tiers T1 to T4 and the extras, the prior registry of protocol section 7, and a blocklist asserting that final-visit glucose, the diagnosis flag, event, time, identifier and site never enter a predictor set. Saves the files with hashes and a participant-flow ledger. | Everything downstream reads these two files; the blocklist makes outcome leakage impossible by construction. | Rows in equal rows out; events unchanged; blocklist assertion passes |
| 10 | C | Table 1 by cohort and by event status; standardized differences between cohorts; missingness by tier and by RICH site. | The descriptive basis of any transport result (proposal p. 12); informative missingness must be visible. | Table renders; no value typed by hand |
| 11 | C | Kaplan-Meier 3-year risk by quintile of each constrained variable in each cohort; Kaplan-Meier curves by cohort. | Shows whether each prior holds descriptively ("generally", proposal pp. 6 to 7), including the age plateau in NAGALA that made the age prior soft. | No RICH event before 2 years |
| 12 | M | Gate 1: label, time origin, coding, D-3 and the leakage blocklist are confirmed. | A written stop point before any model is fitted. | Signed off |

### D. Toolkit with unit tests on toy arrays (cells 13 to 19)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 13 | C | Preprocessing fitted on training rows only (median or mode, missingness indicators, standardization); fold generators (repeated stratified 5-fold, leave-one-site-out, shift splits, inner 80/20 and 10% validation). | Protocol section 5; prevents row 8 of section 1.2. | Test: changing test rows leaves fitted statistics unchanged |
| 14 | C | Cox partial likelihood with ties, Breslow baseline hazard, Kaplan-Meier, absolute risk 1 - exp(-H0(3) exp(lp)). | The estimand of the whole study. | Toy values match hand calculation |
| 15 | C | C-index, IPCW time-dependent AUC and Brier at 3 years, scaled Brier, calibration slope and distance, observed-to-expected ratio, decile calibration, net benefit at 1% to 10%. | Protocol section 9; proposal pp. 15 to 16, 22. | Toy tests pass |
| 16 | C | Cox with ridge 0.01; boosted Cox; boosted Cox with the monotone constraints of section 7. One smoke-test fold on RICH tier T1. | Comparators for C1 to C3. | Fits complete; risks lie in [0, 1] |
| 17 | C | PyTorch MLP with Cox loss, hidden layers (64, 64); CKINN as the same network plus the penalty, with derivatives by automatic differentiation on standardized inputs. | The proposed model (proposal p. 9; protocol section 6). | Lambda = 0 reproduces the MLP exactly; penalty gradient matches finite differences; penalty is zero for a monotone toy network |
| 18 | C | Lambda selection rule; violation rate (finite differences, +0.25 SD, 5,000 participants); Nadeau-Bengio interval; paired bootstrap; the resumable unit runner and manifest writer. | Inference and bookkeeping for the decision rules. | Toy tests pass |
| 19 | M | Gate 2: all tests pass; no cross-cohort model has been fitted. | Stop point before results. | Signed off |

### E. R1 internal validation in RICH (cells 20 to 23)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 20 | C | Cox, XGB, XGB-mono: 3 repeats of 5-fold, tiers T1 to T4. | RQ1; baselines for C1. Replaces the superseded part files. | 60 units per model |
| 21 | C | MLP and CKINN on the same folds; lambda chosen inside each training fold and logged; out-of-fold predictions saved for T2 and T3. | RQ2; inputs to C1 and C4. | 60 units per model; lambda log complete |
| 22 | C | Mean and SD of every metric by model and tier; Nadeau-Bengio intervals for CKINN minus the best baseline (C-index) and CKINN minus MLP (violation rate) at T2. | The numbers on which C1 and C4 are decided. | Table complete |
| 23 | C | Decile calibration plots and decision curves from pooled out-of-fold predictions, T2 and T3. | Calibration and clinical usefulness (proposal pp. 15 to 16). | Figures saved |

### F. Sensitivity analyses (cells 24 and 25)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 24 | C | S1: diagnosis-only outcome, RICH T2. S2 and R5: T4 complete cases against all rows; missingness by site. | Tests dependence on the glucose-defined part of the outcome and on missing data. | 1,304 events in S1 |
| 25 | C | S4: extras within each cohort. RICH: family history, smoking. NAGALA: waist, exercise, fatty liver, GGT, and HbA1c in E3. | The only place the family-history, HbA1c and exercise priors of the proposal can be examined. | Units complete |

### G. Transport inside RICH (cells 26 to 28)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 26 | C | R2: leave-one-site-out, T1 to T3, 12 test sites (D-1); sex and age-band metrics at T2; predictions saved. | Generalization across centres (proposal p. 12, "Hospital A to D"). | 12 sites per tier |
| 27 | C | R4: train under 45 years, test 45 and over; train BMI under 25, test 25 and over. | Performance under real covariate shift. | 2 splits by 3 tiers |
| 28 | C | Summary tables and a site-level plot for R2 and R4. | Descriptive evidence for C2. | Saved |

### H. Cross-cohort transport (cells 29 to 32)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 29 | M | Gate 3: lock hash verified, deviations read, R1 complete, code frozen. | NAGALA outcomes meet a RICH-trained model only after this point. | Signed off |
| 30 | C | R3: RICH to NAGALA (T1 to T4) and NAGALA to RICH (T1 to T3), five models, raw and recalibrated on 1,000 random target participants; predictions and the RICH-trained networks saved. | RQ3 and RQ4; the analyses on which C2 and C3 rest. | Refuses to run if the lock hash differs |
| 31 | C | Paired bootstrap (200 resamples) of CKINN minus MLP and minus XGB-mono on scaled Brier and calibration distance, T2 and T3; subgroups at T2; S3 without sex. | The intervals named in the decision rules. | Intervals for both directions |
| 32 | C | Calibration plots (raw and recalibrated) and decision curves for each direction. | Shows what the summary numbers mean. | Saved |

### I. Exploratory analyses (cells 33 to 40; deviation D-4; none can support C1 to C4)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 33 | M | States that cells 34 to 40 are exploratory, were specified before their results existed, and are reported in full. | Keeps the confirmatory and exploratory evidence apart. | None |
| 34 | C | Censoring-adjusted sensitivity, specificity, PPV, NPV, F1 and false-negative rate at 3 years, at 3-year risk of 2% and 5% and at the top tenth of predicted risk (cut-off from training data); overall and by sex and age band. | Proposal pp. 15 and 22; RQ6. | Uses saved predictions only |
| 35 | C | Partial-dependence and individual curves for each constrained variable, MLP against CKINN; Integrated Gradients attributions (SHAP as a cross-check where it installs); rank stability across folds and between cohorts. | Proposal pp. 13 to 14; RQ5, H4. | Curves and rank correlations saved |
| 36 | C | Single-variable what-if changes within the observed range for a few participants, and a fixed explanation template worded as model behaviour, never as advice. | Proposal p. 14 ("clinical reasoning explanation"; no "lose 5 kg" statements). | Wording check |
| 37 | C | Performance and violation rate along the lambda path; CKINN refitted with one prior removed at a time (T2, one repeat of 5-fold). | Proposal p. 18: which constraint, if any, changes anything. | Units complete |
| 38 | C | Two-year landmark: RICH-trained 3-year risks evaluated in NAGALA participants still at risk at 2 years. | Matches the RICH design (section 1.2, closing paragraph). | 12,513 at risk; 40 events |
| 39 | C | R3 at T2 and T3 with RICH restricted to baseline glucose below 6.1 mmol/L. | Aligns eligibility with NAGALA (row 3 of section 1.2). | 204,640 participants; 2,238 events |
| 40 | C | Metrics among participants with baseline glucose below 5.6 mmol/L, from saved predictions. | RQ7: risk before conventional thresholds are reached. | No refit |

### J. Decisions and export (cells 41 to 44)

| Cell | Type | What it does | Why it is needed | Check before moving on |
| --- | --- | --- | --- | --- |
| 41 | C | Applies the rules of protocol section 11 mechanically and prints "supported" or "not supported" for C1 to C4, with the intervals. | No judgement enters after results are seen. | Four verdicts |
| 42 | C | Adherence audit: every unit present, counts, seeds, lock hash, deviations, manifest. | Evidence that the reported analysis is the planned one. | No missing unit |
| 43 | C | Writes the tables, figures and number macros in the format `scripts/make_report.py` reads, plus a results bundle to share. | Every number in the paper must come from `results/`. | Bundle written |
| 44 | M | Limitations: incident-risk framing; diagnosis- or biomarker-defined outcome; two East Asian screening cohorts; glucose dominance at T3; 101 external events at 3 years; cohort eligibility and follow-up differences. | Required statements for the thesis chapter and the paper. | None |

Totals: 44 cells, 38 code and 6 markdown.

Relation to the earlier nine-cell iPad notebook: its cells 1 to 3 correspond to cells 2 and 3 here, cell 4 to cell 9, cell 5 to cell 11, cell 6 to cells 13 to 15, cell 7 to cell 20, cell 8 to cell 17, and cell 9 to cell 21.

## 4. Working procedure

One cell is issued at a time. After it is run, the notebook with its output is shared; the output is compared with the check column above and with the locked plan; only then is the next cell issued. A failed check is fixed in that cell, not worked around in a later one. Anything that changes the plan is written to `protocol/deviations.md` first, with the date and whether results had been seen.
