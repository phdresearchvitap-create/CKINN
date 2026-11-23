---
title: "CKINN Objective 1: evidence review, audit, execution plan and submission plan"
subtitle: "Recorded incident diabetes by three years (RICH development, NAGALA frozen evaluation) and NHANES screening"
date: "23 November 2026"
---

# 0. Four things first

**Which run you supplied.** Three runs, none of them a finished scientific run.

* `CKINN_Objective1_Colab_SMOKE_2.ipynb` is the executed **SMOKE** run `9952757e3896a2d3`. It completed all 40 stages and exported its ZIP, but it ran an older software package (four of nine code files differ from version 2) on 6,000-row subsamples with two training epochs. It shows that the workflow executes. It says nothing about performance.
* `CKINN_Objective1_Colab_STANDARD_v2.ipynb` has **no saved outputs**. It is the notebook that produced the STANDARD run, not a record of it.
* The results ZIP holds an AUDIT run, the SMOKE run and a **partial STANDARD run `96ccd0057ac85891`**. I rebuilt that identifier from its parts. It matches exactly only for the version 2 code embedded in `STANDARD_v2`, with `broader_rich`, `site_holdout` and `alt_zero` all switched on, on the same Colab CPU runtime as the SMOKE run. Stages 5 to 28 finished: audit, all 5 outer folds of the nested RICH validation, final selection, freezing, NAGALA scoring, both paired contrasts, the calibration figure and the ablations. Stages 29 to 39 did not run, so there are no sensitivity analyses, explanations, subgroups, NHANES results, manifest or export ZIP. The timestamps stop during stage 29, the leave-one-site-out analysis.

**What the evidence currently supports.**

* **Primary contrast.** In NAGALA (15,459 people, 101 known three-year cases), M3 and M1 cannot be told apart. Brier(M3) − Brier(M1) is about −2 × 10⁻⁷ with a 95% interval spanning zero (Digest D4).
* **Matched secondary contrast.** Same picture: about −3 × 10⁻⁸, interval spanning zero.
* **Discrimination and calibration.** Both networks discriminate well externally (AUROC 0.86) but under-predict absolute risk (calibration intercept +0.32).
* **Why the contrast cannot differ.** The final networks have **no wrong-way gradients at all for FPG and BMI**, in RICH or in NAGALA (E2). The clinical penalty for the two pre-specified directions is therefore exactly zero at the solution. This is a non-binding constraint, not a failed one.
* **Replication.** I refitted the frozen models here. All 13 networks (and M2, which reuses M1), the reference model, LR, XGBoost, LightGBM and Cox reproduce your NAGALA Brier and AUROC to within 10⁻⁶ (E1). The pipeline ChatGPT wrote computes what it claims.

**The most important issue to resolve.** As specified, the primary comparison cannot answer the scientific question, whether clinical knowledge helps. Two reasons:

* the two constrained directions are already learned from 204,640 RICH participants;
* for the directions the unconstrained network does contradict (HDL− and SBP+), λ ≤ 100 is about a millionth of the data loss.

On top of this, the development and evaluation cohorts estimate different outcome windows. Every RICH participant has at least 2 years of follow-up, so all 785 RICH cases were detected between years 2 and 3. NAGALA includes detections from year 0 (D2), which distorts external calibration (E6).

How to make the knowledge test informative is a design decision for you and your supervisor (section 4.3). I have not changed the primary endpoint, comparison, eligibility or design.

**The exact next action.**

1. Open `CKINN_Objective1_Colab_STANDARD_v3.ipynb` in Colab on a CPU runtime.
2. Keep `RUN_MODE = "STANDARD"` and the three `OPTION_OVERRIDES` unchanged.
3. Run cells 00 to 04b and check that the resume check prints `same run: True`.
4. Then run `RUN_STAGE = "EXPLAIN"`, `"SUBGROUPS"`, `"SCREEN"`, `"EXT"`, `"SENSITIVITY"` and `"REPORT"`, in that order (section 5).

# 1. What actually ran

## 1.1 Runs and artefacts

| Item | Evidence | What it shows |
|---|---|---|
| SMOKE run `9952757e3896a2d3` | Executed SMOKE_2 notebook; manifest, summary, 23 tables, export ZIP in the results ZIP | Software path works end to end; 10 software checks pass; budgets of 2 folds, 2 trials, 2 epochs and 6,000 rows per cohort; AUROC 0.20 in task A reflects two epochs on a subsample, not a defect |
| AUDIT run `02d1d1bcd11fd209` | Manifest, summary, audit tables | Input hashes, schemas, eligibility flow, site support |
| STANDARD run `96ccd0057ac85891` | 10 tables and 1 figure in the results ZIP; **no manifest, summary or ZIP** | Stages 5 to 28 complete; 29 to 39 not run |
| STANDARD_v2 notebook | Source only, no outputs | Same code and options as the STANDARD run (identifier reconstructed exactly) |

Input hashes (SHA-256) match the deposited files for RICH (`ffd341a1…`), NAGALA (`671359ae…`) and the proposal (`fc5f4d61…`). Your NHANES input on Drive (`3af63d0d…`, a ZIP with two period CSVs) is a different file from the `nhanes_t2d_extract.csv` uploaded to this chat earlier (`d39795db…`), so task B could not be reproduced here.

## 1.2 Displayed, opened, referenced

* *Opened and read:* every table and figure in the ZIP, all notebook source and outputs, and all nine package files decoded from the base64 bundle in the STANDARD_v2 notebook.
* *Displayed only in the notebook, truncated:* the `display()` previews show the first rows only (for example the A_RICH_nested head). The CSV files hold the full tables.
* *Referenced but not supplied:* the STANDARD logs (`logs/STANDARD/96ccd0057ac85891/fits.jsonl`, `candidates.jsonl`, `diagnostics.jsonl`), the frozen manifest, and private predictions and checkpoints. These are needed to report failed fits and per-fold λ; see section 10.

## 1.3 Sample sizes, events and censoring (STANDARD)

RICH: 211,833 raw rows, 204,640 eligible. By year 3 there are 785 known cases, 101,111 known controls and 102,744 people of unknown status. The reverse Kaplan–Meier censoring survival at 3 years is G(3) = 0.497.

NAGALA: 15,464 raw rows, 15,459 eligible (five people under 20). By year 3 there are 101 known cases, 10,994 controls and 4,364 people of unknown status; G(3) = 0.717 (Digest D1 to D3).

No failed fit is visible in the tables. Every model in every outer fold has status `complete`, and all 200 bootstrap draws were valid for both contrasts. Candidate-level failures can only be confirmed from the logs.

## 1.4 Proposal against implementation

The proposal (CKINN_OBJ1.pdf, byte-identical to the earlier upload) asks seven research questions and five hypotheses about detection, future risk, transitions, transport, calibration, explanation and fairness. The implementation covers a narrower, well-defined part of this.

* **Decided before the full analysis (protocol v2):** the three-year horizon, eligibility (including FPG < 6.1), eight predictors, FPG+ and BMI+ penalties, the budgets, and the primary contrast. The matched positive-penalty comparison was added after the SMOKE run had selected λ = 0 and before the STANDARD run. The protocol says so, and there is no dated external registration.
* **Decided after the STANDARD results:** everything labelled EXT in this document, with rules fixed in `EXT_PRESPECIFICATION.md` (SHA-256 recorded with time stamps) before each EXT analysis was computed. E9b was added after E9 had been seen, and is labelled as such.
* **Not supported by these data:** subtype adjudication, repeated prediabetes transitions, glucose–insulin dynamics, clinician agreement, the knowledge graph and patient benefit. The protocol already lists these as unmeasured.

# 2. Audit of design and implementation

Severity: **H** means it changes what can be claimed, **M** means it must be reported or checked, **L** means cosmetic.

| # | Area | Finding | Evidence | Sev. | Action |
|---|---|---|---|---|---|
| 1 | Provenance | STANDARD identifier reproduced from code, configuration and runtime | Reconstruction script; section 1.1 | — | None |
| 2 | Reproducibility | 19 of 21 frozen models reproduce to < 10⁻⁶. RF and SVM differ slightly (RF AUROC 0.871 here against 0.867) under scikit-learn 1.9.1 here and 1.6.1 in Colab | E1 | M | Report package versions; keep scikit-learn 1.6.1 for reruns |
| 3 | Eligibility | FPG < 6.1 removes 7,193 RICH participants (3.4%) but 979 of 1,764 known three-year cases (55%). NAGALA already excluded FPG ≥ 6.1 at source, so the rule aligns the cohorts | D1, D2 | H (interpretation) | State the estimand as adults with baseline FPG < 6.1 mmol/L; keep the broader-RICH sensitivity (stage 29) |
| 4 | Prediction time and horizon | RICH minimum follow-up is 2.0 years (the source excluded intervals under 2 years and stopped follow-up at diagnosis), so no RICH case was detected before year 2. In NAGALA, 61 of 101 cases were detected before year 2 | D2 | **H** | Report as an estimand mismatch. E6 shows the calibration intercept moving from +0.32 to −0.52 when NAGALA is restricted the same way. The ranking contrast M3 − M1 is unaffected because both models share it |
| 5 | Censoring | Half of RICH has unknown three-year status. Sites 14 and 16 have nobody followed beyond 3 years (cases weighted, non-cases all unknown); sites 10 to 12 have G(3) of 0.07 to 0.17 (weights up to 15). Independent censoring across sites is doubtful | D3 | M | Site-specific G is already in stage 21 (`site_ipcw`). E11 (excluding sites 14 and 16) changes NAGALA Brier by about 5 × 10⁻⁶ |
| 6 | HDL availability | HDL is missing for 44.9% of RICH (15% to 62% by site); the crude known-case share is 0.88% with HDL against 0.65% without. In NAGALA, 0.07% are missing. The networks learned a wrong-signed (positive) HDL slope for 84% to 89% of eligible people | D2, E2, E8 | M | Report. E7: treating HDL as missing in NAGALA gives AUROC 0.866 against 0.860, with calibration intercept 0.22 against 0.32 |
| 7 | Leakage | None found. Preprocessing (median, scale, support limits, masks), IPCW G, early stopping, λ, epochs and thresholds all come from training or inner-validation rows. Outer folds are scored after selection; NAGALA only sees frozen models (the integration test patches `fit_model` and `Prep.fit` to fail during scoring). Folds are person-disjoint (asserted) | Code review | — | None |
| 8 | Nested and held-out-site analysis | Nested CV is as claimed: 5 outer × 2 inner folds, stratified by status and site. The held-out-site analysis tunes only M1 and M3, skips sites with fewer than 20 cases or controls, and has not run (stage 29) | `runner.internal` | M | Run stage 29; report skipped sites |
| 9 | M1, M3 and matched pair | M1 and M3 share the candidate schedule, seeds, batch order and stopping rules. The schedule ties one λ to each base configuration (λ = grid[trial mod 4]), so "M3 chose λ = 10" means trial 2's base settings won for both; the CV margin was 2.5 × 10⁻⁸. The matched pair is verified to differ only in λ (λ = 1 selected) and to share architecture, predictors, preprocessing, splits, seeds and 17 epochs. fixed_zero reproduces M1 exactly | D7, E1 | M | Report the margin; do not read the selected λ as evidence for the penalty |
| 10 | Penalty magnitude | The penalty acts on probability-scale gradients. For FPG and BMI it is exactly 0 at the final solution in both cohorts. For the expanded directions at λ = 1 it is 6 × 10⁻⁶ of the data loss. It binds only at λ ≥ 1,000 | E2, E9 | **H** | Design decision (section 4.3) |
| 11 | Brier and bootstrap | IPCW Brier (Graf) over all N; cases 1/G(T−), controls 1/G(3), unknowns 0. Paired percentile bootstrap draws the same NAGALA participants for both models and re-estimates G per draw. B = 200 in STANDARD. Intervals cover evaluation-sample variation only, not training or seed variation | `stats.py` | M | Use B = 1,000 for the paper (EXT does); report the conditionality; FULL adds 3 seeds |
| 12 | Calibration and discrimination | Intercept with logit offset; slope from a weighted logistic fit; weighted cumulative/dynamic AUROC; equal-frequency ECE; thresholds at the RICH inner-OOF 90th percentile of controls | `stats.py` | L | None |
| 13 | Comparator fairness | Same participants, predictors, masks and IPCW weights. Tuning budgets are unequal: networks get 6 trials, comparators 2 joint settings (RF depth ≤ 5, XGBoost/LightGBM 100 to 150 trees), and comparators get a single seed | `runner.candidates` | M | Report; FULL raises comparators to 3 trials; consider a larger comparator grid in any confirmatory rerun |
| 14 | Subgroups | NAGALA women have 17 known cases and people aged 60 or over have 11, below the 20-case rule, so sex gaps cannot be estimated externally | EXT subgroups | M | Report as unsupported; use RICH OOF for sex and site (stage 31) |
| 15 | External sample size | 101 known cases is fewer than the roughly 177 events Riley et al. needed in their worked example for a precise calibration slope | Riley 2021 | M | Report slope intervals; avoid strong calibration claims |
| 16 | NHANES (task B) | Label is HbA1c ≥ 6.5% among extract `Diabetes_self == 0`. The producer's recode of borderline and unknown responses is undocumented. Concurrent screening, not incidence; no survey design variables | Protocol B; SMOKE | M | Keep `as_coded`; supply the producer crosswalk before any "verified" analysis |
| 17 | Warnings | LightGBM "X does not have valid feature names": LightGBM 4.6 gives NumPy inputs placeholder names; predictions are identical with and without names (tested). No Cox convergence warning in the SMOKE outputs; STANDARD logs not supplied | Local test | L | v3 filters the warning; send the logs |
| 18 | Cost of stage 29 | The three enabled sensitivity options roughly double the compute of stages 21 to 28 | Timestamps | M | Run it last; it resumes after a disconnect |
| 19 | Export | The stage 39 ZIP holds aggregate tables, figures, reports, software and logs, and excludes checkpoints and pickles (tested) | `analysis.export` | — | None |

# 3. New evidence computed in this session

Every number below is in `RESULTS_DIGEST.md`, generated from stored tables by `make_digest.py`. "Local" means computed here on your two workbooks, with the unchanged version 2 code and the extension module.

**Replication (E1).** Rebuilding the frozen plans from `source_selection.csv` and refitting reproduces your NAGALA metrics. All 13 networks, M2, LR, XGBoost, LightGBM and Cox match to < 10⁻⁶; the reference model matches exactly; RF and SVM differ slightly (scikit-learn versions). The local stage 26 contrasts and their intervals agree with yours to within 10⁻¹⁰.

**Is the constraint active? (E2, E8 and stage 30).** No, for the pre-specified directions.

* FPG, BMI and age have no wrong-way gradients in any final network, in either cohort.
* HDL (prior −) goes the wrong way for 82% to 89% of people and SBP (prior +) for 50% to 64%, in every network including `expanded_prior`. Both effects are near-flat: a 0.1 mmol/L HDL step changes risk by 0.0002 on average and a 5 mmHg SBP step by 0.00006, against 0.0014 for 0.1 mmol/L of FPG.
* Integrated gradients rank FPG first, then age, then BMI (completeness error under 5 × 10⁻⁶).

**Comparison with simple baselines (E3 to E5, B = 1,000, conditional on the fitted models).**

* **Brier:** M1 is clearly better than the intercept-only reference, Cox and the SVM. It is slightly better than LR (interval just excluding zero) and indistinguishable from RF, XGBoost and LightGBM.
* **Index of prediction accuracy:** about 2.6% for M1 and M3 and 2.3% for LR. The absolute gain over the reference is small because three-year risk is under 1%, even though AUROC is high.
* **Net benefit:** M1 beats "treat all" and "treat none" at every risk threshold from 0.5% to 5%, and is close to LR.

**Estimand alignment (E6).** Restricting NAGALA to people followed for at least 2 years, as in RICH, leaves 12,509 people and 40 cases. Calibration flips from under-prediction (intercept +0.32) to over-prediction (intercept −0.52), while AUROC stays at 0.86. Absolute-risk transport therefore cannot be judged without fixing the outcome window.

**Penalty-strength path (E9, E9b).**

* With the six expanded directions, the RICH inner-validation Brier is flat across λ from 0 to 10⁵, and the pre-specified rule selects λ = 1.
* At λ ≥ 1,000 the penalty binds: HDL wrong-way share falls from 85% to below 25%, and to almost zero at λ = 10⁴. NAGALA error then falls. At λ = 10⁴, Brier 0.007567 against 0.007588, AUROC 0.868 against 0.860, calibration intercept 0.29 against 0.32. The paired difference is −2.1 × 10⁻⁵, with an interval excluding zero.
* The benefit appears externally but not in RICH validation, which is what a correct prior that the source data contradict would produce. It is post hoc, from a single seed, chosen after looking at a λ path, and its interval ignores training variability. It is a hypothesis for a confirmatory study, not a finding.

**Training-size curve (E10).** RICH subsamples of 1,000, 3,000, 10,000 and 30,000 people (4, 12, 38 and 115 known cases), 5 draws each, scored in NAGALA:

* The penalised networks beat M1 in 37 of the 40 penalised-versus-M1 comparisons.
* The gain shrinks as data grow: mean AUROC gain for the expanded penalty is 0.10, 0.035, 0.015 and 0.012.
* In small samples the internal rule usually chooses strong penalties (λ of 1,000 to 100,000 in 37 of 40 arm-draws).

These are exploratory, but they are the clearest evidence that the constraint is useful when data are scarce and redundant when they are not.

**Sites (E11).** Excluding the two sites without three-year support changes NAGALA Brier by about 5 × 10⁻⁶.

# 4. What would and would not support a CKINN claim

## 4.1 Decision rules for the primary contrast (fix these before the next confirmatory run)

Let Δ = Brier(M3) − Brier(M1) in NAGALA, with its paired 95% interval.

* **Advantage for CKINN.** All of the following must hold:
  1. M3's RICH selection yields λ > 0;
  2. the penalty is active at the solution (E2 shows a material reduction in wrong-way share, or a penalty above about 1% of the data loss during training);
  3. the whole interval for Δ lies below zero;
  4. the gain is practically relevant, for example an IPA gain of at least 0.5 percentage points (a threshold you must fix in advance);
  5. calibration intercept and slope are no worse;
  6. M3 is at least as good as the best simple baseline (LR or gradient boosting).
* **Inconclusive.** The interval contains zero; or the penalty is inactive; or λ = 0 is selected. **This is the current result**, and the explanation is a non-binding constraint.
* **Favours the baseline.** The whole interval lies above zero, or calibration is worse.
* **If M3 selects λ = 0 again,** M3 is M1 for that candidate and Δ measures the tuning rule, not the penalty. Report it as "the tuning rule did not choose a penalty", and use the matched contrast and the λ path to describe what a penalty would have done.

## 4.2 Claims that are defensible now

1. An eight-variable network trained on 204,640 Chinese health-check participants discriminated three-year recorded diabetes well in an independent Japanese cohort (AUROC 0.86). Its error was comparable to gradient boosting and slightly lower than logistic regression.
2. Soft FPG and BMI direction constraints did not change external error, because the unconstrained network already respected those directions.
3. Absolute risk was under-predicted. Part of this is a design difference between the cohorts (outcome windows), not only population shift.
4. Exploratory only: enforcing directions that the source data contradicted (HDL, SBP) reduced external error slightly, and constraints helped most when training data were scarce.

Claims that are **not** supported: clinical utility, treatment decisions, fairness across sexes in NAGALA, generalisation beyond East Asian health-check attendees, and anything about NHANES beyond the as-coded extract.

## 4.3 Design decisions that are yours to make

1. **Penalty scale and λ grid.** Keep the probability-gradient penalty but pre-register λ ∈ {0, 10², 10³, 10⁴, 10⁵}; or move to a logit-scale penalty, under which λ of order 1 binds. Either changes the definition of M3.
2. **Prior set.** Keep FPG+ and BMI+ (redundant here), or make the six expanded directions primary (they bind and, post hoc, helped).
3. **Selection rule.** RICH random cross-validation does not reward transport. A leave-site-out selection rule within RICH would target robustness, which is the stated purpose of the constraints.
4. **Estimand.** Keep "detection by year 3"; or co-primary "detection in years 2 to 3 among people followed for at least 2 years" (estimable in both cohorts); or accept the mismatch and report it.
5. **Profile.** FULL (3 seeds, 1,000 bootstrap draws, 12 trials) gives seed-averaged models and seed variation for the paper. It is a new run, roughly 2 to 3 times the STANDARD cost.

My recommendation is to finish the current STANDARD run as pre-specified and report it as the primary, inconclusive result with its mechanism. Then register a short confirmatory protocol using options 1 to 3, with the expanded prior set, a binding λ grid and leave-site-out selection, plus the FULL profile. Run that as a new, clearly labelled analysis.

# 5. Ordered execution plan

| P | Exact action | Scientific reason | Cell or file | Expected output | Compute | Pass/fail check |
|---|---|---|---|---|---|---|
| 0 | Open v3 on a **CPU** runtime; keep STANDARD and the three overrides; run 00 to 04b | Reuse stages 21 to 28 | Cells 00–04b | `same run: True` | Minutes | Fail: a field differs (often versions or the platform string). Install the listed versions and restart, or accept a new run |
| 1 | `RUN_STAGE="EXPLAIN"` | Direction audit and attributions of the frozen models | Stage 30 | `directions.csv`, `attributions.csv`, `perturbation.csv`, ICE figure | About 10 min | All finite gradient checks True; completeness error < 10⁻⁴ (local: 5 × 10⁻⁶) |
| 2 | `RUN_STAGE="SUBGROUPS"` | Pre-specified groups with support rule; RICH OOF sex and site | Stage 31 | `subgroups.csv`, `sex_gaps.csv` | 0.5 to 1 h | Women and age 60 or over flagged as insufficient in NAGALA |
| 3 | `RUN_STAGE="SCREEN"` | Task B as coded | Stages 33–35 | `B_as_coded/*` | About 0.5 h | Frozen B manifest exists; labels as coded |
| 4 | `RUN_STAGE="EXT"` | Penalty activity, baselines, estimand, λ path, size curve | EXT cell | `A/EXT/*.csv` (14 files) | 2 to 4 h | E2 on your frozen models matches the local E2 (FPG and BMI share 0) |
| 5 | `RUN_STAGE="SENSITIVITY"` | Site-held-out, broader RICH, ALT zero | Stage 29 | `A_RICH_site.csv`, `A_RICH_broad_nested.csv`, `A_RICH_alt0_nested.csv` | 9 to 12 h, resumable | Sites without support listed as "insufficient support" |
| 6 | `RUN_STAGE="REPORT"` | Fit budget, narrative, manifest, export | Stages 36–39 | `fit_budget.csv`, `summary.md`, `run_manifest.json`, aggregate ZIP | Minutes | Manifest lists every table; ZIP contains no `.pkl` |
| 7 | Decide section 4.3 with your supervisor; write the confirmatory protocol | Make the knowledge test informative | Protocol | Dated, hashed protocol | — | Written before any confirmatory fit |
| 8 | Optional: FULL profile under the confirmatory protocol | Seed variation, B = 1,000 | New run | New run ID | 40 h or more | All folds complete; ≥ 80% valid bootstrap draws |
| 9 | Build manuscript tables from the digest | No hand-typed numbers | `make_digest.py` | Tables 1 to 4, S1 to S10 | Minutes | Every number traceable to a CSV |

**Resuming.** Every fit, split, plan, prediction and epoch is checkpointed on Drive under `checkpoints/<mode>/<run>/`. Rerunning the same `RUN_STAGE` skips finished work.

**What changes the run identifier, and so forces refitting:**

* any byte of `study/*.py` or `config.json`;
* the profile or any option (including `OPTION_OVERRIDES`);
* the device (a GPU runtime turns `auto` into `cuda`);
* the Python version, platform string or package versions;
* the input files.

**What does not change it:** `RUN_STAGE`, notebook cells outside the embedded package (including the v3 additions and the EXT module), and the results folder. The EXT module lives outside `study/` for this reason.

# 6. Code changes in version 3 and the tests run

**Changed:**

* `CKINN_Objective1_Colab_STANDARD_v3.ipynb`, built by `build_notebook_v3.py` from the submitted v2 notebook, which is preserved in `evidence/`;
* a LightGBM warning filter;
* a resume check (cell 04b);
* `RUN_STAGE = "EXT"`;
* the EXT stage, embedding `ext/ckinn_ext.py` with a SHA-256 check;
* the v3 run recipe.

**Unchanged:** the embedded v2 package (byte-identical, checked).

**Run here:**

* v2 package software checks: 10 of 10 passed;
* the v2 integration test on toy records: passed;
* EXT unit tests (net benefit and IPA by hand, plan reconstruction): 3 of 3 passed;
* EXT end-to-end test on toy records through the same `Study` API (freeze, score, all EXT analyses, rerun restores every checkpoint with identical tables): passed;
* notebook cells 01 to 03 and the EXT decode with a stand-in for the Drive mount: passed;
* the local scientific replication E1 to E11 on the real RICH and NAGALA workbooks.

**Needs Colab:** cells 04 onwards with your four inputs, including NHANES; the real resume against your checkpoints; and every stage in section 5.

Never shown here: anything about task B, and the RICH OOF subgroup analyses (these need your private OOF predictions).

# 7. Submission plan

## 7.1 Contribution

The paper is a carefully specified external validation of a knowledge-constrained neural network for three-year recorded diabetes, with a negative primary result explained by a mechanism. The constraint was redundant where data were informative (FPG and BMI at n = 204,640). Exploratory analyses suggest it can help transport and small-data training where learned directions conflict with clinical knowledge.

The practical significance is methodological. Knowledge constraints should be judged by whether they are active, against strong simple baselines, and under an aligned estimand. For a reader, a null CKINN result with this explanation is more useful than an unexplained small gain.

## 7.2 Outline

1. Introduction: knowledge-informed ML for diabetes risk; why directional constraints might help transport; aim and pre-specified contrast.
2. Methods: data sources and eligibility (flow chart); outcome and the three-year IPCW estimand; predictors and missing data; models (M1, M3, comparators); tuning and freezing; metrics; paired bootstrap; sensitivity and EXT analyses with their status (pre-specified or post hoc).
3. Results: cohorts (Table 1); internal validation; external validation (Table 2, Figure 2); primary and matched contrasts (Table 3); constraint activity (Figure 3); baselines, IPA and net benefit (Table 4, Figure 4); sensitivity; exploratory λ path and size curve (Figure 5).
4. Discussion: non-binding constraints; estimand mismatch and calibration; HDL and SBP; when constraints help; limitations; implications.

## 7.3 Tables, figures and their sources

| Item | Content | Source file(s) |
|---|---|---|
| Fig 1 | Flow chart | `audit/cohort_flow.csv`; D1 |
| Table 1 | Baseline characteristics by cohort and status | `audit/baseline.csv`; `feature_set_availability.csv` |
| Table 2 | External metrics for every model | `A/external_metrics.csv` (D5) |
| Table 3 | Primary and matched contrasts | `A/paired_brier.csv`, `A/matched_positive_paired_brier.csv` (D4) |
| Table 4 | Baselines: paired Brier, IPA, net benefit | `A/EXT/E3`, `E4`, `E5` |
| Fig 2 | Calibration, ROC and PR curves | `A/calibration_roc_pr.png`, `calibration_bins.csv` |
| Fig 3 | Constraint activity (wrong-way shares; λ path) | `A/EXT/E2`, `E8`, `E9`, `E9b` |
| Fig 4 | Decision curve | `A/EXT/E5` |
| Fig 5 | Training-size curve | `A/EXT/E10_*` |
| S1 | IPCW support by site | `A/ipcw_support.csv` (D3) |
| S2 | Nested RICH folds and pooled OOF | `A_RICH_nested.csv`, `pooled_oof.csv` (D6) |
| S3 | Selected configurations | `source_selection.csv` (D7) |
| S4 | Ablations and component variants | `ablations.csv` |
| S5 | Sensitivity: site-held-out, broader RICH, ALT zero, site G | Stage 29 tables; `A_RICH_nested.csv` (`site_ipcw` rows) |
| S6 | Estimand-aligned NAGALA and HDL missing | `A/EXT/E6`, `E7` |
| S7 | Subgroups with denominators | `subgroups.csv`, `sex_gaps.csv`, `EXT_external_subgroups.csv` |
| S8 | Attributions and direction checks | `attributions.csv`, `directions.csv` |
| S9 | NHANES as coded | `B_as_coded/*` |
| S10 | Reproducibility: run identity, versions, E1 | `run_manifest.json`, `E1_reconstruction_check.csv` |

Checks before drafting Results:

* all stages complete;
* the manifest frozen before scoring (the frozen manifest hash is in `run_manifest.json`);
* every bootstrap at least 80% valid;
* fit failures reviewed in `fits.jsonl` and `candidates.jsonl`;
* support flags respected;
* every number generated from tables (extend `make_digest.py`);
* pre-specified and post hoc analyses labelled.

## 7.4 TRIPOD+AI (Collins et al., BMJ 2024), by section

Check item numbers against the official checklist.

* **Title and abstract:** development and external validation; three-year recorded diabetes; adults with FPG < 6.1 mmol/L at health checks; the TRIPOD+AI for Abstracts items.
* **Introduction:** clinical context; existing models; rationale for knowledge constraints; objectives, including the pre-specified primary contrast.
* **Methods, data:** sources (RICH, NAGALA, NHANES extract), deposits and DOIs, dates, settings; eligibility with counts (D1); the source exclusions (RICH follow-up of at least 2 years; NAGALA medication, alcohol and liver exclusions).
* **Methods, outcome:** definition and ascertainment per cohort (RICH: FPG ≥ 7.0 or self-report; NAGALA: deposited flag including HbA1c); the 3-year horizon; censoring and IPCW; blinding is not applicable.
* **Methods, predictors:** the eight predictors, units, harmonisation, timing.
* **Methods, sample size:** events per cohort; Riley et al. for external validation precision.
* **Methods, missing data:** median imputation with masks fitted within folds; HDL missingness by site.
* **Methods, analysis:** architecture, loss and penalty; tuning; comparators and their budgets; nested CV; freezing; metrics; bootstrap; thresholds.
* **Methods, class imbalance:** IPCW only, no resampling.
* **Methods, fairness:** groups, support rule.
* **Methods, model output:** a probability for one horizon.
* **Methods, training versus evaluation:** development and evaluation differences (D2, E6).
* **Open science:** ethics, funding, conflicts, protocol (protocol v2 and EXT pre-specification, with dates and hashes), registration (none prospective; state it), data availability (Dryad deposits; NHANES public), code availability (the notebook and package; archive with a DOI, for example Zenodo).
* **Patient and public involvement:** state if none.
* **Results:** participant flow and characteristics; model specification (selected configurations, parameter counts); performance with intervals (discrimination, calibration, overall, net benefit); updating (target intercept update is disabled; say so).
* **Discussion:** interpretation, limitations (below), usability and the next steps.

## 7.5 Risk of bias and applicability (PROBAST+AI, Moons et al., BMJ 2025): a self-assessment for your supervisor to check

* **Participants and data sources:** high concern for applicability, and possible bias. RICH and NAGALA are selected health-check attendees with repeat visits; RICH excludes follow-up under 2 years; both are East Asian.
* **Predictors:** low risk. Routine baseline measurements are harmonised. Missingness differs (HDL) and is handled consistently.
* **Outcome:** high concern. Ascertainment differs between cohorts (HbA1c in NAGALA only); detection is at visits; the windows are mismatched.
* **Analysis, development:** low risk for leakage and tuning, with nested CV. Moderate for censoring (site-dependent), for the unequal comparator budget and for a single seed in STANDARD.
* **Analysis, evaluation:** 101 events gives imprecise calibration; intervals are conditional on the models.

## 7.6 Ethics and data-use information you must supply

* Ethics approval or exemption from the VIT-AP institutional ethics committee for secondary analysis of de-identified data.
* The licence and terms of each deposit (RICH Dryad 10.5061/dryad.ft8750v; NAGALA Dryad 10.5061/dryad.8q0p192). Verify them; I could not open the deposit pages from this session.
* The original studies' ethics statements: Chen et al. 2018 for RICH; Okamura et al. 2019 for NAGALA.
* For NHANES: the NCHS public-use status, plus the producer and documentation of your extract (the DIQ010 recode).
* Funding, conflicts and author contributions.

## 7.7 Limitations the paper should state

* Incident-risk framing with a visit-detected, diagnosis- or biomarker-defined outcome that differs between cohorts.
* The FPG < 6.1 restriction removes 55% of RICH three-year cases.
* The RICH design excludes detections before year 2.
* Half of RICH is censored before year 3, with sites lacking support.
* Two East Asian health-check cohorts.
* FPG dominates the predictions.
* Calibration is assessed with only 101 external events.
* Comparator tuning budgets are unequal, with a single seed.
* The penalty was inactive for the pre-specified directions.
* The EXT analyses are post hoc.
* NHANES is a concurrent screening extract without its survey design.
* No clinical utility or implementation evidence.

# 8. Journals (to be verified at the primary source)

I could not open SCImago (scimagojr.com) or the publisher pages from this session; the network blocks those hosts. Quartiles are therefore **not verified here**. Before choosing, check the most recent SJR and JCR year, subject category and quartile via scimagojr.com and your library's Journal Citation Reports, and read each journal's guide for authors.

Candidates by scope fit:

| Journal | Why it fits | What to verify |
|---|---|---|
| International Journal of Medical Informatics | Clinical prediction and AI evaluation in health informatics | Quartile (a third-party site reports SJR 2025 of 1.278 and Q1 in Health Informatics, unverified); TRIPOD+AI requirement; APC |
| Journal of Biomedical Informatics | Methods papers on knowledge-informed ML | Quartile (a third-party site reports Scopus CiteScore Q1 in Health Informatics, which is not SJR); methods emphasis |
| Diagnostic and Prognostic Research (BMC) | Prediction-model methodology, external validation, IPA | Quartile (a third-party OpenAlex-based site reports Q1 in Epidemiology, unverified); APC |
| BMC Medical Research Methodology | Methodological evaluation of prediction approaches | Quartile; APC |
| JAMIA | High-profile informatics venue | Quartile; fit for a negative methods result |

Quartile helps choose a venue. Acceptance depends on the completed study and editorial review.

# 9. References (checked in PubMed)

* Collins GS, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378 (PMID 38626948)
* Moons KGM, et al. PROBAST+AI. BMJ 2025;388:e082505. doi:10.1136/bmj-2024-082505 (PMID 40127903)
* Graf E, et al. Assessment and comparison of prognostic classification schemes for survival data. Stat Med 1999;18:2529–45. doi:10.1002/(sici)1097-0258(19990915/30)18:17/18<2529::aid-sim274>3.0.co;2-5 (PMID 10474158)
* Chen Y, et al. Association of body mass index and age with incident diabetes in Chinese adults. BMJ Open 2018;8:e021768. doi:10.1136/bmjopen-2018-021768 (PMID 30269064)
* Okamura T, et al. Ectopic fat obesity presents the greatest risk for incident type 2 diabetes. Int J Obes 2019;43:139–48. doi:10.1038/s41366-018-0076-3 (PMID 29717276)
* Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making 2006;26:565–74. doi:10.1177/0272989X06295361 (PMID 17099194)
* Kattan MW, Gerds TA. The index of prediction accuracy. Diagn Progn Res 2018;2:7. doi:10.1186/s41512-018-0029-2 (PMID 31093557)
* Van Calster B, et al. A calibration hierarchy for risk models. J Clin Epidemiol 2016;74:167–76. doi:10.1016/j.jclinepi.2015.12.005 (PMID 26772608)
* Riley RD, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med 2021;40:4230–51. doi:10.1002/sim.9025 (PMID 34031906)

# 10. Files to return from your next Colab run

1. The executed v3 notebook (File > Download .ipynb) **with outputs**.
2. `results/reports/STANDARD/<run>/CKINN_STANDARD_aggregate.zip`, which contains tables, figures, reports and logs.
3. If stage 39 did not finish: the folders `results/tables/STANDARD/<run>/`, `results/reports/STANDARD/<run>/`, `results/figures/STANDARD/<run>/` and `logs/STANDARD/<run>/` (`fits.jsonl`, `candidates.jsonl`, `diagnostics.jsonl`, `environment.json`).
4. The printed output of cell 04b (resume check).
5. For task B: the producer's documentation of the DIQ010 recode, if you have it.

Never send `checkpoints/` or `data/`; they hold participant-level files.
