# Protocol v2: CKINN versus M1 and M2 on real data (NAGALA, Rich Healthcare Group, NHANES)

Status: **frozen 2026-11-20, after the version 1 results had been seen and before any version 2 result existed.** The configuration hash written in section 9 is checked by notebook cell 16; a run whose configuration differs from it stops. Any later change is appended to section 10 with its date and the reason, and is labelled post hoc.

Version 1 (`frozen_protocol.json`, hash prefix `d5840db6d905283e`) and its results are kept unchanged as the record. Everything below that differs from version 1 is an amendment made after seeing the version 1 results; section 2 lists each one with its reason.

Labels used throughout: **[P1]** pre-specified in v1 and carried over unchanged; **[P2]** pre-specified here in v2, decided after seeing v1 results and before any v2 result; **[E]** exploratory (reported in full, no multiplicity adjustment, cannot change a verdict); **[PH]** post hoc (added after v2 results were seen; none at the freeze).

Outputs are research risk estimates. Nothing in this study is a diagnosis or a clinical recommendation.

## 1. Question and estimand

Does adding the clinical-knowledge penalty to a mask-aware neural network (CKINN versus M2, the same network without the penalty) or to the simpler neural baseline (CKINN versus M1) change discrimination, calibration, transport, explanation behaviour or subgroup performance when predicting type 2 diabetes from routine measurements without glucose or HbA1c?

The estimand for every model comparison is the expected difference in a metric between two training procedures, averaged over network seeds and cross-validation splits, in the evaluation population of each cohort. [P2]

## 2. Amendment log (all decided after seeing v1 results)

| ID | Change from v1 | Reason (audit finding) | Label |
| --- | --- | --- | --- |
| A1 | CKINN, M2 and M1 share the seed, inner split and training rows within each fold; only the model differs. | F2 | P2 |
| A2 | Repeated stratified 5-fold cross-validation, 5 repeats, in every cohort; network seed r in repeat r; seed study with 5 seeds on repeat 0. | F4 | P2 |
| A3 | No cap on training rows inside a fold (v1: 40,000). Development subset cap of 20,000 rows kept. | F11 | P2 |
| A4 | Equal tuning budget of 32 evaluated configurations for M1, M2 and CKINN; CKINN tunes lambda over an 8-value grid with a declared one-standard-error rule. | F5, F6 | P2 |
| A5 | Penalty evaluated on the whole mini-batch (v1: 256 rows per batch). | brief, F1 | P2 |
| A6 | Inference from 2,000 paired bootstrap draws with p-values taken from the draws; DeLong and the corrected repeated-CV t-test as cross-checks; equivalence (TOST) and minimum detectable effect reported; cluster bootstrap by site in RICH. | F7 | P2 |
| A7 | Two hypothesis families, CKINN versus M2 (primary) and CKINN versus M1, each with Holm correction across its primary endpoints, reported side by side. | brief, F7b | P2 |
| A8 | H3 redefined: transport gap against the target cohort's own cross-validated AUROC with the same predictors. | F3 | P2 |
| A9 | H4 redefined: across-seed stability of attributions (not trained for). The v1 violation endpoints become a manipulation check. Five non-circular constraint analyses added. | F1 | P2 (H4), E (analyses) |
| A10 | H5 redefined: sensitivity at matched specificity (true-positive rate at false-positive rate 0.10 within each group), groups with at least 20 events. | F8 | P2 |
| A11 | NHANES label corrected: people taking insulin or diabetes pills, and people whose diagnosis answer is refused, unknown or missing, are excluded; FPG counts only after at least 8 hours of fasting. | F13 | P2 |
| A12 | Exact zeros in RICH ALT and AST set to missing, as already done for TG, HDL and LDL. | F17 | P2 |
| A13 | Competitors added: XGBoost and LightGBM with monotone constraints from the sign table, sign-constrained logistic regression, a spline additive logistic model, multiple imputation plus logistic regression. RBF SVM dropped (v1 record keeps it). | F20, brief | P2 |
| A14 | Censoring-aware evaluation in NAGALA and RICH (Harrell's C, Uno's C, IPCW time-dependent AUROC at 3 years; NAGALA also 5 years) replaces v1's horizon analysis that dropped censored people; 3-year IPCW-trained models added. | F15 | P2 (evaluation), E (models) |
| A15 | Calibration reported with intervals (slope, intercept, ECE) and reliability curves with bootstrap bands; decision-curve analysis; threshold metrics at fixed sensitivity and specificity. | F19, brief | P2 |
| A16 | Missingness: MCAR, MAR and MNAR each at 10%, 30%, 50%; structural-missingness evaluation in RICH; the match between augmentation and the primary test scenario is disclosed. | F10 | P2 |
| A17 | Pooled training ablation standardises with training statistics only. | F12 | P2 |
| A18 | Outputs archive never contains raw or row-level data; row-level predictions stay in a local folder. | F18 | P2 |
| A19 | Sensitivity analyses: RICH without family history; NHANES with v1 label; NHANES survey-weighted metrics. | F13, F14, F16 | E |

## 3. Data

Files (sha256 checked in cell 4): NAGALA `DatasetIntJObesity.xlsx`, RICH `RCHealthCareData-20180820.xlsx`, NHANES `nhanes_t2d_extract.csv`.

- **Task B, future risk [P1]:** NAGALA (Japan) and RICH (China, 15 sites); label = incident diabetes recorded during follow-up (binary), with `follow_years` used for the censoring-aware analyses (A14).
- **Task A, detection [P1, corrected by A11]:** NHANES adults aged 20 or more, cycles coded 1 to 10 (1999 to 2018); undiagnosed diabetes = FPG of at least 7.0 mmol/L after at least 8 hours of fasting, or HbA1c of at least 6.5%, among adults who report no diagnosis, take no insulin or diabetes pills, and answered the diagnosis question. Code -1 (probably NHANES III) is excluded as in v1.
- Cleaning [P1 + A12]: NAGALA sex code 1 = men; NAGALA HDL stored as 0 set to missing; RICH exact zeros in TG, HDL, LDL, ALT and AST set to missing. No other value is changed and no row is removed.
- Predictor tiers [P1]: T2 (headline) = age, sex, BMI, waist, SBP, DBP, smoking, alcohol, exercise, family history, TG, HDL, LDL, total cholesterol, ALT, AST, GGT, each where the cohort records it; T3 = T2 + FPG + HbA1c where recorded (Task B only, sensitivity). Feature sets are those of v1 (RICH drinking status stays unused, a stated limitation).
- Development and evaluation subsets [P1]: the v1 split (20% development, stratified, same seed). Development rows are used only for tuning and lambda; evaluation rows only for cross-validated evaluation. Both were used in v1, so neither is untouched by the investigators.

## 4. Models

Preprocessing inside each training fold [P1]: winsorise at the 0.5 and 99.5 percentiles, log1p for TG, ALT, AST and GGT, z-score; a missing value is 0 on the z-scale (the training mean) with a mask of 1 = observed.

- **M1 [P1, description corrected]:** group-encoder MLP; missing values at the training mean, no mask input, no missingness augmentation.
- **M2 [P1]:** the same network with the mask as input and training-time missingness augmentation (half of the rows; the laboratory panel removed with probability 0.3, single clinical values with probability 0.1).
- **CKINN [P1 + A5]:** M2 plus lambda times the clinical penalty (squared hinge on sign-adjusted input derivatives for continuous constrained predictors, finite differences for binary ones, observed entries only), evaluated on the whole mini-batch.
- Training [P1]: AdamW, gradient clipping 1, batch 1,024, up to 40 epochs, early stopping (patience 4) on log loss plus Brier in a 15% inner validation split, best weights restored; output bias initialised at the training prevalence.
- Pairing [A1]: within a fold, M1, M2 and CKINN use the same seed, the same inner split and the same rows, so CKINN with lambda 0 equals M2 exactly.
- Sign table [P1]: increasing risk with age, BMI, waist, SBP, family history, TG (T3 adds FPG and HbA1c); decreasing with HDL and regular exercise. Unchanged.
- Competitors [A13]: LR (L2); sign-constrained LR (same table, coefficients bounded on the standardised scale); spline additive logistic model (cubic B-splines, 5 knots per continuous predictor, missing indicators, L2); random forest; XGBoost; XGBoost with `monotone_constraints` from the sign table; LightGBM; LightGBM with monotone constraints; multiple imputation (5 chained-equation imputations with posterior draws, fitted on the training fold) plus LR, averaging the 5 predicted probabilities; LR and XGBoost trained with the same missingness augmentation (LR_aug, XGB_aug). The last three are fitted in repeat 0 only.

## 5. Tuning [A4]

On the development subset (capped at 20,000 rows, stratified), score = log loss plus Brier [P1].
- M1 and M2: 32 configurations each (the v1 default and 31 drawn without replacement from the v1 space of 72), 3-fold CV. M2's final configuration is the best of its 32.
- CKINN: M2's best configuration among its first 24, then lambda in {0.03, 0.1, 0.3, 1, 3, 10, 30, 100}: 24 + 8 = 32 evaluations, the same budget as M1 and M2. Lambda is scored, together with lambda 0, on 3-fold CV repeated twice. **Rule:** take the configuration with the lowest mean score among lambda 0 and the grid; choose the largest positive lambda whose mean score is within one standard error (over the 6 fits) of that minimum; if none qualifies, choose 0.03.
- Competitors: LR, constrained LR and the additive model 4 values of C each; random forest the 9 v1 combinations; XGBoost, monotone XGBoost, LightGBM and monotone LightGBM 16 configurations each; 3-fold CV. MI plus LR uses LR's C; the augmented models use their base model's configuration.
- Tier T3 reuses the T2 configurations; lambda is re-selected for T3 by the same rule.

## 6. Evaluation design

- Repeated cross-validation [A2]: 5 repeats of stratified 5-fold CV on the evaluation subset of each cohort. In repeat r every network uses seed r ("main runs"); every model in a repeat sees the same folds. Seed study: in repeat 0, M1, M2 and CKINN are also trained with seeds 1 to 4.
- Statistic per run: the metric of the pooled out-of-fold predictions of that run. Point estimate: the mean over the 5 main runs.
- Primary inference [A6]: paired participant bootstrap, 2,000 draws; in each draw the same participant weights are applied to all runs and models and the run-averaged difference is recomputed. 95% percentile intervals; one-sided p = (1 + number of draws on the null side) / (2,001).
- Cross-checks [A6, E]: DeLong paired test on run 0; corrected resampled t-test over the 25 folds (variance multiplied by 1/25 + n_test/n_train); RICH cluster bootstrap over the 15 sites; TOST equivalence with margin 0.01 for AUROC.
- Variance components [P2]: bootstrap SE of one run (sampling), SD across the 5 seeds of repeat 0 (seed), SD across the 5 main runs (split plus seed).
- Minimum detectable effect [P2]: for NAGALA, (1.96 + 0.84) times the bootstrap SE of the CKINN minus M2 AUROC difference, and the power of the non-inferiority test if the true difference were 0.

## 7. Hypotheses, endpoints and decision rules

Two families: **A, CKINN minus M2** (primary) and **B, CKINN minus M1**. Within each family, Holm's correction is applied across all its primary endpoints (22 in each when every cohort has at least two age bands with 20 or more events; otherwise the age-band endpoint of that cohort is dropped and the count falls accordingly). An endpoint is "predicted" if its Holm-adjusted one-sided p in the predicted direction is below 0.025, "opposite" if the Holm-adjusted p in the opposite direction is below 0.025, otherwise "null". A unit is predicted if all its primary endpoints are predicted, opposite if any is opposite. **PASS**: no unit opposite and a majority of units predicted (H1: all units); **FAIL**: a majority opposite; otherwise **NULL** [P1 rule].

Units: NAGALA, RICH and NHANES at tier T2 (H1, H2, H4, H5); NAGALA to RICH and RICH to NAGALA (H3).

- **H1 discrimination [P1]:** AUROC difference; non-inferiority with margin 0.01 (null: difference at most -0.01; opposite: difference below -0.01 with the upper bound). Superiority and TOST are reported, not tested.
- **H2 calibration [P1]:** Brier difference (lower predicted) and difference in |calibration slope - 1| (lower predicted).
- **H3 transport [A8]:** models trained on all evaluation rows of the source with the 10 shared predictors and applied unchanged to the target; reference = 5-fold CV inside the target with the same predictors and model class (repeat 0). Both averaged over network seeds 0 to 4. Endpoints: transport gap = reference AUROC minus external AUROC (lower predicted), and |calibration-in-the-large| in the target (lower predicted). The bootstrap resamples target participants.
- **H4 explanation stability [A9]:** in repeat 0, 1,000 held-out rows per fold are explained by integrated gradients (zero baseline on the z-scale, 32 steps) for each of the 5 seeds; per row, the mean Spearman correlation of the attribution vectors over the 10 seed pairs; endpoint = mean over rows (higher predicted). The bootstrap resamples explained rows.
- **H5 subgroup sensitivity [A10]:** true-positive rate at false-positive rate 0.10 within each group; endpoints = absolute sex difference and range across age bands (20 to 39, 40 to 59, 60 and over) with at least 20 events (lower predicted).

Manipulation check (not in a family): in-distribution violation rate and direction consistency, as in v1.

## 8. Secondary and exploratory analyses

- **CKINN against each competitor [P2, descriptive]:** AUROC and Brier differences with bootstrap intervals, no multiplicity correction.
- **Calibration [A15]:** slope, calibration-in-the-large and ECE (10 equal-frequency bins) with 500-draw intervals; reliability curves with 95% bands for M1, M2, CKINN, LR and monotone XGBoost.
- **Clinical utility [A15]:** net benefit over thresholds 0.5% to 10% (Task B) and 1% to 20% (Task A), with intervals at 2% and 5% (Task B) and 5% and 10% (Task A); sensitivity, specificity, PPV and NPV at fixed sensitivity 0.80 and fixed specificity 0.90, thresholds taken from the inner validation split of each training fold.
- **Censoring-aware [A14]:** Harrell's C, Uno's C truncated at 3 years and IPCW cumulative/dynamic AUROC at 3 years (NAGALA also 5 years) for every model's out-of-fold predictions (P2); IPCW 3-year models (event by 3 years = 1, follow-up beyond 3 years without event = 0, censored earlier = weight 0, others weighted by the inverse Kaplan-Meier censoring probability from the training fold) for LR, monotone XGBoost, M2 and CKINN in repeat 0, with IPCW Brier and observed-versus-predicted 3-year risk (E).
- **Non-circular constraint evidence [A9, E]**, repeat 0, seed 0: (a) leave one constraint out; (b) wrong-sign controls (age flipped, BMI flipped, all signs flipped) at the selected lambda; (c) sign of each constrained predictor in an adjusted logistic regression on the evaluation rows (descriptive) and in quintile event rates, against the held-out partial-dependence slope of M2 and CKINN; (d) violation rate on the other cohort and in sparse regions (held-out rows beyond the training 95th percentile of Mahalanobis distance); (e) sign-table sensitivity: a core table (age, BMI, waist, SBP, TG) and an extended table (v1 plus DBP, ALT, GGT); and the lambda path over the grid.
- **Transport, secondary [E]:** recalibration in the small with 1,000 target participants (target labels used); label-free importance reweighting (domain classifier on the shared predictors, weights clipped to 0.1 to 10 and scaled to mean 1); pooled NAGALA plus RICH training with training-only standardisation [A17]; each cohort's training rows are capped at 20,000 (stratified), as in v1, so that RICH does not swamp NAGALA; RICH's tuned configuration and lambda are used.
- **Fairness [A10, E]:** by sex, age band, RICH site and NHANES ethnic group (last two descriptive): AUROC, sensitivity at matched specificity, true- and false-positive rates at the common threshold (equalised-odds gaps), calibration-in-the-large and slope, 500-draw intervals; a group with fewer than 20 events is not estimable.
- **Missingness [A16]:** test-time removal of the laboratory panel by MCAR, MAR and MNAR at 10%, 30% and 50%; AUROC loss at MCAR 30% for CKINN against each comparator; RICH performance by observed missingness pattern and leave-one-site-out (sites with at least 100 events).
- **Sensitivity [A19, E]:** tier T3 (repeat 0); RICH without family history (repeat 0); NHANES with the v1 label (repeat 0); NHANES survey-weighted metrics with combined MEC weights (strata-PSU bootstrap); seed variance table.

## 9. Runtime modes and freeze

`CKINN_REAL_MODE=full` is the reported analysis. `smoke` runs the same code on stratified subsamples (NAGALA 4,000, RICH 20,000, NHANES 6,000 rows; 2 repeats of 3-fold; 2 seeds; 3 tuning configurations; 6 epochs; 200 draws) into a separate folder; smoke numbers test the code and are never interpreted.

Configuration hash (sha256 of the canonical JSON of the full-mode configuration, sign tables, model lists, search spaces and hypotheses; computed by `ckv2.config.config_hash` and checked by notebook cell 16): `7cdc5bdde12b17df19a15650f00f1fe693168b33f39b193219ded6d70ea0468b`

## 10. Changes after the freeze

- 2026-11-20, while the full run was in progress and before any full-mode result had been seen. Clarification, not a change: the fairness-weighted CKINN of v1 (`CKINN_fair`) is not part of v2; the amendment table omitted it by oversight (it should appear beside the RBF SVM in A13). Its v1 results remain in the v1 record. The configuration hash is unaffected.
