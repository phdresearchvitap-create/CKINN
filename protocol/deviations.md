# Deviations from protocol v1.0

None recorded at lock time. Add one entry per deviation: date, what changed, why, and whether results had been seen.

## D-1 (2026-11-11, before any LOSO result was inspected)
Site 14 of RICH has maximum follow-up 2.92 years, shorter than the 3-year horizon, so no 3-year metric can be computed when it is the test site. It stays in every training set and is not a test fold, leaving 12 test sites instead of the 13 stated in section 8 of the locked protocol. Effect on claims: none (C2 and C3 rest on the cross-cohort analyses; LOSO is descriptive).

## D-2 (2026-11-11, before any R1, LOSO, shift or cross-cohort result was inspected)
The kit's `scripts/run_experiments.py` dropped every row with a missing predictor in all analyses (complete-case), which for tier T4 in RICH keeps 49,695 of 211,833 participants and contradicts section 5 of the locked protocol (training-fold median fill plus missingness indicators; complete-case is sensitivity analysis S2 only). The code was corrected to use all rows, and S2 was added as its own unit (`sensS2`, RICH T4 complete cases, 1 repeat of 5-fold). This is a correction of the code to the protocol, not a change to the protocol. Effect on claims: none; the analysis now matches what was pre-specified.

## D-3 (2026-11-11, adopted with `notebooks/MASTER_PLAN.md`)
Exact zeros in triglycerides, HDL cholesterol, ALT and AST are treated as not measured and set to missing in both cohorts. Counts in the raw files: RICH triglycerides 860 (637 at site 2), HDL 102, ALT 2, AST 1; NAGALA HDL 11, all with the mg/dL column blank. No row is removed and no other value is altered. Reason: zero is not a possible result for these assays and its clustering by site indicates a code for "not measured"; left in, it is read as the lowest value in the cohort. The rule is applied in notebook cell 9 and must be added to `scripts/00_harmonize.py` before `make data` is used again. Results seen: the R1 part files for Cox and XGB (`results/parts/cv_*`) existed and had been produced with the zeros retained; they are superseded and are re-run. No neural, sensitivity, leave-one-site-out, shift or cross-cohort result existed. Effect on claims: the decision rules are unchanged; tier T2 inputs change for 0.4% of RICH participants.

## D-4 (2026-11-11, adopted with `notebooks/MASTER_PLAN.md`, before any of these analyses was run)
Seven exploratory analyses are added because the proposal asks for them or because the data audit showed the cohorts differ in design. None can support claims C1 to C4, and all are reported in full whatever they show.
1. Censoring-adjusted sensitivity, specificity, PPV, NPV, F1 and false-negative rate at 3 years, at predicted 3-year risk of 2% and 5% and at the top tenth of predicted risk (cut-off taken from training data), overall and by sex and age band.
2. Partial-dependence and individual curves for the constrained variables; Integrated Gradients attributions on the log hazard (baseline: training medians; SHAP as a cross-check); rank stability across folds and cohorts.
3. Single-variable what-if changes within the observed range for example participants, worded as model behaviour.
4. Performance and violation rate along the lambda path, and CKINN refitted with one prior removed at a time (T2, one repeat of 5-fold).
5. Two-year landmark: RICH-trained 3-year risks evaluated in NAGALA participants still at risk at 2 years. Reason: RICH has no follow-up under 2 years and no event before 2.0 years; NAGALA has 61 events before 2 years.
6. Cross-cohort analysis at T2 and T3 with RICH restricted to baseline fasting glucose below 6.1 mmol/L. Reason: NAGALA baseline glucose stops at 6.05 mmol/L; RICH extends to 6.99.
7. Metrics among participants with baseline fasting glucose below 5.6 mmol/L, from saved predictions.
Results seen: as in D-3. Effect on claims: none.

## D-5 (2026-11-11, before any analysis named in D-4 was run)
The list of exploratory analyses in D-4 is replaced by `protocol/amendment_A1_PROPOSED.md`, which extends it so that every requirement of the proposal has an analysis (register: `protocol/proposal_coverage.csv`; plan: `notebooks/MASTER_PLAN.md` version 2, 88 cells). D-4 said "adopted"; that is corrected here: every item of A1 is proposed and takes effect only when the investigator records approval. None can enter a C1 to C4 verdict. Results seen: as in D-3. One statement in the earlier 44-cell plan was wrong and is withdrawn: RICH does record a later glycaemic state (fasting glucose at the final visit), so a one-step progression analysis is possible in RICH (A1.14).
