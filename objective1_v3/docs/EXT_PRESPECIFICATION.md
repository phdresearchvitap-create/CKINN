# Post hoc extension analyses (label EXT): rules fixed before any EXT result

Written 2026-11-23, after the STANDARD run 96ccd0057ac85891 had been seen (stages 5 to 28) and after two diagnostics
(cohort timing and M1 gradient signs, recorded in `OBJ1_REVIEW.md`), and before any analysis below was computed.
None of these analyses changes the primary endpoint, the primary comparison (Brier(M3) - Brier(M1) in NAGALA),
the eligibility rules or the frozen models of the v2 protocol. They are exploratory and are reported in full,
whatever they show. The code is `ext/ckinn_ext.py`; it sits outside `study/`, so it does not change the run identity.

Common rules. Task A only. Development data: the eligible RICH cohort (204,640). Evaluation: the eligible NAGALA
cohort (15,459). IPCW, Brier, AUROC, calibration and the paired bootstrap are the package's own functions
(`stats.weights`, `stats.metrics`, `stats.bootstrap`). Bootstrap draws: 1,000, paired, G re-estimated in each draw.
Intervals are conditional on the fitted models. No hypothesis test is added; no result is used to select a model.

E1. Frozen-model reconstruction. Refit every frozen model from `source_selection.csv` (settings, epochs, seed 1729)
on all eligible RICH rows and score NAGALA. Pass if |difference| < 1e-6 in Brier and AUROC against
`external_metrics.csv` of run 96ccd0057ac85891 for every model.

E2. Penalty diagnostics. For M1, M3, fixed_zero, fixed_positive and expanded_prior: share of eligible
participant-feature pairs with a wrong-way probability gradient, for FPG, BMI, age, SBP, TG and HDL, in RICH and
NAGALA; the clinical penalty value against the weighted data loss.

E3. Baselines. Paired NAGALA Brier differences M1 - X and M3 - X for X in {reference, LR, RF, XGB, LGB, SVM, Cox}.

E4. Index of prediction accuracy (scaled Brier) 1 - Brier(model)/Brier(reference) for every frozen model, with
paired bootstrap intervals.

E5. Net benefit with IPCW at risk thresholds 0.5%, 1%, 2%, 3% and 5% for M1, M3, LR, treat all and treat none.

E6. Estimand-aligned NAGALA. RICH contains only people followed for at least 2 years, so its three-year cases are
detections in years 2 to 3. Apply the same condition to NAGALA (follow-up of at least 2 years; case = detection by
3 years; control = follow-up beyond 3 years; others unknown; G re-estimated in the subset) and score the frozen
models. Descriptive.

E7. HDL availability. Score the frozen models on NAGALA with HDL set to missing for everyone (the model's
missingness indicator then switches on), to show how much the predictions depend on the HDL input that is missing
for 44.9% of RICH.

E8. Finite-step direction checks (package rules: +0.1 mmol/L FPG, +1 kg/m2 BMI, +1 year, +5 mmHg SBP, +0.1 mmol/L
TG or HDL; 512 NAGALA records, outcome-blind seed 1748) for M1, M3, fixed_positive and expanded_prior.

E9. Penalty-strength path for the six expanded directions. M3 frozen settings and epochs, directions
{FPG+, BMI+, age+, SBP+, TG+, HDL-}, lambda in {0, 1, 10, 100, 1e3, 1e4, 1e5}. Each lambda is scored by the
inner-validation IPCW Brier on the existing final inner splits of RICH; the selected lambda is the lowest score,
ties within 1e-9 going to the smaller lambda. Every lambda is refitted on all RICH rows and reported in NAGALA
(Brier, AUROC, calibration, violation shares); only the selected one is singled out.

E10. Training-size curve. Stratified RICH subsamples of 1,000, 3,000, 10,000 and 30,000 participants, 5 draws
each (seeds 1 to 5 per size). In each draw: M1 (lambda 0) and two penalised networks (primary directions FPG, BMI;
expanded six directions), all with the M3 frozen architecture, early stopping on a 2-fold inner split (patience
10, at most 100 epochs) and refitting on the whole subsample; lambda for each penalised network chosen by the same
inner split from {10, 1e3, 1e5}. Report NAGALA Brier and AUROC for each arm and the paired differences
(penalised minus M1) per draw, and the mean over draws per size. No draw is discarded.

E11. Site support. Refit the frozen M1 and M3 settings on RICH without sites 14 and 16 (no participant observed
beyond 3 years) and score NAGALA.

## Addendum (written after E9 had been seen, before E9b was computed)

E9b. Paired NAGALA Brier difference of every lambda on the E9 path against lambda 0 of the same path (same frozen
architecture, epochs, seed and data), 1,000 paired bootstrap draws with G re-estimated, percentile intervals. Added
because E9 showed lower NAGALA error at large lambda without an interval. Descriptive; it does not change the
pre-specified E9 selection rule, which chose lambda by RICH inner-validation error.

## Addendum 2 (written 2026-11-23 before E12 or the proposal table was computed)

E12. Calibration contrast for H2. In NAGALA, paired bootstrap (1,000 draws, G re-estimated) of
|calibration intercept|, |calibration slope - 1| and ECE, M3 minus M1. Negative favours M3. Descriptive.

Proposal table. A generated table maps RQ1 to RQ7 and H1 to H5 of the proposal to the computed estimates. Only H1 has a
decision rule (the pre-specified primary contrast: supported if the whole 95% interval of Brier(M3) - Brier(M1) is
below zero, contradicted if it is above zero, otherwise inconclusive). Every other row is descriptive. Items the data
cannot address are listed as not measurable.
