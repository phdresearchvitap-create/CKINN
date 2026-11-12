# Amendment A1 to CKINN protocol v1.0 (PROPOSED, not yet approved)

Drafted 2026-11-11 with `notebooks/MASTER_PLAN.md` version 2. The locked file `protocol_v1.0_LOCKED.md` is unchanged and remains the confirmatory plan.

Purpose: to give every requirement of the 26-page proposal an analysis, without altering the pre-specified study.

## Rules that apply to every item

1. Every item is exploratory. None can change, add to or replace the decision rules of protocol section 11, and none enters a C1 to C4 verdict.
2. An item takes effect only when the investigator records approval for it in notebook cell 3. Until then its cells refuse to run.
3. No hyperparameter of an added model is tuned on test data; the values below are fixed.
4. Each item is reported in full whatever it shows, with 95% intervals and no p-value-only statements. No multiplicity adjustment is applied; the results are labelled exploratory for that reason.
5. Real data only. A branch that lacks data stays blocked.

State of knowledge when drafted: the R1 part files for Cox and XGB existed (produced before deviation D-3 and superseded). No MLP or CKINN, sensitivity, leave-one-site-out, shift or cross-cohort result is in the repository. The investigator must confirm, before approving A1.3, whether cell 9 of the earlier iPad notebook (R1 for MLP and CKINN) had been run and its output seen; if so, A1.3 is recorded as specified after results.

This amendment consolidates and replaces the list in deviation D-4.

## Items

**A1.1 Additional metrics.** Censoring-weighted average precision (AUPRC) at 3 years. Calibration intercept: complementary log-log of the Kaplan-Meier observed 3-year risk minus that of the mean predicted risk. Censoring-weighted sensitivity, specificity, PPV, NPV, F1, accuracy, false-negative and false-positive rates at predicted 3-year risk of 2% and 5% and at the top tenth of predicted risk (cut-off from training data). Violation magnitude: mean size of the change in log hazard against the prior after +0.25 SD.

**A1.2 Additional comparators.** Tiers T2 and T3; R1 repeat 0 and both R3 directions. They are not part of the "best baseline" of claim C1.
- Random survival forest: 200 trees, minimum leaf 50, each tree on 20,000 sampled training rows. If it cannot run on the available machine, that is reported; nothing is substituted.
- Classifiers on the 3-year endpoint: an event by 3 years is 1; follow-up beyond 3 years without an event is 0; participants censored before 3 years have weight 0; the others are weighted by the inverse of the Kaplan-Meier censoring probability estimated in the training fold. Logistic regression (L2, C = 1); random forest (300 trees, minimum leaf 50); XGBoost (300 trees, depth 3, rate 0.05); LightGBM (300 trees, 8 leaves, rate 0.05); SVM, linear (C = 1) and radial kernel by Nystroem approximation (300 components), with weighted Platt scaling on an inner 20% split.

**A1.3 H1 reported separately from C1.** H1 (CKINN better than MLP) is judged on the Nadeau-Bengio 95% interval for CKINN minus MLP in C-index, R1, tier T2: supported if the interval lies above zero, otherwise not supported. Scaled Brier is reported beside it.

**A1.4 Prior ablations.** R1 repeat 0, tiers T2 and T3: MLP plus one prior at a time; CKINN minus one prior at a time; the lambda path 0, 0.1, 1, 10. Single-prior fits use the lambda selected for the full CKINN in the same fold.

**A1.5 Extended extras E4.** E3 plus LDL (RICH) and alcohol (RICH drinking status; NAGALA ethanol in grams per week). Within cohort, one repeat of 5-fold, no prior on either.

**A1.6 Recalibration audit.** Before the locked recalibration is fitted, the number of events in the 1,000-participant sample and the standard error of the slope are reported. The locked procedure itself is unchanged. Recalibrated results are tabulated apart from unchanged external predictions.

**A1.7 Cohort-comparability sensitivity.** (a) Two-year landmark: among target participants still at risk at 2 years, predicted risk is 1 - exp(-(H0(3) - H0(2)) exp(lp)). (b) Both R3 directions at T2 and T3 with RICH restricted to baseline fasting glucose below 6.1 mmol/L.

**A1.8 Explainability.** Integrated Gradients (baseline: training medians; 64 steps) on the log hazard and on 3-year risk; SHAP on 3-year risk (200 background rows, 1,000 participants); partial-dependence, individual and accumulated-local-effect curves; what-if changes of one variable (BMI, SBP, triglycerides, glucose) within the 1st to 99th percentile of the same sex and age band; fidelity, fold stability and cross-cohort consistency as defined in the master plan, section 9; a fixed patient-level report template.

**A1.9 Clinical expert review (gated).** Sixty cases by two models, blinded, five-item rubric, at least two independent clinicians, weighted kappa. Nothing is reported without real ratings.

**A1.10 Subgroup audit.** A1.1 metrics and protocol section 9 metrics by sex, age band (under 40, 40 to 54, 55 and over), RICH site and cohort, with bootstrap intervals. A group with fewer than 20 events by 3 years is reported as not estimable.

**A1.11 Fairness-aware CKINN.** Loss = Cox loss + lambda times clinical penalty + mu times F, where F is the variance of the mean Cox loss across sex groups plus the variance across age bands, each group's loss using the full-batch risk set. Lambda is the value selected for CKINN in that fold; mu is chosen from 0.1, 1, 10 by the rule of protocol section 6. Tier T2; R1 repeat 0 and both R3 directions.

**A1.12 Population alignment.** Training rows are weighted by the estimated odds of belonging to the target cohort, from an L2 logistic regression on the tier's predictors only, clipped to 0.1 to 10 and scaled to mean 1; the Cox loss and the Breslow hazard are weighted. No target outcome is used. Tiers T2 and T3, both R3 directions. Effective sample size is reported.

**A1.13 Knowledge-graph CKINN.** Nodes and edges follow the diagram on proposal pages 24 to 25, restricted to measured features. Each edge needs a reference verified in PubMed or Scite; an unverified edge is not used. The network's connections follow the graph and the clinical penalty is unchanged. Latent nodes are learned summaries, not measurements of insulin resistance. Tiers T2 and T3; R1 repeat 0 and both R3 directions.

**A1.14 Glycaemic strata and progression.** (a) Metrics from saved predictions among participants with baseline fasting glucose below 5.6 mmol/L (in NAGALA also HbA1c below 5.7%), and among those in the prediabetes range. (b) In RICH only, among participants below 5.6 mmol/L at baseline: status at the final visit (fasting glucose of 5.6 mmol/L or more, or diabetes) modelled with a complementary log-log link and log follow-up time as offset; linear model, MLP and CKINN; tier T2 without glucose; one repeat of 5-fold. The time of transition is not observed and no external validation is possible.

**A1.15 Gated branches.** Third cohort (including pooled training on RICH and NAGALA); multistate model with transition times; glucose-insulin differential-equation model; fairness by ethnicity or socio-economic group; current-diabetes classification (not identifiable: no prevalent cases). Each has a readiness check and no analysis.

## Approval record

| Item | Approved by investigator (date) | Approved by supervisor (date) |
| --- | --- | --- |
| A1.1 to A1.15 | requested 2026-11-12 for full proposal coverage; recorded with its date when notebook cell 3 runs (`ckinn_results/audit/amendment_A1_approvals.json`) | pending |

An item can be withdrawn before its cells run by setting it to False in notebook cell 3.
