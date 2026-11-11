# CKINN protocol v0.1 (DRAFT, not locked)

Status: draft for supervisor review. Lock date: ____. After locking, any change is logged as a deviation with a reason.

## 1. Aim and claims
Test whether clinical-knowledge constraints improve the generalization, calibration and explanation consistency of neural networks for incident type 2 diabetes (T2D) risk prediction, using two real cohorts.

- C1 Discrimination: CKINN is not inferior to tuned baselines.
- C2 Generalization: CKINN transports better across sites and across cohorts.
- C3 Calibration: CKINN is better calibrated under shift, before and after recalibration.
- C4 Consistency: CKINN's risk surface and attributions violate clinical priors less often and are more stable than an unconstrained MLP's.
- Secondary (exploratory): subgroup gaps by sex and age band; robustness to missing predictors.

## 2. Data (real only)
| Cohort | n | Events | Follow-up | Role |
| --- | --- | --- | --- | --- |
| RICH | 211,833 (15 sites) | 4,174 | mean 3.12 y | development; leave-one-site-out; external for NAGALA-trained models |
| NAGALA | 15,464 (1 centre) | 373 | mean 6.04 y | external for RICH-trained models; development for cohort-specific secondary models |

Population: adults without diabetes at baseline (baseline FPG below 7.0 mmol/L in both files). No synthetic, simulated or augmented data at any stage. Both datasets are public and de-identified; confirm licences and cite the original cohort papers from the data records.

## 3. Outcome and time
- RICH event: `censor of diabetes at followup` = 1, which is diagnosis during follow-up (1,304) or final-visit FPG of at least 7.0 mmol/L. Verified: all 4,174 events are explained by these two criteria. Sensitivity analysis: diagnosis-only.
- NAGALA event: `Incident DM`.
- Time: RICH `year of followup`; NAGALA `Follow up duration, days` / 365.25. Confirm in the source paper that RICH time is time-to-event for cases. Events are detected at visits, so treat time as interval-censored in a sensitivity analysis.
- Label wording: "incident diabetes, diagnosis- or biomarker-defined", not "diagnosed diabetes".

## 4. Predictor tiers
| Tier | Variables | RICH complete cases (events) | NAGALA complete cases (events) |
| --- | --- | --- | --- |
| T1 low-cost | age, sex, BMI, SBP, DBP | 211,809 (4,173) | 15,464 (373) |
| T2 | T1 + total cholesterol, triglycerides, ALT | 205,291 (4,071) | 15,464 (373) |
| T3 | T2 + FPG | 205,291 (4,071) | 15,464 (373) |
| T4 | T3 + HDL, AST | 49,695 (1,275) | 15,464 (373) |

Cohort-specific extras, used only in within-cohort secondary models: RICH family history and smoking; NAGALA HbA1c, waist circumference, exercise, fatty liver, GGT. HbA1c is never in a cross-cohort model.

## 5. Missing data (pre-specified)
Primary: gradient-boosting models use native missing handling; MLP, CKINN and linear models use training-fold median or mode plus missingness indicators. Sensitivity: multiple imputation (MICE) fitted inside training folds only; complete cases. Test data are never used to fit imputers.

## 6. Models
1. Penalized Cox and discrete-time logistic regression
2. Random survival forest or gradient-boosted survival model
3. Monotone-constrained gradient boosting (the baseline reviewers will demand)
4. Neural discrete-time hazard MLP (unconstrained)
5. CKINN: the MLP plus a soft constraint penalty on the predicted 3-year risk
6. CKINN with missingness indicators; every model also evaluated after recalibration on a small target sample (intercept and slope only)

Primary prediction target: absolute risk at 3 years (RICH median follow-up is 2.99 y). NAGALA-only secondary analysis at 5 years. Binary 3-year labels with inverse-probability-of-censoring weights are a secondary check.

```
L = L_discrete-time-NLL + lambda * sum_j w_j * mean_x [ max(0, -s_j * d risk_3y(x) / d x_j) ^ 2 ]
```
lambda = 0 recovers the plain MLP. Tune lambda on validation folds only and report the full sweep.

## 7. Clinical constraints (expected direction of risk)
| Variable | Direction | Available in | Note |
| --- | --- | --- | --- |
| BMI | up | both | |
| FPG | up | both | tier T3 and above |
| Triglycerides | up | both | |
| SBP | up | both | proxy for hypertension |
| Age | up | both | soft, low weight: NAGALA risk plateaus after the fourth age quintile |
| HDL | down | both | tier T4 |
| Family history | up | RICH only | |
| HbA1c | up | NAGALA only | |
| Exercise | down | NAGALA only | |

Sex, ALT, AST and cholesterol are left unconstrained. All constraints are soft; none is applied as a hard rule.

## 8. Evaluation regimes (all person-level, real data)
- R1 Internal: repeated 5-fold stratified CV (5 repeats).
- R2 Cross-site: leave-one-site-out in RICH. Sites 0 and 16 (61 and 247 people) are pooled and never used as a test fold.
- R3 Cross-cohort: RICH to NAGALA and NAGALA to RICH, using the shared features of each tier. External data are never used for tuning.
- R4 Real covariate shift inside the data: train on age below 45, test on 45 and above; train on BMI below 25, test on 25 and above.
- R5 Natural missingness: performance by site and by observed pattern of HDL and AST; train on T4, deploy on T1 to T3 only.

## 9. Metrics
Time-dependent AUROC and C-index; IPCW Brier and scaled Brier; 3-year calibration (slope, observed-to-expected ratio, decile plot); decision-curve net benefit at 1% to 10% thresholds; constraint-violation rate over the observed feature range; attribution stability across seeds and bootstrap resamples; the same metrics by sex, age band (under 40, 40 to 54, 55 and over) and site. Uncertainty: paired bootstrap, 500 or more resamples, 95% CIs. Report the Riley sample-size calculation for the development models.

## 10. Sensitivity analyses
Tier T1 and T2 (no glucose); diagnosis-only outcome in RICH; discrete-time interval-censored model; MICE versus indicators; excluding events in the first year; lambda sweep.

## 11. Pre-specified decision rules
- C2 and C3 are supported only if the paired-bootstrap 95% CI for CKINN minus the best of {MLP, monotone GBM} excludes zero on cross-cohort calibration slope distance from 1 or scaled Brier, in both transport directions.
- C1 uses a non-inferiority margin on the C-index (proposed 0.01; decision D4).
- If no claim is supported, the paper is reframed as a rigorous negative or neutral transportability evaluation; no claim is added after seeing results.

## 12. Reproducibility
Five seeds per experiment, config files, run logs, dataset hashes, code version recorded; code and configs released with the paper.

## 13. Out of scope
Synthetic data, knowledge graphs, glucose-insulin physiological models, fairness beyond sex and age (no ethnicity variable), clinical decision-support deployment.

## 14. Decisions needed
- D1 Headline tier. Recommended: report T2 and T3 equally, with T2 as the "early risk" headline.
- D2 Confirm NAGALA sex coding (data are 0/1; male = 1 inferred from incidence) against the source record.
- D3 Thesis wording with your supervisor. Both cohorts exclude baseline diabetes, so they support incident (future) risk, not detection of existing undiagnosed diabetes. Decide whether OBJ-1 is framed as incident risk with low-cost tiers T1 and T2 standing in for "labs unavailable".
- D4 Non-inferiority margin for the C-index.
- D5 Whether a third cohort of different ancestry (for example an Indian cohort) can be obtained; if yes, it becomes the primary external validation.
- D6 Target journal.
