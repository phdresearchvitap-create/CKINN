# CKINN protocol v1.0 (LOCKED)

Locked: 2026-11-11, before any analysis in which a model was trained on one cohort and tested on the other, and before the full-scale runs of any analysis below. Any later change is recorded in `protocol/deviations.md` with a reason. The SHA-256 of this file is stored in `protocol/LOCK.sha256` and checked by the cross-cohort scripts.

## 0. Prior exposure (disclosed)
Before locking, the investigators and the assistant had (i) profiled both cohorts descriptively (event rates by quintile, missingness, follow-up), (ii) run small code checks of the pipeline on a 40,000-row RICH subsample, and (iii) timed single full-size model fits on RICH. No model was trained on NAGALA and tested on RICH or vice versa. The descriptive NAGALA profile showed that risk plateaus at higher age, which is why the age prior is soft and low-weight.

## 1. Aim and claims
Test whether soft clinical-knowledge constraints improve generalization, calibration and constraint-consistency of neural survival models for incident type 2 diabetes (T2D) in two real cohorts.

- C1 Discrimination (non-inferiority): CKINN is not inferior to the best tuned baseline.
- C2 Generalization: CKINN transports better across cohorts than its comparators.
- C3 Calibration: CKINN is better calibrated under transport.
- C4 Consistency: CKINN violates the clinical priors less often than the unconstrained MLP.
- Secondary, exploratory: subgroup performance by sex and age band; behaviour under missing predictors; cohort-specific predictors.

## 2. Data (real only)
RICH: Chen et al., BMJ Open 2018 (Rich Healthcare Group, China; Dryad doi 10.5061/dryad.ft8750v); 211,833 adults, 15 site codes in the file. NAGALA: Okamura et al., Int J Obes 2019 (Murakami Memorial Hospital, Gifu, Japan; Dryad doi 10.5061/dryad.8q0p192); 15,464 adults. Raw-file SHA-256 values are in `data/raw/CHECKSUMS.sha256`. No synthetic, simulated or augmented data are used at any stage. Unit tests use hand-made toy arrays only and produce no results.

## 3. Outcome and time
- RICH: `censor of diabetes at followup` = 1 (fasting plasma glucose of at least 7.00 mmol/L or self-reported diabetes; follow-up ends at diagnosis or final visit). Verified in the file: all 4,174 events are explained by diagnosis in follow-up (1,304) or final-visit FPG at least 7.0 mmol/L.
- NAGALA: `Incident DM`, as defined in the source study.
- Time in years: RICH `year of followup`; NAGALA `Follow up duration, days` / 365.25.
- Label wording: "incident diabetes, diagnosis- or biomarker-defined".

## 4. Predictor tiers
| Tier | Variables |
| --- | --- |
| T1 | age, sex, BMI, SBP, DBP |
| T2 | T1 + total cholesterol, triglycerides, ALT |
| T3 | T2 + FPG |
| T4 | T3 + HDL, AST (RICH: heavy missingness) |

Sex: male = 1 in both cohorts. The NAGALA column is coded 0/1; its counts (8,430 and 7,034) match the 8,430 men and 7,034 women reported by the source study, so 1 = male. A no-sex sensitivity analysis is still run (section 10).
Cohort-specific extras (secondary, within-cohort only): RICH family history and smoking; NAGALA HbA1c, waist circumference, exercise, fatty liver, GGT. HbA1c is never used across cohorts.

## 5. Missing data
Gradient boosting: native handling. Linear Cox, MLP and CKINN: training-fold median or mode plus missingness indicators. Test data never inform imputation. Complete-case analysis is a sensitivity analysis only.

## 6. Models (all produce absolute 3-year risk)
1. Cox regression, ridge penalty 0.01 (`Cox`)
2. Gradient-boosted Cox, unconstrained (`XGB`)
3. Gradient-boosted Cox with monotone constraints from section 7 (`XGB-mono`)
4. MLP with Cox partial likelihood, hidden layers (64, 64), no constraint (`MLP`)
5. CKINN: the same MLP plus the constraint penalty (`CKINN`)

Risk at horizon h = 3 years is 1 - exp(-H0(h) exp(lp)) with a Breslow baseline estimated on the training data of each analysis. The 3-year horizon is fixed: RICH median follow-up is 2.99 years. Penalty:

```
L = L_Cox + lambda * sum_j mean_x [ max(0, -s_j * d lp(x) / d x_j) ^ 2 ]
```
Derivatives are taken with respect to standardized inputs for the variables in section 7. lambda is chosen inside the training data of every analysis (inner 80/20 split, fixed seed): fit lambda in {0, 0.1, 1, 10}; choose the largest positive lambda whose inner scaled Brier is within 0.002 of the best value over all four; if none qualifies, lambda = 0.1. Training: AdamW, learning rate 2e-3, weight decay 1e-4, dropout 0.1, batch 8192, up to 40 epochs with early stopping (patience 4) on a 10% validation split. No other hyperparameter is tuned.

## 7. Clinical constraints
| Variable | Expected direction | Cohorts |
| --- | --- | --- |
| BMI | up | both |
| FPG | up | both (T3, T4) |
| Triglycerides | up | both |
| SBP | up | both |
| Age | up (soft) | both |
| HDL | down | both (T4) |
| Family history | up | RICH extras |
| HbA1c | up | NAGALA extras |
| Exercise | down | NAGALA extras |

Sex, ALT, AST, cholesterol, DBP, waist, fatty liver, GGT, smoking are unconstrained. All constraints enter as soft penalties with equal weight; the same list defines the monotone constraints of `XGB-mono`.

## 8. Evaluation (person-level, real data)
- R1 internal: repeated stratified 5-fold CV in RICH, 3 repeats, tiers T1 to T4.
- R2 cross-site: leave-one-site-out in RICH, tiers T1 to T3; sites 0 and 16 are pooled into training and never tested; 13 test sites.
- R3 cross-cohort: RICH to NAGALA and NAGALA to RICH, tiers T1 to T3, plus RICH to NAGALA for T4. Intercept-and-slope recalibration on a random 1,000 target participants (fixed seed), evaluated on the rest. External data never used for tuning.
- R4 covariate shift inside RICH: train age under 45, test 45 and over; train BMI under 25, test 25 and over; tiers T1 to T3.
- R5 natural missingness: RICH T4 complete-case versus full-data models (sensitivity), and per-site missingness reported.

## 9. Metrics
At 3 years: C-index; time-dependent AUC (IPCW); IPCW Brier and scaled Brier; calibration slope (Cox refit on the log cumulative hazard), calibration distance |slope - 1|, observed-to-expected ratio (Kaplan-Meier observed risk over mean predicted); Kaplan-Meier calibration by risk decile; decision-curve net benefit at 1% to 10% thresholds; constraint-violation rate (finite differences, +0.25 SD, 5,000 test patients). Subgroups by sex and age band (under 40, 40 to 54, 55 and over) for tier T2 in R2 and R3. Uncertainty: paired bootstrap, 200 resamples, 95% percentile intervals, in R3 for tiers T2 and T3 on a fixed random subsample of at most 40,000 target participants. Fold-level comparisons in R1 use the Nadeau-Bengio corrected resampled t interval.

## 10. Sensitivity analyses
S1 diagnosis-only outcome in RICH (T2, 1 repeat of 5-fold). S2 RICH T4 complete cases (1 repeat of 5-fold). S3 cross-cohort T2 and T3 without sex. S4 cohort-specific extras within each cohort (tiers E2 = T2 + extras without glucose or HbA1c; E3 = T3 + extras; 1 repeat of 5-fold).

## 11. Pre-specified decision rules (headline tier T2)
- C1 supported if the lower 95% limit of CKINN minus the best baseline C-index (Nadeau-Bengio, R1, T2) exceeds -0.01. T3 is reported but supports no claim alone.
- C2 and C3 supported only if, in both R3 directions at tier T2, the bootstrap 95% interval for CKINN minus each of `MLP` and `XGB-mono` excludes zero in favour of CKINN on scaled Brier or on calibration distance |slope - 1|.
- C4 supported if CKINN's violation rate is lower than MLP's by the Nadeau-Bengio interval in R1 at T2.
- Claims not supported are reported as not supported. No claim, tier, metric or comparison is added after seeing results. If neither C2 nor C3 holds, the paper is a transportability evaluation with a negative or neutral finding for constraints.

## 12. Reproducibility
Seeds: CV seed = repeat index; model seed = 1000 x repeat + fold (other analyses: seed 0). Configuration in `configs/protocol_v1.json`; every run appends to `results/manifest.jsonl` with code commit, protocol hash, data hashes and package versions. Code, configuration and processed results are released with the paper.

## 13. Changes from draft v0.1 (made before running)
- Random survival forest and discrete-time logistic models dropped; the interval-censored discrete-time sensitivity analysis and the binary-IPCW check dropped (stated as limitations).
- CV repeats 5 to 3; bootstrap resamples 500 to 200 (compute budget).
- Attribution stability dropped; C4 uses violation rate only.
- Real-data covariate shift splits limited to age and BMI.
- D1 to D6 resolved: headline tier T2 (D1); male = 1 confirmed from source counts (D2); framing is incident risk, not detection of undiagnosed diabetes (D3); C-index non-inferiority margin 0.01 (D4); no third cohort, claims restricted to cross-site and cross-cohort transport between two East Asian health-screening cohorts (D5); journal-agnostic manuscript, venue chosen after results (D6).

## 14. Out of scope
Synthetic data, knowledge graphs, physiological differential-equation models, fairness beyond sex and age, deployment.
