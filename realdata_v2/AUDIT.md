# Audit of the version 1 real-data study (CKINN versus M1 and M2)

Audited 2026-11-20, before any version 2 analysis was run. Material audited: `CKINN_real_data_study.ipynb` (clean), `CKINN_real_data_study_EXECUTED.ipynb`, `ckinn_real_outputs.zip` (tables, figures, report, manifest, hashes), the three data files and the proposal transcript.

Every number below is printed by `audit/audit_checks.py`; its output is stored in `audit/audit_checks_output.txt`, and the bracketed codes (A1, B2, ...) refer to lines of that output. Numbers quoted from the v1 report are labelled "v1 report".

Severity: **critical** = can change a conclusion; **major** = weakens a claim or its inference; **minor** = reporting or engineering.

## 1. What was confirmed

- The archive is internally consistent: all 28 output files match `output_hashes.json` (A1); the three input files are byte-identical to those named in the v1 manifest and to `data/raw/CHECKSUMS.sha256` of this repository (A3).
- The clean and executed notebooks have identical cell sources and the executed run has no error output (A4, A5).
- Every AUROC interval in `table09_main_results.csv` and every estimate in `table10_hypothesis_endpoints.csv` appears in the executed notebook's printed output (A6, A7). Recomputing the Holm adjustment and the PASS/FAIL/NULL rule from table 10 reproduces table 11 exactly (H1, H2, H3, H5 NULL; H4 PASS).
- NAGALA sex coding: the file uses 0/1 although the header says 1/2; code 1 has the higher ALT (B2) and waist, so 1 = men, as v1 assumed.
- NAGALA unit columns are internally consistent (mg/dL to mmol/L ratios constant at 38.67 for HDL and total cholesterol, 88.57 for triglycerides, 18.0 for glucose) (B4). No participant has baseline FPG of 7.0 mmol/L or more, or HbA1c of 6.5% or more, in NAGALA or RICH (B5, C7).
- RICH: the column "censor of diabetes at followup (1, Yes)" is the diabetes event, as v1 assumed; no participant with final-visit FPG of 7.0 mmol/L or more is coded as event-free (C7).

What could not be verified: the v1 archive holds no out-of-fold predictions, so none of the v1 performance numbers can be recomputed independently from stored predictions; they can only be checked against the printed output, which they match. The v1 run used Python 3.13 on another machine; it was not re-executed here.

## 2. Findings

### Design and inference

**F1 (critical). H4, the only PASS, is circular.** CKINN is trained to minimise a squared hinge on sign-adjusted input gradients; the H4 endpoints are the share of those gradients with the wrong sign and an integrated-gradient direction score that, with a zero baseline, is the sign of the path-averaged gradient. The test therefore checks that the optimiser reduced its own penalty. It is a manipulation check, not evidence that the explanations are more clinically valid.

**F2 (major). CKINN and M2 were not paired at the training level.** The fold seed is `seed_for("fit", coh, tier, m, f)`, which includes the model name (A11). CKINN and M2 therefore used different initial weights, different 85/15 inner splits for early stopping and thresholds, and, in RICH, different 40,000-row training subsamples (A12). The unit test that CKINN with lambda 0 equals M2 used a shared seed, but the engine did not. The protocol statement that "any difference between CKINN and M2 is caused by the constraint" is unsupported: the difference also contains seed and subsample noise.

**F3 (major). The H3 transport drop is ill-posed.** The drop compares AUROC in the source cohort with AUROC in a different target cohort. NAGALA to RICH gave a negative drop for LR, M2 and CKINN (v1 report, section 5) because RICH is easier to rank on these predictors, not because the models transport well. A transport gap must be measured against what the same model class achieves inside the target cohort with the same predictors.

**F4 (major). One split and one seed per network.** Each cohort had one 5-fold split and each fold one network seed. Seed variance, which for small neural networks can be of the same order as the CKINN minus M2 differences (v1 report: 0.0011 in RICH, 0.0035 in NHANES), was neither estimated nor separated from sampling variance.

**F5 (major). Lambda selection was noise-driven and hit the grid boundary.** Lambda was chosen at the largest grid value (10) in 4 of 5 cohort-tiers (A13), and the dev-score spread across lambda was at most 0.0049, with NAGALA's development subset holding 75 events. The grid {0.1, 1, 10} excluded both smaller and larger values and no rule was declared for near-ties.

**F6 (major). Unequal tuning.** M1 and M2 each evaluated six configurations; CKINN inherited M2's configuration and evaluated three lambda values. The baselines evaluated up to six. The budget was small for all models (A13 context; frozen config `tune_trials` 5).

**F7 (major). Inference.** (a) p-values were a normal approximation to 300 bootstrap draws, used because 300 draws cannot resolve Holm-adjusted p-values; the percentile intervals and the p-values could disagree. (b) The protocol text says Holm "across the five hypotheses"; the code pools all 22 primary endpoints in one family (A9). (c) The bootstrap resamples participants with fold models held fixed, so it ignores training variability. (d) No cluster bootstrap by site for RICH, no exact AUROC method (DeLong), no equivalence or power statement for NAGALA's 298 evaluation events.

**F8 (major). H5's statistic mixes base rates with model behaviour.** The gap is the larger of the sex gap and the age-band range in sensitivity at one common threshold. With a single threshold, older people (higher risk) cross it more often, so RICH shows age-band sensitivity gaps of 0.43 to 0.49 for every model (v1 report, section 6). The statistic is dominated by risk differences between groups and by groups with few events, and it says little about whether a model ranks equally well within groups.

**F9 (major). M1 is mislabelled and M1 versus M2 is confounded.** M1 is described as median-imputed; the code fills missing values with 0 on the transformed z-scale, which is the training mean (A14). M1 is also trained without the missingness augmentation that M2 and CKINN receive (A15), so M2 minus M1 mixes the mask input with the augmentation.

**F10 (major). The missingness endpoint favours augmented models by design.** The training augmentation removes the whole laboratory panel with probability 0.3, which is exactly the primary test scenario (MCAR panel removal at 30%) (A15). Models trained with augmentation (M2, CKINN, XGB_aug, LR_aug) are tested on the corruption they were trained on. CKINN versus M2 is unaffected; CKINN versus the non-augmented baselines is not a fair robustness comparison. Structural (site-driven) missingness in RICH was not evaluated.

**F11 (major). The RICH models used 40,000 of about 135,000 training rows per fold** (A12) and the development subset was capped at 20,000 rows. This is not stated in the report's limitations.

**F12 (moderate). The pooled-training ablation is transductive.** Each cohort is standardised with its own statistics, and at prediction time the statistics of the test fold itself are used (A16). No labels are used, but test-set feature distributions enter the prediction.

### Data

**F13 (critical for NHANES). The undiagnosed-diabetes label includes people who are treated.** Among the 29,698 labelled NHANES adults (D3), 263 report taking insulin or diabetes pills without reporting a diagnosis, and 119 of them are labelled positive (D4): 8.3% of the 1,432 positives. A further 23 had a refused, unknown or missing diagnosis question (D5). Of 16,440 fasting glucose values, 1,097 come from people who had fasted for less than 8 hours, and 69 positives rest only on such a value (D6). Under the corrected definition (no reported diagnosis, no glucose-lowering medication, fasting of at least 8 hours for FPG) the task has 29,410 labelled adults and 1,241 positives, prevalence 4.22% (D7).

**F14 (moderate). The NHANES extract is a non-random subset.** Cycles 7 to 10 hold 9,254 to 10,175 rows each, while cycles 1, 2, 4 and 5 hold 2,126 to 3,151 (D1); the selection behind the extract is undocumented. Code -1 (18,825 adults, excluded by v1 and documented only in a code comment) appears to be NHANES III (D2). Exercise is structurally missing in cycles 1 to 4 and alcohol in cycle 10 (D9, D10). Survey weights and design variables are present but unused (D8); because of the subset, weighted estimates would still not represent the US population.

**F15 (moderate). Outcome windows differ greatly between cohorts.** NAGALA follow-up runs from 0.45 to 12.96 years (median 5.39) and 2,431 participants were followed for more than 11 years (B1); RICH follow-up runs from 2.00 to 7.56 years (median 2.99), with no event before 2.00 years (C1). v1 does not clip NAGALA follow-up (the reported "clip at 11 years" is not in the code). The binary label therefore means "diabetes within up to 13 years" in NAGALA and "within 2 to 7.6 years" in RICH, which bears directly on transport (F3). At 3 years, RICH has 1,764 events and 106,055 participants censored earlier (C2), so a horizon analysis must be censoring-aware; v1's horizon analysis dropped censored people.

**F16 (moderate). RICH family history is probably under-recorded.** The overall "yes" rate is 0.0205, and 5 of 15 sites (30,053 participants) record fewer than 0.5% (C3). A zero is likely to mean "not recorded" at those sites, so the family-history constraint acts on a mostly uninformative variable.

**F17 (minor). RICH drinking status is recorded (missing 0.716, the same as smoking) but was not used** (C4). Exact zeros remain in RICH ALT (2) and AST (1) (C6); the locked protocol of this repository (deviation D-3) treats them as not measured.

### Reporting and engineering

**F18 (major). Raw data inside the outputs archive.** The archive contains full CSV copies of the NAGALA and RICH files (A2). Distributing the results archive redistributes the raw data.

**F19 (minor).** The verdict table writes the string NULL, which pandas reads back as missing (4 of 5 verdicts, A8). Brier differences are printed to four decimals and appear as -0.0000. Two no-op `... or None` lines and one `except Exception: pass` that hides bootstrap failures remain (A10); no `or True` was found. The protocol hash covers the configuration but not the code. Torch used all CPU threads, so results are not bit-reproducible across machines. Calibration has no confidence bands, no ECE and no intervals for slope and intercept; figures are 150 dpi PNG only.

**F20 (minor). Weak or capped comparators.** The RBF SVM was fitted on at most 8,000 rows and gave AUROC 0.749 in RICH (v1 report). No monotone-constrained competitor (XGBoost or LightGBM with `monotone_constraints`, sign-constrained logistic regression) and no additive model were included, so CKINN was compared only with unconstrained models.

## 3. Implications for the v1 conclusions

H1 to H3 and H5 were NULL and remain the record. The H4 PASS shows that the penalty works as an optimisation device; it does not show that CKINN's explanations are more valid (F1). Because CKINN and M2 were not paired (F2) and had one seed (F4), the v1 CKINN minus M2 intervals understate training noise. The NHANES label error (F13) affects every NHANES result, including the statement that CKINN had the highest AUROC there. Version 1 is kept unchanged as the record; how each finding is handled in version 2 is listed in `protocol_v2.md`, section 2.
