# CKINN versus M1 and M2 on real data, version 2: results report (generated)

Generated 2026-11-21 in `full` mode; configuration hash `7cdc5bdde12b17df`. Every number below is formatted by code from the
stored tables in `tables/`; none was typed. Outputs are research risk estimates, not diagnoses. Labels: P1 pre-specified in v1, P2 pre-specified
in v2 (decided after seeing v1), E exploratory.

## 1. Data
| cohort | rows | labelled | positives | prevalence_pct | predictors_present |
|---|---|---|---|---|---|
| NAGALA | 15,464 | 15,464 | 373 | 2.41 | 17 |
| RICH | 211,833 | 211,833 | 4,174 | 1.97 | 14 |
| NHANES | 37,522 | 29,410 | 1,241 | 4.22 | 18 |

NHANES label (amendment A11):

| definition | labelled | positives |
|---|---|---|
| v1 (no reported diagnosis) | 29,698 | 1,432 |
| v2 (A11: also no medication, answered, fasting >= 8 h for FPG) | 29,410 | 1,241 |

Development and evaluation rows:

| cohort | dev_rows | dev_events | eval_rows | eval_events |
|---|---|---|---|---|
| NAGALA | 3,092 | 75 | 12,372 | 298 |
| RICH | 20,000 | 394 | 169,467 | 3,339 |
| NHANES | 5,882 | 248 | 23,528 | 993 |
| NHANESv1 | 5,882 | 248 | 23,816 | 1,170 |

## 2. Pre-specified verdicts (P2)
- **H1 PASS** (family A: CKINN-M2): 3 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H2 NULL** (family A: CKINN-M2): 0 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H3 NULL** (family A: CKINN-M2): 1 of 2 units in the predicted direction after Holm correction, 1 in the opposite direction.
- **H4 PASS** (family A: CKINN-M2): 3 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H5 NULL** (family A: CKINN-M2): 0 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H1 PASS** (family B: CKINN-M1): 3 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H2 NULL** (family B: CKINN-M1): 0 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.
- **H3 NULL** (family B: CKINN-M1): 1 of 2 units in the predicted direction after Holm correction, 1 in the opposite direction.
- **H4 NULL** (family B: CKINN-M1): 2 of 3 units in the predicted direction after Holm correction, 1 in the opposite direction.
- **H5 NULL** (family B: CKINN-M1): 0 of 3 units in the predicted direction after Holm correction, 0 in the opposite direction.

| family | hypothesis | units | units_predicted | units_opposite | verdict |
|---|---|---|---|---|---|
| A: CKINN-M2 | H1 | 3 | 3 | 0 | PASS |
| A: CKINN-M2 | H2 | 3 | 0 | 0 | NULL |
| A: CKINN-M2 | H3 | 2 | 1 | 1 | NULL |
| A: CKINN-M2 | H4 | 3 | 3 | 0 | PASS |
| A: CKINN-M2 | H5 | 3 | 0 | 0 | NULL |
| B: CKINN-M1 | H1 | 3 | 3 | 0 | PASS |
| B: CKINN-M1 | H2 | 3 | 0 | 0 | NULL |
| B: CKINN-M1 | H3 | 2 | 1 | 1 | NULL |
| B: CKINN-M1 | H4 | 3 | 2 | 1 | NULL |
| B: CKINN-M1 | H5 | 3 | 0 | 0 | NULL |

H1 in detail (run-averaged AUROC difference, 95% paired-bootstrap interval, 2,000 draws):

- A: CKINN-M2, NAGALA (T2): AUROC difference 0.0024 [-0.0005, 0.0054]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.
- A: CKINN-M2, RICH (T2): AUROC difference 0.0002 [-0.0002, 0.0006]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.
- A: CKINN-M2, NHANES (T2): AUROC difference 0.0003 [-0.0005, 0.0011]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.
- B: CKINN-M1, NAGALA (T2): AUROC difference 0.0028 [-0.0006, 0.0064]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.
- B: CKINN-M1, RICH (T2): AUROC difference -0.0006 [-0.0014, 0.0001]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.
- B: CKINN-M1, NHANES (T2): AUROC difference 0.0054 [0.0026, 0.0084]; Holm-adjusted one-sided p for non-inferiority 0.0110; status predicted.

All primary endpoints (difference = CKINN minus comparator):

| family | hypothesis | unit | endpoint | difference | ci_lo | ci_hi | p_pred_holm | p_opp_holm | status |
|---|---|---|---|---|---|---|---|---|---|
| A: CKINN-M2 | H1 | NAGALA (T2) | AUROC (non-inferiority, margin 0.01) | 0.0024 | -0.0005 | 0.0054 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H2 | NAGALA (T2) | Brier score | -0.0001 | -0.0001 | 0.0000 | 0.4288 | 1.0000 | null |
| A: CKINN-M2 | H2 | NAGALA (T2) | |calibration slope - 1| | -0.0319 | -0.0450 | 0.0194 | 0.6957 | 1.0000 | null |
| A: CKINN-M2 | H4 | NAGALA (T2) | across-seed attribution agreement (Spearman) | 0.0602 | 0.0583 | 0.0620 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H5 | NAGALA (T2) | sex gap in sensitivity at FPR 0.10 | 0.0065 | -0.0275 | 0.0435 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H5 | NAGALA (T2) | age-band range in sensitivity at FPR 0.10 | 0.0174 | -0.0536 | 0.0797 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H1 | RICH (T2) | AUROC (non-inferiority, margin 0.01) | 0.0002 | -0.0002 | 0.0006 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H2 | RICH (T2) | Brier score | -0.0000 | -0.0000 | 0.0000 | 0.7346 | 1.0000 | null |
| A: CKINN-M2 | H2 | RICH (T2) | |calibration slope - 1| | 0.0072 | 0.0006 | 0.0107 | 1.0000 | 0.4513 | null |
| A: CKINN-M2 | H4 | RICH (T2) | across-seed attribution agreement (Spearman) | 0.0245 | 0.0228 | 0.0264 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H5 | RICH (T2) | sex gap in sensitivity at FPR 0.10 | -0.0043 | -0.0106 | 0.0037 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H5 | RICH (T2) | age-band range in sensitivity at FPR 0.10 | -0.0089 | -0.0183 | 0.0043 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H1 | NHANES (T2) | AUROC (non-inferiority, margin 0.01) | 0.0003 | -0.0005 | 0.0011 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H2 | NHANES (T2) | Brier score | -0.0000 | -0.0000 | 0.0000 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H2 | NHANES (T2) | |calibration slope - 1| | -0.0075 | -0.0117 | 0.0046 | 0.6957 | 1.0000 | null |
| A: CKINN-M2 | H4 | NHANES (T2) | across-seed attribution agreement (Spearman) | 0.0097 | 0.0090 | 0.0104 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H5 | NHANES (T2) | sex gap in sensitivity at FPR 0.10 | -0.0133 | -0.0231 | 0.0144 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H5 | NHANES (T2) | age-band range in sensitivity at FPR 0.10 | 0.0006 | -0.0164 | 0.0201 | 1.0000 | 1.0000 | null |
| A: CKINN-M2 | H3 | NAGALA -> RICH | transport gap (target CV AUROC - external AUROC) | -0.0110 | -0.0125 | -0.0096 | 0.0110 | 1.0000 | predicted |
| A: CKINN-M2 | H3 | NAGALA -> RICH | |calibration-in-the-large| in the target | 0.0942 | 0.0924 | 0.0959 | 1.0000 | 0.0110 | opposite |
| A: CKINN-M2 | H3 | RICH -> NAGALA | transport gap (target CV AUROC - external AUROC) | -0.0061 | -0.0095 | -0.0028 | 0.0140 | 1.0000 | predicted |
| A: CKINN-M2 | H3 | RICH -> NAGALA | |calibration-in-the-large| in the target | -0.0254 | -0.0277 | -0.0230 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H1 | NAGALA (T2) | AUROC (non-inferiority, margin 0.01) | 0.0028 | -0.0006 | 0.0064 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H2 | NAGALA (T2) | Brier score | -0.0000 | -0.0001 | 0.0000 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H2 | NAGALA (T2) | |calibration slope - 1| | 0.0232 | -0.0365 | 0.0345 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H4 | NAGALA (T2) | across-seed attribution agreement (Spearman) | 0.0954 | 0.0933 | 0.0974 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H5 | NAGALA (T2) | sex gap in sensitivity at FPR 0.10 | -0.0194 | -0.0529 | 0.0293 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H5 | NAGALA (T2) | age-band range in sensitivity at FPR 0.10 | 0.0474 | -0.0224 | 0.1004 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H1 | RICH (T2) | AUROC (non-inferiority, margin 0.01) | -0.0006 | -0.0014 | 0.0001 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H2 | RICH (T2) | Brier score | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H2 | RICH (T2) | |calibration slope - 1| | 0.0131 | -0.0171 | 0.0223 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H4 | RICH (T2) | across-seed attribution agreement (Spearman) | 0.0148 | 0.0127 | 0.0170 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H5 | RICH (T2) | sex gap in sensitivity at FPR 0.10 | -0.0060 | -0.0168 | 0.0050 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H5 | RICH (T2) | age-band range in sensitivity at FPR 0.10 | -0.0009 | -0.0159 | 0.0150 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H1 | NHANES (T2) | AUROC (non-inferiority, margin 0.01) | 0.0054 | 0.0026 | 0.0084 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H2 | NHANES (T2) | Brier score | -0.0002 | -0.0003 | -0.0001 | 0.0210 | 1.0000 | predicted |
| B: CKINN-M1 | H2 | NHANES (T2) | |calibration slope - 1| | 0.0292 | -0.0400 | 0.0512 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H4 | NHANES (T2) | across-seed attribution agreement (Spearman) | -0.0045 | -0.0061 | -0.0030 | 1.0000 | 0.0110 | opposite |
| B: CKINN-M1 | H5 | NHANES (T2) | sex gap in sensitivity at FPR 0.10 | -0.0414 | -0.0626 | 0.0163 | 0.8706 | 1.0000 | null |
| B: CKINN-M1 | H5 | NHANES (T2) | age-band range in sensitivity at FPR 0.10 | -0.0189 | -0.0544 | 0.0206 | 1.0000 | 1.0000 | null |
| B: CKINN-M1 | H3 | NAGALA -> RICH | transport gap (target CV AUROC - external AUROC) | -0.0121 | -0.0139 | -0.0102 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H3 | NAGALA -> RICH | |calibration-in-the-large| in the target | 0.1795 | 0.1775 | 0.1815 | 1.0000 | 0.0110 | opposite |
| B: CKINN-M1 | H3 | RICH -> NAGALA | transport gap (target CV AUROC - external AUROC) | -0.0067 | -0.0102 | -0.0033 | 0.0110 | 1.0000 | predicted |
| B: CKINN-M1 | H3 | RICH -> NAGALA | |calibration-in-the-large| in the target | -0.0366 | -0.0398 | -0.0333 | 0.0110 | 1.0000 | predicted |

## 3. Cross-checks of H1 and sources of variance (P2)
| family | cohort | auroc_diff_runavg | boot_lo | boot_hi | delong_diff_run0 | delong_p | nb_lo | nb_hi | nb_p | p_tost_margin_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|
| A: CKINN-M2 | NAGALA | 0.0024 | -0.0005 | 0.0054 | 0.0018 | 0.5025 | -0.0056 | 0.0104 | 0.5455 | 0.0005 |
| A: CKINN-M2 | RICH | 0.0002 | -0.0002 | 0.0006 | -0.0001 | 0.8180 | -0.0008 | 0.0010 | 0.8208 | 0.0005 |
| A: CKINN-M2 | NHANES | 0.0003 | -0.0005 | 0.0011 | -0.0016 | 0.0107 | -0.0033 | 0.0044 | 0.7717 | 0.0005 |
| B: CKINN-M1 | NAGALA | 0.0028 | -0.0006 | 0.0064 | 0.0072 | 0.0289 | -0.0058 | 0.0080 | 0.7444 | 0.0005 |
| B: CKINN-M1 | RICH | -0.0006 | -0.0014 | 0.0001 | -0.0006 | 0.2993 | -0.0015 | 0.0013 | 0.8692 | 0.0005 |
| B: CKINN-M1 | NHANES | 0.0054 | 0.0026 | 0.0084 | 0.0025 | 0.2041 | 0.0021 | 0.0094 | 0.0031 | 0.0035 |

For NAGALA, the bootstrap standard deviation of the CKINN minus M2 AUROC difference was 0.0015, so the difference detectable with 80% power at two-sided 5% is 0.0042, and the power of the non-inferiority test (margin 0.01) if the true difference were zero is 1.000. Both ignore the Holm correction and the variability of model training, so they overstate the power of the actual decision.

| cohort | model | metric | sampling_se_run0 | sd_between_main_runs | sd_across_seeds_repeat0 |
|---|---|---|---|---|---|
| NAGALA | M1 | auroc | 0.0126 | 0.0085 | 0.0078 |
| NAGALA | M2 | auroc | 0.0123 | 0.0048 | 0.0044 |
| NAGALA | CKINN | auroc | 0.0122 | 0.0052 | 0.0038 |
| NAGALA | LR | auroc | -- | 0.0030 | -- |
| NAGALA | XGB_mono | auroc | -- | 0.0034 | -- |
| NAGALA | CKINN-M2 | auroc | 0.0026 | 0.0026 | 0.0013 |
| NAGALA | CKINN-M1 | auroc | 0.0032 | 0.0043 | 0.0042 |
| RICH | M1 | auroc | 0.0029 | 0.0004 | 0.0008 |
| RICH | M2 | auroc | 0.0029 | 0.0004 | 0.0008 |
| RICH | CKINN | auroc | 0.0029 | 0.0004 | 0.0004 |
| RICH | LR | auroc | -- | 0.0001 | -- |
| RICH | XGB_mono | auroc | -- | 0.0003 | -- |
| RICH | CKINN-M2 | auroc | 0.0003 | 0.0003 | 0.0009 |
| RICH | CKINN-M1 | auroc | 0.0006 | 0.0003 | 0.0005 |
| NHANES | M1 | auroc | 0.0057 | 0.0017 | 0.0014 |
| NHANES | M2 | auroc | 0.0057 | 0.0018 | 0.0008 |
| NHANES | CKINN | auroc | 0.0057 | 0.0015 | 0.0010 |
| NHANES | LR | auroc | -- | 0.0010 | -- |
| NHANES | XGB_mono | auroc | -- | 0.0015 | -- |
| NHANES | CKINN-M2 | auroc | 0.0007 | 0.0019 | 0.0016 |
| NHANES | CKINN-M1 | auroc | 0.0019 | 0.0018 | 0.0018 |

## 4. Discrimination and calibration of every model (P2)
- NAGALA: highest run-averaged AUROC cLR (0.8057); lowest Brier cLR (0.02223).
- NHANES: highest run-averaged AUROC GAM (0.8271); lowest Brier CKINN (0.03741).
- RICH: highest run-averaged AUROC M1 (0.8494); lowest Brier M1 (0.01841).

| cohort | model | runs | events | auroc | auroc_lo | auroc_hi | auprc | brier | scaled_brier | slope | citl | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | LR | 5 | 298 | 0.8053 | 0.7814 | 0.8284 | 0.1288 | 0.0222 | 0.0538 | 0.9420 | 0.0058 | 0.0027 |
| NAGALA | cLR | 5 | 298 | 0.8057 | 0.7818 | 0.8291 | 0.1311 | 0.0222 | 0.0544 | 0.9463 | 0.0039 | 0.0027 |
| NAGALA | GAM | 5 | 298 | 0.8015 | 0.7779 | 0.8242 | 0.1213 | 0.0224 | 0.0492 | 1.1149 | 0.0037 | 0.0035 |
| NAGALA | RF | 5 | 298 | 0.8047 | 0.7835 | 0.8278 | 0.1122 | 0.0224 | 0.0453 | 1.0884 | 0.0050 | 0.0029 |
| NAGALA | XGB | 5 | 298 | 0.8008 | 0.7785 | 0.8237 | 0.1080 | 0.0225 | 0.0415 | 0.8971 | 0.0144 | 0.0029 |
| NAGALA | XGB_mono | 5 | 298 | 0.7983 | 0.7758 | 0.8214 | 0.1068 | 0.0226 | 0.0368 | 0.8521 | 0.0319 | 0.0038 |
| NAGALA | LGBM | 5 | 298 | 0.7964 | 0.7732 | 0.8204 | 0.1031 | 0.0226 | 0.0366 | 0.9002 | 0.0315 | 0.0029 |
| NAGALA | LGBM_mono | 5 | 298 | 0.7965 | 0.7733 | 0.8203 | 0.1086 | 0.0227 | 0.0350 | 0.8627 | 0.0091 | 0.0037 |
| NAGALA | M1 | 5 | 298 | 0.7981 | 0.7758 | 0.8208 | 0.1105 | 0.0225 | 0.0449 | 0.9716 | 0.0182 | 0.0031 |
| NAGALA | M2 | 5 | 298 | 0.7985 | 0.7747 | 0.8207 | 0.1097 | 0.0225 | 0.0438 | 0.9165 | -0.0004 | 0.0030 |
| NAGALA | CKINN | 5 | 298 | 0.8009 | 0.7771 | 0.8230 | 0.1133 | 0.0224 | 0.0461 | 0.9484 | 0.0141 | 0.0028 |
| NAGALA | MI_LR | 1 | 298 | 0.8039 | 0.7802 | 0.8278 | 0.1312 | 0.0222 | 0.0545 | 0.9210 | 0.0142 | 0.0031 |
| NAGALA | LR_aug | 1 | 298 | 0.8021 | 0.7774 | 0.8262 | 0.1341 | 0.0222 | 0.0538 | 0.9358 | -0.0464 | 0.0035 |
| NAGALA | XGB_aug | 1 | 298 | 0.7960 | 0.7716 | 0.8204 | 0.1123 | 0.0225 | 0.0446 | 0.8799 | 0.0181 | 0.0031 |
| RICH | LR | 5 | 3,339 | 0.8469 | 0.8412 | 0.8523 | 0.0947 | 0.0185 | 0.0410 | 1.0021 | -0.0020 | 0.0025 |
| RICH | cLR | 5 | 3,339 | 0.8469 | 0.8412 | 0.8523 | 0.0947 | 0.0185 | 0.0410 | 1.0017 | -0.0002 | 0.0026 |
| RICH | GAM | 5 | 3,339 | 0.8483 | 0.8425 | 0.8535 | 0.0990 | 0.0185 | 0.0448 | 0.9934 | -0.0008 | 0.0017 |
| RICH | RF | 5 | 3,339 | 0.8430 | 0.8373 | 0.8488 | 0.0925 | 0.0185 | 0.0411 | 1.3665 | 0.0012 | 0.0045 |
| RICH | XGB | 5 | 3,339 | 0.8481 | 0.8424 | 0.8533 | 0.0957 | 0.0184 | 0.0457 | 1.0847 | 0.0004 | 0.0015 |
| RICH | XGB_mono | 5 | 3,339 | 0.8455 | 0.8396 | 0.8509 | 0.0929 | 0.0185 | 0.0435 | 1.1890 | 0.0035 | 0.0030 |
| RICH | LGBM | 5 | 3,339 | 0.8479 | 0.8421 | 0.8531 | 0.0957 | 0.0184 | 0.0456 | 1.1049 | -0.0003 | 0.0017 |
| RICH | LGBM_mono | 5 | 3,339 | 0.8474 | 0.8417 | 0.8526 | 0.0961 | 0.0185 | 0.0442 | 1.0767 | 0.0014 | 0.0021 |
| RICH | M1 | 5 | 3,339 | 0.8494 | 0.8436 | 0.8545 | 0.0983 | 0.0184 | 0.0470 | 0.9990 | 0.0076 | 0.0011 |
| RICH | M2 | 5 | 3,339 | 0.8486 | 0.8430 | 0.8536 | 0.0979 | 0.0184 | 0.0464 | 0.9833 | -0.0123 | 0.0006 |
| RICH | CKINN | 5 | 3,339 | 0.8488 | 0.8433 | 0.8539 | 0.0987 | 0.0184 | 0.0468 | 0.9774 | -0.0137 | 0.0007 |
| RICH | MI_LR | 1 | 3,339 | 0.8476 | 0.8419 | 0.8531 | 0.0951 | 0.0185 | 0.0415 | 1.0061 | -0.0013 | 0.0024 |
| RICH | LR_aug | 1 | 3,339 | 0.8467 | 0.8409 | 0.8522 | 0.0944 | 0.0185 | 0.0405 | 1.0022 | -0.0306 | 0.0027 |
| RICH | XGB_aug | 1 | 3,339 | 0.8477 | 0.8419 | 0.8527 | 0.0939 | 0.0185 | 0.0446 | 1.0797 | 0.0026 | 0.0014 |
| NHANES | LR | 5 | 993 | 0.8142 | 0.8032 | 0.8258 | 0.1553 | 0.0379 | 0.0636 | 0.9988 | 0.0007 | 0.0037 |
| NHANES | cLR | 5 | 993 | 0.8142 | 0.8032 | 0.8258 | 0.1554 | 0.0379 | 0.0636 | 1.0000 | 0.0002 | 0.0036 |
| NHANES | GAM | 5 | 993 | 0.8271 | 0.8170 | 0.8378 | 0.1702 | 0.0374 | 0.0742 | 1.1083 | -0.0004 | 0.0041 |
| NHANES | RF | 5 | 993 | 0.8111 | 0.8005 | 0.8231 | 0.1570 | 0.0378 | 0.0648 | 1.0650 | -0.0012 | 0.0027 |
| NHANES | XGB | 5 | 993 | 0.8156 | 0.8055 | 0.8271 | 0.1620 | 0.0377 | 0.0665 | 1.2769 | 0.0041 | 0.0074 |
| NHANES | XGB_mono | 5 | 993 | 0.8155 | 0.8050 | 0.8270 | 0.1621 | 0.0377 | 0.0663 | 1.2797 | 0.0053 | 0.0078 |
| NHANES | LGBM | 5 | 993 | 0.8186 | 0.8084 | 0.8293 | 0.1610 | 0.0377 | 0.0672 | 1.0678 | 0.0078 | 0.0054 |
| NHANES | LGBM_mono | 5 | 993 | 0.8187 | 0.8084 | 0.8298 | 0.1637 | 0.0377 | 0.0662 | 1.0495 | 0.0041 | 0.0053 |
| NHANES | M1 | 5 | 993 | 0.8201 | 0.8098 | 0.8315 | 0.1636 | 0.0376 | 0.0702 | 0.9887 | 0.0208 | 0.0030 |
| NHANES | M2 | 5 | 993 | 0.8252 | 0.8151 | 0.8360 | 0.1688 | 0.0374 | 0.0742 | 0.9390 | 0.0151 | 0.0028 |
| NHANES | CKINN | 5 | 993 | 0.8256 | 0.8154 | 0.8365 | 0.1692 | 0.0374 | 0.0745 | 0.9465 | 0.0020 | 0.0027 |
| NHANES | MI_LR | 1 | 993 | 0.8140 | 0.8032 | 0.8254 | 0.1514 | 0.0379 | 0.0622 | 1.0163 | 0.0024 | 0.0019 |
| NHANES | LR_aug | 1 | 993 | 0.8151 | 0.8046 | 0.8265 | 0.1570 | 0.0378 | 0.0649 | 1.0182 | -0.0328 | 0.0034 |
| NHANES | XGB_aug | 1 | 993 | 0.8166 | 0.8062 | 0.8281 | 0.1625 | 0.0377 | 0.0667 | 1.2923 | 0.0082 | 0.0083 |

CKINN minus each competitor (AUROC; descriptive, no multiplicity correction):

| cohort | comparator | metric | ckinn_minus_comparator | lo | hi | runs |
|---|---|---|---|---|---|---|
| NAGALA | LR | auroc | -0.0044 | -0.0110 | 0.0027 | 5 |
| NAGALA | cLR | auroc | -0.0049 | -0.0115 | 0.0018 | 5 |
| NAGALA | GAM | auroc | -0.0006 | -0.0085 | 0.0075 | 5 |
| NAGALA | RF | auroc | -0.0038 | -0.0121 | 0.0043 | 5 |
| NAGALA | XGB | auroc | 0.0001 | -0.0068 | 0.0087 | 5 |
| NAGALA | XGB_mono | auroc | 0.0025 | -0.0037 | 0.0089 | 5 |
| NAGALA | LGBM | auroc | 0.0045 | -0.0028 | 0.0118 | 5 |
| NAGALA | LGBM_mono | auroc | 0.0044 | -0.0015 | 0.0106 | 5 |
| NAGALA | M1 | auroc | 0.0028 | -0.0004 | 0.0061 | 5 |
| NAGALA | M2 | auroc | 0.0024 | -0.0004 | 0.0056 | 5 |
| NAGALA | MI_LR | auroc | -0.0094 | -0.0187 | 0.0014 | 1 |
| NAGALA | LR_aug | auroc | -0.0077 | -0.0169 | 0.0020 | 1 |
| NAGALA | XGB_aug | auroc | -0.0016 | -0.0098 | 0.0085 | 1 |
| RICH | LR | auroc | 0.0019 | 0.0006 | 0.0033 | 5 |
| RICH | cLR | auroc | 0.0019 | 0.0005 | 0.0032 | 5 |
| RICH | GAM | auroc | 0.0005 | -0.0006 | 0.0016 | 5 |
| RICH | RF | auroc | 0.0058 | 0.0038 | 0.0076 | 5 |
| RICH | XGB | auroc | 0.0007 | -0.0006 | 0.0019 | 5 |
| RICH | XGB_mono | auroc | 0.0033 | 0.0018 | 0.0048 | 5 |
| RICH | LGBM | auroc | 0.0009 | -0.0004 | 0.0021 | 5 |
| RICH | LGBM_mono | auroc | 0.0014 | 0.0001 | 0.0028 | 5 |
| RICH | M1 | auroc | -0.0006 | -0.0013 | 0.0001 | 5 |
| RICH | M2 | auroc | 0.0002 | -0.0002 | 0.0007 | 5 |
| RICH | MI_LR | auroc | 0.0016 | -0.0000 | 0.0033 | 1 |
| RICH | LR_aug | auroc | 0.0025 | 0.0008 | 0.0041 | 1 |
| RICH | XGB_aug | auroc | 0.0016 | 0.0001 | 0.0032 | 1 |
| NHANES | LR | auroc | 0.0113 | 0.0070 | 0.0156 | 5 |
| NHANES | cLR | auroc | 0.0114 | 0.0069 | 0.0155 | 5 |
| NHANES | GAM | auroc | -0.0015 | -0.0051 | 0.0019 | 5 |
| NHANES | RF | auroc | 0.0145 | 0.0092 | 0.0198 | 5 |
| NHANES | XGB | auroc | 0.0100 | 0.0056 | 0.0143 | 5 |
| NHANES | XGB_mono | auroc | 0.0101 | 0.0058 | 0.0143 | 5 |
| NHANES | LGBM | auroc | 0.0070 | 0.0033 | 0.0111 | 5 |
| NHANES | LGBM_mono | auroc | 0.0068 | 0.0030 | 0.0110 | 5 |
| NHANES | M1 | auroc | 0.0054 | 0.0026 | 0.0085 | 5 |
| NHANES | M2 | auroc | 0.0003 | -0.0005 | 0.0012 | 5 |
| NHANES | MI_LR | auroc | 0.0101 | 0.0048 | 0.0150 | 1 |
| NHANES | LR_aug | auroc | 0.0090 | 0.0042 | 0.0134 | 1 |
| NHANES | XGB_aug | auroc | 0.0076 | 0.0024 | 0.0125 | 1 |

## 5. Clinical utility (P2)
| cohort | model | threshold | net_benefit | lo | hi | treat_all | minus_M2 | minus_M2_lo | minus_M2_hi |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | M1 | 0.0200 | 0.0120 | 0.0096 | 0.0146 | 0.0042 | 0.0000 | -0.0002 | 0.0003 |
| NAGALA | M1 | 0.0500 | 0.0058 | 0.0039 | 0.0077 | -0.0273 | -0.0001 | -0.0004 | 0.0003 |
| NAGALA | M2 | 0.0200 | 0.0120 | 0.0097 | 0.0145 | 0.0042 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | M2 | 0.0500 | 0.0058 | 0.0040 | 0.0079 | -0.0273 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | CKINN | 0.0200 | 0.0122 | 0.0097 | 0.0148 | 0.0042 | 0.0002 | -0.0001 | 0.0005 |
| NAGALA | CKINN | 0.0500 | 0.0059 | 0.0040 | 0.0079 | -0.0273 | 0.0000 | -0.0002 | 0.0003 |
| NAGALA | LR | 0.0200 | 0.0121 | 0.0098 | 0.0146 | 0.0042 | 0.0001 | -0.0004 | 0.0006 |
| NAGALA | LR | 0.0500 | 0.0061 | 0.0039 | 0.0080 | -0.0273 | 0.0002 | -0.0004 | 0.0008 |
| NAGALA | cLR | 0.0200 | 0.0122 | 0.0097 | 0.0147 | 0.0042 | 0.0001 | -0.0003 | 0.0006 |
| NAGALA | cLR | 0.0500 | 0.0062 | 0.0040 | 0.0082 | -0.0273 | 0.0003 | -0.0003 | 0.0008 |
| NAGALA | GAM | 0.0200 | 0.0120 | 0.0095 | 0.0145 | 0.0042 | -0.0000 | -0.0006 | 0.0004 |
| NAGALA | GAM | 0.0500 | 0.0064 | 0.0044 | 0.0084 | -0.0273 | 0.0006 | -0.0000 | 0.0011 |
| NAGALA | XGB | 0.0200 | 0.0122 | 0.0098 | 0.0146 | 0.0042 | 0.0002 | -0.0004 | 0.0006 |
| NAGALA | XGB | 0.0500 | 0.0060 | 0.0043 | 0.0080 | -0.0273 | 0.0002 | -0.0005 | 0.0008 |
| NAGALA | XGB_mono | 0.0200 | 0.0120 | 0.0098 | 0.0145 | 0.0042 | -0.0000 | -0.0006 | 0.0005 |
| NAGALA | XGB_mono | 0.0500 | 0.0058 | 0.0040 | 0.0076 | -0.0273 | -0.0000 | -0.0007 | 0.0005 |
| NAGALA | LGBM_mono | 0.0200 | 0.0121 | 0.0098 | 0.0146 | 0.0042 | 0.0001 | -0.0004 | 0.0006 |
| NAGALA | LGBM_mono | 0.0500 | 0.0062 | 0.0042 | 0.0080 | -0.0273 | 0.0003 | -0.0003 | 0.0008 |
| RICH | M1 | 0.0200 | 0.0109 | 0.0103 | 0.0116 | -0.0003 | 0.0001 | -0.0000 | 0.0001 |
| RICH | M1 | 0.0500 | 0.0050 | 0.0045 | 0.0055 | -0.0319 | 0.0001 | 0.0000 | 0.0002 |
| RICH | M2 | 0.0200 | 0.0109 | 0.0103 | 0.0115 | -0.0003 | 0.0000 | 0.0000 | 0.0000 |
| RICH | M2 | 0.0500 | 0.0049 | 0.0044 | 0.0053 | -0.0319 | 0.0000 | 0.0000 | 0.0000 |
| RICH | CKINN | 0.0200 | 0.0109 | 0.0103 | 0.0115 | -0.0003 | 0.0000 | -0.0000 | 0.0001 |
| RICH | CKINN | 0.0500 | 0.0049 | 0.0044 | 0.0054 | -0.0319 | 0.0000 | -0.0001 | 0.0001 |
| RICH | LR | 0.0200 | 0.0107 | 0.0102 | 0.0113 | -0.0003 | -0.0002 | -0.0003 | -0.0000 |
| RICH | LR | 0.0500 | 0.0046 | 0.0041 | 0.0051 | -0.0319 | -0.0003 | -0.0005 | -0.0001 |
| RICH | cLR | 0.0200 | 0.0107 | 0.0102 | 0.0113 | -0.0003 | -0.0001 | -0.0003 | -0.0000 |
| RICH | cLR | 0.0500 | 0.0046 | 0.0041 | 0.0051 | -0.0319 | -0.0003 | -0.0005 | -0.0001 |
| RICH | GAM | 0.0200 | 0.0109 | 0.0103 | 0.0114 | -0.0003 | -0.0000 | -0.0001 | 0.0001 |
| RICH | GAM | 0.0500 | 0.0048 | 0.0043 | 0.0053 | -0.0319 | -0.0000 | -0.0002 | 0.0001 |
| RICH | XGB | 0.0200 | 0.0109 | 0.0103 | 0.0115 | -0.0003 | 0.0000 | -0.0001 | 0.0001 |
| RICH | XGB | 0.0500 | 0.0049 | 0.0044 | 0.0053 | -0.0319 | -0.0000 | -0.0001 | 0.0002 |
| RICH | XGB_mono | 0.0200 | 0.0108 | 0.0102 | 0.0114 | -0.0003 | -0.0001 | -0.0002 | 0.0000 |
| RICH | XGB_mono | 0.0500 | 0.0046 | 0.0041 | 0.0050 | -0.0319 | -0.0003 | -0.0005 | -0.0001 |
| RICH | LGBM_mono | 0.0200 | 0.0109 | 0.0103 | 0.0115 | -0.0003 | -0.0000 | -0.0001 | 0.0001 |
| RICH | LGBM_mono | 0.0500 | 0.0047 | 0.0043 | 0.0052 | -0.0319 | -0.0001 | -0.0003 | 0.0000 |
| NHANES | M1 | 0.0500 | 0.0181 | 0.0159 | 0.0207 | -0.0082 | -0.0004 | -0.0010 | 0.0001 |
| NHANES | M1 | 0.1000 | 0.0080 | 0.0062 | 0.0101 | -0.0642 | -0.0005 | -0.0012 | 0.0001 |
| NHANES | M2 | 0.0500 | 0.0185 | 0.0165 | 0.0211 | -0.0082 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | M2 | 0.1000 | 0.0086 | 0.0067 | 0.0108 | -0.0642 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | CKINN | 0.0500 | 0.0186 | 0.0165 | 0.0211 | -0.0082 | 0.0001 | -0.0001 | 0.0003 |
| NHANES | CKINN | 0.1000 | 0.0085 | 0.0067 | 0.0109 | -0.0642 | -0.0000 | -0.0003 | 0.0002 |
| NHANES | LR | 0.0500 | 0.0174 | 0.0153 | 0.0199 | -0.0082 | -0.0011 | -0.0019 | -0.0003 |
| NHANES | LR | 0.1000 | 0.0077 | 0.0060 | 0.0097 | -0.0642 | -0.0009 | -0.0018 | 0.0000 |
| NHANES | cLR | 0.0500 | 0.0174 | 0.0153 | 0.0199 | -0.0082 | -0.0011 | -0.0019 | -0.0003 |
| NHANES | cLR | 0.1000 | 0.0076 | 0.0060 | 0.0097 | -0.0642 | -0.0009 | -0.0018 | -0.0000 |
| NHANES | GAM | 0.0500 | 0.0186 | 0.0164 | 0.0211 | -0.0082 | 0.0000 | -0.0005 | 0.0006 |
| NHANES | GAM | 0.1000 | 0.0081 | 0.0063 | 0.0102 | -0.0642 | -0.0004 | -0.0013 | 0.0002 |
| NHANES | XGB | 0.0500 | 0.0178 | 0.0156 | 0.0203 | -0.0082 | -0.0007 | -0.0013 | -0.0000 |
| NHANES | XGB | 0.1000 | 0.0069 | 0.0054 | 0.0086 | -0.0642 | -0.0016 | -0.0025 | -0.0008 |
| NHANES | XGB_mono | 0.0500 | 0.0180 | 0.0158 | 0.0206 | -0.0082 | -0.0005 | -0.0011 | 0.0002 |
| NHANES | XGB_mono | 0.1000 | 0.0070 | 0.0055 | 0.0087 | -0.0642 | -0.0016 | -0.0026 | -0.0007 |
| NHANES | LGBM_mono | 0.0500 | 0.0180 | 0.0159 | 0.0204 | -0.0082 | -0.0006 | -0.0012 | 0.0000 |
| NHANES | LGBM_mono | 0.1000 | 0.0076 | 0.0060 | 0.0095 | -0.0642 | -0.0010 | -0.0018 | -0.0002 |

| cohort | model | operating_point | sens | spec | ppv | npv | f1 | fpr |
|---|---|---|---|---|---|---|---|---|
| NAGALA | M1 | fixed specificity 0.90 | 0.462 | 0.900 | 0.103 | 0.985 | 0.168 | 0.100 |
| NAGALA | M1 | fixed sensitivity 0.80 | 0.785 | 0.650 | 0.053 | 0.992 | 0.099 | 0.350 |
| NAGALA | M2 | fixed specificity 0.90 | 0.460 | 0.899 | 0.101 | 0.985 | 0.166 | 0.101 |
| NAGALA | M2 | fixed sensitivity 0.80 | 0.773 | 0.660 | 0.054 | 0.992 | 0.100 | 0.340 |
| NAGALA | CKINN | fixed specificity 0.90 | 0.462 | 0.900 | 0.102 | 0.985 | 0.167 | 0.100 |
| NAGALA | CKINN | fixed sensitivity 0.80 | 0.783 | 0.654 | 0.053 | 0.992 | 0.100 | 0.346 |
| NAGALA | LR | fixed specificity 0.90 | 0.463 | 0.901 | 0.104 | 0.986 | 0.169 | 0.099 |
| NAGALA | LR | fixed sensitivity 0.80 | 0.791 | 0.646 | 0.053 | 0.992 | 0.099 | 0.354 |
| NAGALA | XGB_mono | fixed specificity 0.90 | 0.457 | 0.899 | 0.101 | 0.985 | 0.165 | 0.101 |
| NAGALA | XGB_mono | fixed sensitivity 0.80 | 0.787 | 0.651 | 0.053 | 0.992 | 0.099 | 0.349 |
| RICH | M1 | fixed specificity 0.90 | 0.504 | 0.901 | 0.093 | 0.989 | 0.156 | 0.099 |
| RICH | M1 | fixed sensitivity 0.80 | 0.802 | 0.751 | 0.061 | 0.995 | 0.113 | 0.249 |
| RICH | M2 | fixed specificity 0.90 | 0.508 | 0.900 | 0.093 | 0.989 | 0.157 | 0.100 |
| RICH | M2 | fixed sensitivity 0.80 | 0.796 | 0.754 | 0.061 | 0.995 | 0.113 | 0.246 |
| RICH | CKINN | fixed specificity 0.90 | 0.504 | 0.900 | 0.092 | 0.989 | 0.156 | 0.100 |
| RICH | CKINN | fixed sensitivity 0.80 | 0.799 | 0.753 | 0.061 | 0.995 | 0.113 | 0.247 |
| RICH | LR | fixed specificity 0.90 | 0.497 | 0.900 | 0.091 | 0.989 | 0.154 | 0.100 |
| RICH | LR | fixed sensitivity 0.80 | 0.797 | 0.750 | 0.060 | 0.995 | 0.112 | 0.250 |
| RICH | XGB_mono | fixed specificity 0.90 | 0.492 | 0.900 | 0.090 | 0.989 | 0.153 | 0.100 |
| RICH | XGB_mono | fixed sensitivity 0.80 | 0.799 | 0.748 | 0.060 | 0.995 | 0.111 | 0.252 |
| NHANES | M1 | fixed specificity 0.90 | 0.435 | 0.901 | 0.162 | 0.973 | 0.236 | 0.099 |
| NHANES | M1 | fixed sensitivity 0.80 | 0.809 | 0.671 | 0.098 | 0.988 | 0.175 | 0.329 |
| NHANES | M2 | fixed specificity 0.90 | 0.454 | 0.900 | 0.166 | 0.974 | 0.243 | 0.100 |
| NHANES | M2 | fixed sensitivity 0.80 | 0.811 | 0.684 | 0.102 | 0.988 | 0.181 | 0.316 |
| NHANES | CKINN | fixed specificity 0.90 | 0.458 | 0.899 | 0.167 | 0.974 | 0.244 | 0.101 |
| NHANES | CKINN | fixed sensitivity 0.80 | 0.807 | 0.681 | 0.101 | 0.988 | 0.179 | 0.319 |
| NHANES | LR | fixed specificity 0.90 | 0.428 | 0.902 | 0.162 | 0.973 | 0.235 | 0.098 |
| NHANES | LR | fixed sensitivity 0.80 | 0.811 | 0.660 | 0.095 | 0.988 | 0.170 | 0.340 |
| NHANES | XGB_mono | fixed specificity 0.90 | 0.424 | 0.900 | 0.158 | 0.973 | 0.230 | 0.100 |
| NHANES | XGB_mono | fixed sensitivity 0.80 | 0.805 | 0.669 | 0.097 | 0.987 | 0.173 | 0.331 |

## 6. Censoring-aware evaluation (P2) and 3-year IPCW models (E)
| cohort | model | harrell_c | uno_c_3y | td_auroc_3y | td_auroc_3y_lo | td_auroc_3y_hi | td_auroc_5y | td_auroc_5y_lo | td_auroc_5y_hi |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | LR | 0.8016 | 0.7952 | 0.7951 | 0.7533 | 0.8366 | 0.8077 | 0.7748 | 0.8444 |
| NAGALA | cLR | 0.8023 | 0.7955 | 0.7954 | 0.7536 | 0.8371 | 0.8083 | 0.7751 | 0.8451 |
| NAGALA | GAM | 0.7947 | 0.7925 | 0.7916 | 0.7495 | 0.8315 | 0.7977 | 0.7618 | 0.8348 |
| NAGALA | RF | 0.8025 | 0.8062 | 0.8066 | 0.7678 | 0.8460 | 0.8098 | 0.7734 | 0.8419 |
| NAGALA | XGB | 0.7958 | 0.7974 | 0.7967 | 0.7569 | 0.8386 | 0.8007 | 0.7649 | 0.8368 |
| NAGALA | XGB_mono | 0.7957 | 0.7959 | 0.7955 | 0.7574 | 0.8340 | 0.8018 | 0.7678 | 0.8380 |
| NAGALA | LGBM | 0.7924 | 0.7944 | 0.7936 | 0.7527 | 0.8374 | 0.7975 | 0.7611 | 0.8347 |
| NAGALA | LGBM_mono | 0.7935 | 0.7939 | 0.7932 | 0.7517 | 0.8351 | 0.7996 | 0.7650 | 0.8369 |
| NAGALA | M1 | 0.7964 | 0.7927 | 0.7922 | 0.7529 | 0.8338 | 0.8032 | 0.7685 | 0.8377 |
| NAGALA | M2 | 0.7974 | 0.7952 | 0.7948 | 0.7533 | 0.8347 | 0.8051 | 0.7714 | 0.8398 |
| NAGALA | CKINN | 0.8007 | 0.7988 | 0.7986 | 0.7564 | 0.8399 | 0.8089 | 0.7766 | 0.8426 |
| RICH | LR | 0.8350 | 0.8336 | 0.8360 | 0.8274 | 0.8446 | -- | -- | -- |
| RICH | cLR | 0.8350 | 0.8337 | 0.8360 | 0.8275 | 0.8446 | -- | -- | -- |
| RICH | GAM | 0.8361 | 0.8337 | 0.8356 | 0.8270 | 0.8446 | -- | -- | -- |
| RICH | RF | 0.8343 | 0.8323 | 0.8349 | 0.8257 | 0.8438 | -- | -- | -- |
| RICH | XGB | 0.8379 | 0.8361 | 0.8384 | 0.8298 | 0.8474 | -- | -- | -- |
| RICH | XGB_mono | 0.8355 | 0.8340 | 0.8365 | 0.8274 | 0.8452 | -- | -- | -- |
| RICH | LGBM | 0.8378 | 0.8360 | 0.8383 | 0.8296 | 0.8473 | -- | -- | -- |
| RICH | LGBM_mono | 0.8371 | 0.8355 | 0.8378 | 0.8288 | 0.8468 | -- | -- | -- |
| RICH | M1 | 0.8386 | 0.8368 | 0.8390 | 0.8304 | 0.8481 | -- | -- | -- |
| RICH | M2 | 0.8372 | 0.8350 | 0.8370 | 0.8289 | 0.8458 | -- | -- | -- |
| RICH | CKINN | 0.8365 | 0.8342 | 0.8359 | 0.8277 | 0.8446 | -- | -- | -- |

| cohort | model | ipcw_brier_3y | scaled_brier_3y | td_auroc_3y | km_observed_risk_3y | mean_predicted_risk_3y |
|---|---|---|---|---|---|---|
| NAGALA | LR | 0.0086 | -0.0007 | 0.7675 | 0.0086 | 0.0085 |
| NAGALA | XGB_mono | 0.0087 | -0.0092 | 0.7769 | 0.0086 | 0.0079 |
| NAGALA | M2 | 0.0085 | 0.0095 | 0.7763 | 0.0086 | 0.0079 |
| NAGALA | CKINN | 0.0084 | 0.0143 | 0.7808 | 0.0086 | 0.0087 |
| RICH | LR | 0.0114 | 0.0206 | 0.8395 | 0.0118 | 0.0118 |
| RICH | XGB_mono | 0.0114 | 0.0230 | 0.8387 | 0.0118 | 0.0117 |
| RICH | M2 | 0.0113 | 0.0292 | 0.8465 | 0.0118 | 0.0125 |
| RICH | CKINN | 0.0113 | 0.0266 | 0.8417 | 0.0118 | 0.0120 |

## 7. Subgroups (E; H5 is in section 2)
| cohort | group_type | model | gap | value | lo | hi | minus_M2 | minus_M2_lo | minus_M2_hi |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | sex | M1 | auroc | 0.0456 | 0.0098 | 0.1109 | 0.0023 | -0.0098 | 0.0124 |
| NAGALA | sex | M1 | tpr | 0.1324 | 0.0411 | 0.2594 | 0.0259 | -0.0189 | 0.0628 |
| NAGALA | sex | M1 | tprthr | 0.3167 | 0.2013 | 0.4188 | -0.0022 | -0.0314 | 0.0294 |
| NAGALA | sex | M1 | fprthr | 0.1507 | 0.1429 | 0.1601 | 0.0024 | -0.0004 | 0.0050 |
| NAGALA | sex | M1 | citl | 0.1435 | 0.0499 | 0.4486 | -0.0297 | -0.0404 | 0.0271 |
| NAGALA | sex | M2 | auroc | 0.0434 | 0.0142 | 0.1101 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | sex | M2 | tpr | 0.1065 | 0.0302 | 0.2290 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | sex | M2 | tprthr | 0.3189 | 0.2051 | 0.4207 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | sex | M2 | fprthr | 0.1483 | 0.1401 | 0.1576 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | sex | M2 | citl | 0.1732 | 0.0559 | 0.4848 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | sex | CKINN | auroc | 0.0451 | 0.0080 | 0.1091 | 0.0017 | -0.0102 | 0.0099 |
| NAGALA | sex | CKINN | tpr | 0.1130 | 0.0372 | 0.2412 | 0.0065 | -0.0260 | 0.0487 |
| NAGALA | sex | CKINN | tprthr | 0.3176 | 0.2000 | 0.4222 | -0.0013 | -0.0220 | 0.0180 |
| NAGALA | sex | CKINN | fprthr | 0.1479 | 0.1394 | 0.1575 | -0.0005 | -0.0029 | 0.0017 |
| NAGALA | sex | CKINN | citl | 0.1557 | 0.0467 | 0.4645 | -0.0176 | -0.0270 | 0.0185 |
| NAGALA | age_band | M1 | auroc | 0.2031 | 0.0814 | 0.3420 | -0.0036 | -0.0199 | 0.0135 |
| NAGALA | age_band | M1 | tpr | 0.2552 | 0.1025 | 0.4657 | -0.0300 | -0.0796 | 0.0225 |
| NAGALA | age_band | M1 | tprthr | 0.1176 | 0.0542 | 0.3266 | -0.0055 | -0.0544 | 0.0481 |
| NAGALA | age_band | M1 | fprthr | 0.1136 | 0.0867 | 0.1385 | 0.0015 | -0.0080 | 0.0119 |
| NAGALA | age_band | M1 | citl | 0.4099 | 0.1315 | 0.8781 | 0.0571 | -0.0245 | 0.0732 |
| NAGALA | age_band | M2 | auroc | 0.2067 | 0.0767 | 0.3464 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | age_band | M2 | tpr | 0.2852 | 0.1187 | 0.5004 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | age_band | M2 | tprthr | 0.1231 | 0.0598 | 0.3136 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | age_band | M2 | fprthr | 0.1121 | 0.0859 | 0.1387 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | age_band | M2 | citl | 0.3527 | 0.1141 | 0.8248 | 0.0000 | 0.0000 | 0.0000 |
| NAGALA | age_band | CKINN | auroc | 0.2055 | 0.0737 | 0.3438 | -0.0012 | -0.0221 | 0.0200 |
| NAGALA | age_band | CKINN | tpr | 0.3026 | 0.1348 | 0.5031 | 0.0174 | -0.0541 | 0.0755 |
| NAGALA | age_band | CKINN | tprthr | 0.0813 | 0.0459 | 0.2926 | -0.0418 | -0.0805 | 0.0285 |
| NAGALA | age_band | CKINN | fprthr | 0.1202 | 0.0943 | 0.1488 | 0.0082 | 0.0007 | 0.0161 |
| NAGALA | age_band | CKINN | citl | 0.3406 | 0.0817 | 0.8118 | -0.0121 | -0.0402 | 0.0260 |
| RICH | sex | M1 | auroc | 0.0449 | 0.0333 | 0.0580 | -0.0001 | -0.0019 | 0.0018 |
| RICH | sex | M1 | tpr | 0.1673 | 0.1341 | 0.2005 | 0.0017 | -0.0081 | 0.0122 |
| RICH | sex | M1 | tprthr | 0.1202 | 0.0851 | 0.1553 | -0.0223 | -0.0336 | -0.0104 |
| RICH | sex | M1 | fprthr | 0.0914 | 0.0889 | 0.0940 | -0.0043 | -0.0051 | -0.0036 |
| RICH | sex | M1 | citl | 0.0219 | 0.0199 | 0.0925 | -0.0193 | -0.0402 | 0.0390 |
| RICH | sex | M2 | auroc | 0.0449 | 0.0325 | 0.0580 | 0.0000 | 0.0000 | 0.0000 |
| RICH | sex | M2 | tpr | 0.1656 | 0.1279 | 0.1991 | 0.0000 | 0.0000 | 0.0000 |
| RICH | sex | M2 | tprthr | 0.1425 | 0.1071 | 0.1781 | 0.0000 | 0.0000 | 0.0000 |
| RICH | sex | M2 | fprthr | 0.0957 | 0.0932 | 0.0982 | 0.0000 | 0.0000 | 0.0000 |
| RICH | sex | M2 | citl | 0.0413 | 0.0160 | 0.1159 | 0.0000 | 0.0000 | 0.0000 |
| RICH | sex | CKINN | auroc | 0.0459 | 0.0336 | 0.0595 | 0.0010 | -0.0000 | 0.0020 |
| RICH | sex | CKINN | tpr | 0.1613 | 0.1248 | 0.1953 | -0.0043 | -0.0099 | 0.0042 |
| RICH | sex | CKINN | tprthr | 0.1424 | 0.1053 | 0.1776 | -0.0001 | -0.0069 | 0.0064 |
| RICH | sex | CKINN | fprthr | 0.0966 | 0.0943 | 0.0992 | 0.0010 | 0.0005 | 0.0015 |
| RICH | sex | CKINN | citl | 0.0556 | 0.0215 | 0.1310 | 0.0144 | -0.0089 | 0.0168 |
| RICH | age_band | M1 | auroc | 0.1818 | 0.1564 | 0.2043 | 0.0024 | -0.0017 | 0.0069 |
| RICH | age_band | M1 | tpr | 0.3849 | 0.3368 | 0.4343 | -0.0080 | -0.0224 | 0.0106 |
| RICH | age_band | M1 | tprthr | 0.4959 | 0.4546 | 0.5367 | 0.0223 | 0.0052 | 0.0383 |
| RICH | age_band | M1 | fprthr | 0.4318 | 0.4253 | 0.4383 | 0.0117 | 0.0095 | 0.0138 |
| RICH | age_band | M1 | citl | 0.0451 | 0.0378 | 0.1437 | -0.0225 | -0.0436 | 0.0100 |
| RICH | age_band | M2 | auroc | 0.1794 | 0.1547 | 0.2012 | 0.0000 | 0.0000 | 0.0000 |
| RICH | age_band | M2 | tpr | 0.3929 | 0.3457 | 0.4366 | 0.0000 | 0.0000 | 0.0000 |
| RICH | age_band | M2 | tprthr | 0.4736 | 0.4345 | 0.5130 | 0.0000 | 0.0000 | 0.0000 |
| RICH | age_band | M2 | fprthr | 0.4202 | 0.4135 | 0.4271 | 0.0000 | 0.0000 | 0.0000 |
| RICH | age_band | M2 | citl | 0.0676 | 0.0483 | 0.1789 | 0.0000 | 0.0000 | 0.0000 |
| RICH | age_band | CKINN | auroc | 0.1784 | 0.1529 | 0.2012 | -0.0009 | -0.0031 | 0.0012 |
| RICH | age_band | CKINN | tpr | 0.3839 | 0.3393 | 0.4305 | -0.0089 | -0.0192 | 0.0052 |
| RICH | age_band | CKINN | tprthr | 0.4635 | 0.4217 | 0.5039 | -0.0102 | -0.0185 | -0.0020 |
| RICH | age_band | CKINN | fprthr | 0.4113 | 0.4048 | 0.4180 | -0.0088 | -0.0103 | -0.0074 |
| RICH | age_band | CKINN | citl | 0.0961 | 0.0767 | 0.2048 | 0.0285 | 0.0034 | 0.0377 |
| NHANES | sex | M1 | auroc | 0.0298 | 0.0073 | 0.0511 | 0.0068 | 0.0001 | 0.0131 |
| NHANES | sex | M1 | tpr | 0.0645 | 0.0164 | 0.1229 | 0.0281 | -0.0173 | 0.0521 |
| NHANES | sex | M1 | tprthr | 0.0113 | 0.0087 | 0.0684 | -0.0264 | -0.0489 | 0.0279 |
| NHANES | sex | M1 | fprthr | 0.0225 | 0.0150 | 0.0295 | -0.0040 | -0.0076 | -0.0006 |
| NHANES | sex | M1 | citl | 0.0096 | 0.0087 | 0.1433 | -0.0303 | -0.0450 | 0.0422 |
| NHANES | sex | M2 | auroc | 0.0229 | 0.0035 | 0.0447 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | sex | M2 | tpr | 0.0364 | 0.0102 | 0.0961 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | sex | M2 | tprthr | 0.0376 | 0.0148 | 0.0993 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | sex | M2 | fprthr | 0.0265 | 0.0189 | 0.0336 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | sex | M2 | citl | 0.0400 | 0.0091 | 0.1678 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | sex | CKINN | auroc | 0.0224 | 0.0036 | 0.0445 | -0.0005 | -0.0021 | 0.0010 |
| NHANES | sex | CKINN | tpr | 0.0232 | 0.0084 | 0.0856 | -0.0133 | -0.0226 | 0.0149 |
| NHANES | sex | CKINN | tprthr | 0.0364 | 0.0119 | 0.0980 | -0.0013 | -0.0125 | 0.0075 |
| NHANES | sex | CKINN | fprthr | 0.0255 | 0.0178 | 0.0324 | -0.0010 | -0.0024 | 0.0004 |
| NHANES | sex | CKINN | citl | 0.0351 | 0.0151 | 0.1620 | -0.0048 | -0.0071 | 0.0084 |
| NHANES | age_band | M1 | auroc | 0.1361 | 0.0923 | 0.1727 | 0.0091 | -0.0015 | 0.0205 |
| NHANES | age_band | M1 | tpr | 0.3096 | 0.2109 | 0.4057 | 0.0195 | -0.0174 | 0.0529 |
| NHANES | age_band | M1 | tprthr | 0.2732 | 0.1895 | 0.3508 | -0.0016 | -0.0317 | 0.0288 |
| NHANES | age_band | M1 | fprthr | 0.1761 | 0.1664 | 0.1845 | -0.0040 | -0.0084 | 0.0001 |
| NHANES | age_band | M1 | citl | 0.1599 | 0.0525 | 0.3769 | 0.0320 | -0.0196 | 0.0420 |
| NHANES | age_band | M2 | auroc | 0.1269 | 0.0845 | 0.1662 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | age_band | M2 | tpr | 0.2902 | 0.1951 | 0.3857 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | age_band | M2 | tprthr | 0.2748 | 0.1953 | 0.3540 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | age_band | M2 | fprthr | 0.1801 | 0.1706 | 0.1893 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | age_band | M2 | citl | 0.1279 | 0.0382 | 0.3624 | 0.0000 | 0.0000 | 0.0000 |
| NHANES | age_band | CKINN | auroc | 0.1237 | 0.0811 | 0.1632 | -0.0032 | -0.0068 | 0.0002 |
| NHANES | age_band | CKINN | tpr | 0.2908 | 0.1938 | 0.3850 | 0.0006 | -0.0168 | 0.0202 |
| NHANES | age_band | CKINN | tprthr | 0.2793 | 0.1978 | 0.3598 | 0.0046 | -0.0086 | 0.0163 |
| NHANES | age_band | CKINN | fprthr | 0.1805 | 0.1706 | 0.1903 | 0.0005 | -0.0012 | 0.0022 |
| NHANES | age_band | CKINN | citl | 0.1593 | 0.0505 | 0.3860 | 0.0314 | -0.0190 | 0.0348 |

## 8. Missingness (P2)
| cohort | model | runs | none | mcar10 | mcar30 | mcar50 | mar10 | mar30 | mar50 | mnar10 | mnar30 | mnar50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | LR | 5 | 0.8053 | 0.7971 | 0.7897 | 0.7726 | 0.8047 | 0.7974 | 0.7804 | 0.8027 | 0.7971 | 0.7932 |
| NAGALA | cLR | 5 | 0.8057 | 0.7977 | 0.7908 | 0.7742 | 0.8052 | 0.7981 | 0.7814 | 0.8033 | 0.7978 | 0.7942 |
| NAGALA | GAM | 5 | 0.8015 | 0.7901 | 0.7793 | 0.7604 | 0.8005 | 0.7914 | 0.7736 | 0.7964 | 0.7915 | 0.7874 |
| NAGALA | RF | 5 | 0.8047 | 0.7939 | 0.7878 | 0.7674 | 0.8041 | 0.7957 | 0.7814 | 0.8016 | 0.7989 | 0.7974 |
| NAGALA | XGB | 5 | 0.8008 | 0.7658 | 0.7096 | 0.6692 | 0.7554 | 0.6771 | 0.6184 | 0.7696 | 0.7197 | 0.6596 |
| NAGALA | XGB_mono | 5 | 0.7983 | 0.7745 | 0.7367 | 0.7071 | 0.7706 | 0.7196 | 0.6773 | 0.7767 | 0.7455 | 0.7010 |
| NAGALA | LGBM | 5 | 0.7964 | 0.7828 | 0.7716 | 0.7463 | 0.7969 | 0.7845 | 0.7682 | 0.7881 | 0.7736 | 0.7709 |
| NAGALA | LGBM_mono | 5 | 0.7965 | 0.7806 | 0.7672 | 0.7465 | 0.7954 | 0.7829 | 0.7675 | 0.7906 | 0.7788 | 0.7792 |
| NAGALA | M1 | 5 | 0.7981 | 0.7914 | 0.7872 | 0.7740 | 0.7989 | 0.7923 | 0.7792 | 0.7971 | 0.7922 | 0.7911 |
| NAGALA | M2 | 5 | 0.7985 | 0.7940 | 0.7906 | 0.7790 | 0.7991 | 0.7928 | 0.7813 | 0.7982 | 0.7904 | 0.7878 |
| NAGALA | CKINN | 5 | 0.8009 | 0.7963 | 0.7927 | 0.7820 | 0.8008 | 0.7942 | 0.7830 | 0.8001 | 0.7928 | 0.7907 |
| NAGALA | MI_LR | 1 | 0.8039 | 0.7985 | 0.7955 | 0.7918 | 0.8060 | 0.7951 | 0.7877 | 0.8014 | 0.8014 | 0.7978 |
| NAGALA | LR_aug | 1 | 0.8021 | 0.7964 | 0.7894 | 0.7827 | 0.8047 | 0.7979 | 0.7810 | 0.7978 | 0.7978 | 0.7893 |
| NAGALA | XGB_aug | 1 | 0.7960 | 0.7938 | 0.7864 | 0.7788 | 0.7970 | 0.7853 | 0.7695 | 0.7912 | 0.7899 | 0.7840 |
| RICH | LR | 5 | 0.8469 | 0.8448 | 0.8410 | 0.8373 | 0.8467 | 0.8446 | 0.8420 | 0.8464 | 0.8455 | 0.8438 |
| RICH | cLR | 5 | 0.8469 | 0.8448 | 0.8410 | 0.8374 | 0.8467 | 0.8447 | 0.8420 | 0.8465 | 0.8456 | 0.8439 |
| RICH | GAM | 5 | 0.8483 | 0.8462 | 0.8420 | 0.8373 | 0.8480 | 0.8458 | 0.8425 | 0.8468 | 0.8444 | 0.8410 |
| RICH | RF | 5 | 0.8430 | 0.8399 | 0.8351 | 0.8311 | 0.8426 | 0.8400 | 0.8368 | 0.8425 | 0.8415 | 0.8400 |
| RICH | XGB | 5 | 0.8481 | 0.8456 | 0.8412 | 0.8367 | 0.8477 | 0.8452 | 0.8419 | 0.8471 | 0.8454 | 0.8430 |
| RICH | XGB_mono | 5 | 0.8455 | 0.8432 | 0.8390 | 0.8347 | 0.8451 | 0.8429 | 0.8398 | 0.8448 | 0.8434 | 0.8414 |
| RICH | LGBM | 5 | 0.8479 | 0.8455 | 0.8411 | 0.8366 | 0.8476 | 0.8451 | 0.8421 | 0.8470 | 0.8452 | 0.8428 |
| RICH | LGBM_mono | 5 | 0.8474 | 0.8450 | 0.8406 | 0.8360 | 0.8470 | 0.8445 | 0.8415 | 0.8465 | 0.8448 | 0.8426 |
| RICH | M1 | 5 | 0.8494 | 0.8469 | 0.8424 | 0.8379 | 0.8490 | 0.8464 | 0.8428 | 0.8487 | 0.8473 | 0.8449 |
| RICH | M2 | 5 | 0.8486 | 0.8471 | 0.8441 | 0.8403 | 0.8484 | 0.8466 | 0.8440 | 0.8475 | 0.8462 | 0.8439 |
| RICH | CKINN | 5 | 0.8488 | 0.8473 | 0.8444 | 0.8407 | 0.8486 | 0.8467 | 0.8444 | 0.8477 | 0.8466 | 0.8445 |
| RICH | MI_LR | 1 | 0.8476 | 0.8453 | 0.8432 | 0.8395 | 0.8470 | 0.8457 | 0.8432 | 0.8469 | 0.8451 | 0.8428 |
| RICH | LR_aug | 1 | 0.8467 | 0.8447 | 0.8419 | 0.8374 | 0.8461 | 0.8446 | 0.8417 | 0.8464 | 0.8449 | 0.8430 |
| RICH | XGB_aug | 1 | 0.8477 | 0.8459 | 0.8434 | 0.8394 | 0.8474 | 0.8454 | 0.8429 | 0.8470 | 0.8451 | 0.8425 |
| NHANES | LR | 5 | 0.8142 | 0.8088 | 0.7978 | 0.7865 | 0.8109 | 0.8038 | 0.7934 | 0.8111 | 0.8026 | 0.7967 |
| NHANES | cLR | 5 | 0.8142 | 0.8088 | 0.7979 | 0.7866 | 0.8109 | 0.8039 | 0.7934 | 0.8111 | 0.8027 | 0.7968 |
| NHANES | GAM | 5 | 0.8271 | 0.8216 | 0.8108 | 0.7995 | 0.8241 | 0.8163 | 0.8065 | 0.8225 | 0.8120 | 0.8035 |
| NHANES | RF | 5 | 0.8111 | 0.8039 | 0.7900 | 0.7781 | 0.8067 | 0.7974 | 0.7872 | 0.8083 | 0.8014 | 0.7950 |
| NHANES | XGB | 5 | 0.8156 | 0.8102 | 0.7997 | 0.7911 | 0.8127 | 0.8056 | 0.7962 | 0.8116 | 0.8034 | 0.7972 |
| NHANES | XGB_mono | 5 | 0.8155 | 0.8102 | 0.8000 | 0.7917 | 0.8126 | 0.8058 | 0.7963 | 0.8116 | 0.8035 | 0.7972 |
| NHANES | LGBM | 5 | 0.8186 | 0.8132 | 0.8015 | 0.7922 | 0.8156 | 0.8077 | 0.7979 | 0.8143 | 0.8053 | 0.7982 |
| NHANES | LGBM_mono | 5 | 0.8187 | 0.8133 | 0.8024 | 0.7931 | 0.8160 | 0.8086 | 0.7986 | 0.8148 | 0.8068 | 0.7995 |
| NHANES | M1 | 5 | 0.8201 | 0.8134 | 0.8000 | 0.7877 | 0.8160 | 0.8070 | 0.7955 | 0.8170 | 0.8077 | 0.8020 |
| NHANES | M2 | 5 | 0.8252 | 0.8201 | 0.8096 | 0.7998 | 0.8224 | 0.8152 | 0.8060 | 0.8218 | 0.8099 | 0.8049 |
| NHANES | CKINN | 5 | 0.8256 | 0.8205 | 0.8101 | 0.8001 | 0.8227 | 0.8155 | 0.8063 | 0.8222 | 0.8111 | 0.8055 |
| NHANES | MI_LR | 1 | 0.8140 | 0.8090 | 0.8006 | 0.7903 | 0.8128 | 0.8075 | 0.7962 | 0.8106 | 0.8032 | 0.7997 |
| NHANES | LR_aug | 1 | 0.8151 | 0.8106 | 0.7983 | 0.7873 | 0.8122 | 0.8076 | 0.7937 | 0.8130 | 0.8028 | 0.8004 |
| NHANES | XGB_aug | 1 | 0.8166 | 0.8124 | 0.8057 | 0.7934 | 0.8151 | 0.8090 | 0.8016 | 0.8148 | 0.8045 | 0.8033 |

| cohort | scenario | comparator | runs | loss_ckinn | loss_comparator | ckinn_minus_comparator | lo | hi | comparator_trained_with_matching_augmentation |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | mcar30 | LR | 5 | 0.0082 | 0.0155 | -0.0074 | -0.0112 | -0.0038 | no |
| NAGALA | mcar30 | cLR | 5 | 0.0082 | 0.0150 | -0.0068 | -0.0105 | -0.0033 | no |
| NAGALA | mcar30 | GAM | 5 | 0.0082 | 0.0222 | -0.0141 | -0.0195 | -0.0090 | no |
| NAGALA | mcar30 | RF | 5 | 0.0082 | 0.0168 | -0.0087 | -0.0139 | -0.0028 | no |
| NAGALA | mcar30 | XGB | 5 | 0.0082 | 0.0912 | -0.0831 | -0.0936 | -0.0729 | no |
| NAGALA | mcar30 | XGB_mono | 5 | 0.0082 | 0.0616 | -0.0534 | -0.0608 | -0.0459 | no |
| NAGALA | mcar30 | LGBM | 5 | 0.0082 | 0.0248 | -0.0167 | -0.0239 | -0.0088 | no |
| NAGALA | mcar30 | LGBM_mono | 5 | 0.0082 | 0.0292 | -0.0211 | -0.0278 | -0.0135 | no |
| NAGALA | mcar30 | M1 | 5 | 0.0082 | 0.0109 | -0.0027 | -0.0059 | 0.0004 | no |
| NAGALA | mcar30 | M2 | 5 | 0.0082 | 0.0079 | 0.0002 | -0.0010 | 0.0013 | yes |
| NAGALA | mcar30 | MI_LR | 1 | 0.0093 | 0.0083 | 0.0009 | -0.0064 | 0.0071 | no |
| NAGALA | mcar30 | LR_aug | 1 | 0.0093 | 0.0127 | -0.0034 | -0.0110 | 0.0025 | yes |
| NAGALA | mcar30 | XGB_aug | 1 | 0.0093 | 0.0096 | -0.0003 | -0.0072 | 0.0062 | yes |
| RICH | mcar30 | LR | 5 | 0.0044 | 0.0059 | -0.0015 | -0.0020 | -0.0010 | no |
| RICH | mcar30 | cLR | 5 | 0.0044 | 0.0059 | -0.0015 | -0.0020 | -0.0010 | no |
| RICH | mcar30 | GAM | 5 | 0.0044 | 0.0063 | -0.0019 | -0.0023 | -0.0014 | no |
| RICH | mcar30 | RF | 5 | 0.0044 | 0.0079 | -0.0035 | -0.0042 | -0.0028 | no |
| RICH | mcar30 | XGB | 5 | 0.0044 | 0.0069 | -0.0025 | -0.0030 | -0.0019 | no |
| RICH | mcar30 | XGB_mono | 5 | 0.0044 | 0.0064 | -0.0020 | -0.0026 | -0.0015 | no |
| RICH | mcar30 | LGBM | 5 | 0.0044 | 0.0068 | -0.0024 | -0.0030 | -0.0018 | no |
| RICH | mcar30 | LGBM_mono | 5 | 0.0044 | 0.0069 | -0.0024 | -0.0030 | -0.0019 | no |
| RICH | mcar30 | M1 | 5 | 0.0044 | 0.0070 | -0.0025 | -0.0030 | -0.0020 | no |
| RICH | mcar30 | M2 | 5 | 0.0044 | 0.0045 | -0.0001 | -0.0003 | 0.0001 | yes |
| RICH | mcar30 | MI_LR | 1 | 0.0042 | 0.0045 | -0.0003 | -0.0014 | 0.0008 | no |
| RICH | mcar30 | LR_aug | 1 | 0.0042 | 0.0049 | -0.0007 | -0.0016 | 0.0002 | yes |
| RICH | mcar30 | XGB_aug | 1 | 0.0042 | 0.0042 | -0.0000 | -0.0008 | 0.0006 | yes |
| NHANES | mcar30 | LR | 5 | 0.0154 | 0.0164 | -0.0010 | -0.0029 | 0.0010 | no |
| NHANES | mcar30 | cLR | 5 | 0.0154 | 0.0163 | -0.0009 | -0.0028 | 0.0011 | no |
| NHANES | mcar30 | GAM | 5 | 0.0154 | 0.0164 | -0.0009 | -0.0021 | 0.0002 | no |
| NHANES | mcar30 | RF | 5 | 0.0154 | 0.0211 | -0.0056 | -0.0082 | -0.0030 | no |
| NHANES | mcar30 | XGB | 5 | 0.0154 | 0.0159 | -0.0005 | -0.0022 | 0.0011 | no |
| NHANES | mcar30 | XGB_mono | 5 | 0.0154 | 0.0155 | -0.0000 | -0.0017 | 0.0016 | no |
| NHANES | mcar30 | LGBM | 5 | 0.0154 | 0.0171 | -0.0017 | -0.0033 | 0.0000 | no |
| NHANES | mcar30 | LGBM_mono | 5 | 0.0154 | 0.0164 | -0.0009 | -0.0025 | 0.0007 | no |
| NHANES | mcar30 | M1 | 5 | 0.0154 | 0.0201 | -0.0047 | -0.0066 | -0.0028 | no |
| NHANES | mcar30 | M2 | 5 | 0.0154 | 0.0156 | -0.0002 | -0.0006 | 0.0002 | yes |
| NHANES | mcar30 | MI_LR | 1 | 0.0123 | 0.0134 | -0.0010 | -0.0053 | 0.0033 | no |
| NHANES | mcar30 | LR_aug | 1 | 0.0123 | 0.0168 | -0.0045 | -0.0078 | -0.0013 | yes |
| NHANES | mcar30 | XGB_aug | 1 | 0.0123 | 0.0109 | 0.0015 | -0.0009 | 0.0041 | yes |

RICH leave-one-site-out, pooled held-out rows:

| model | auroc_pooled_held_out | ckinn_minus_model | lo | hi |
|---|---|---|---|---|
| LR | 0.8441 | 0.0014 | -0.0001 | 0.0030 |
| XGB | 0.8448 | 0.0007 | -0.0009 | 0.0024 |
| XGB_mono | 0.8433 | 0.0022 | 0.0004 | 0.0042 |
| M1 | 0.8457 | -0.0002 | -0.0015 | 0.0013 |
| M2 | 0.8456 | -0.0001 | -0.0007 | 0.0006 |
| CKINN | 0.8455 | 0.0000 | 0.0000 | 0.0000 |

## 9. Constraint evidence (E; manipulation check P1)
| cohort | model | violation_rate | violation_min | violation_max | ig_direction_consistency |
|---|---|---|---|---|---|
| NAGALA | CKINN | 0.0020 | 0.0013 | 0.0036 | 0.9999 |
| NAGALA | M1 | 0.1429 | 0.1037 | 0.2027 | 0.8895 |
| NAGALA | M2 | 0.1263 | 0.1029 | 0.1571 | 0.9076 |
| NHANES | CKINN | 0.0005 | 0.0003 | 0.0008 | 1.0000 |
| NHANES | M1 | 0.0153 | 0.0092 | 0.0209 | 0.9908 |
| NHANES | M2 | 0.0098 | 0.0051 | 0.0136 | 0.9967 |
| RICH | CKINN | 0.0003 | 0.0000 | 0.0006 | 1.0000 |
| RICH | M1 | 0.0447 | 0.0277 | 0.0585 | 0.9670 |
| RICH | M2 | 0.0496 | 0.0422 | 0.0669 | 0.9576 |

| cohort | variant | auroc_minus_ckinn | lo | hi | brier_minus_ckinn | violation_rate_v1_table | violation_rate_own_table | violation_rate_dropped_feature |
|---|---|---|---|---|---|---|---|---|
| NAGALA | loco_age_years | -0.0014 | -0.0034 | 0.0009 | 0.0000 | 0.0029 | 0.0030 | 0.0024 |
| NAGALA | loco_bmi_kgm2 | -0.0013 | -0.0036 | 0.0008 | 0.0000 | 0.0095 | 0.0010 | 0.0606 |
| NAGALA | loco_waist_cm | -0.0009 | -0.0033 | 0.0016 | 0.0000 | 0.0036 | 0.0025 | 0.0102 |
| NAGALA | loco_sbp_mmHg | -0.0011 | -0.0062 | 0.0034 | 0.0000 | 0.0708 | 0.0020 | 0.4838 |
| NAGALA | loco_tg_mmol | 0.0013 | -0.0012 | 0.0036 | -0.0000 | 0.0071 | 0.0031 | 0.0312 |
| NAGALA | loco_hdl_mmol | 0.0011 | -0.0010 | 0.0029 | -0.0000 | 0.0075 | 0.0042 | 0.0274 |
| NAGALA | loco_exercise_regular | -0.0009 | -0.0032 | 0.0014 | 0.0000 | 0.0407 | 0.0044 | 0.2582 |
| NAGALA | wrong_age | -0.0164 | -0.0262 | -0.0050 | 0.0001 | 0.1316 | 0.0123 | -- |
| NAGALA | wrong_bmi | -0.0106 | -0.0179 | -0.0037 | 0.0001 | 0.1418 | 0.0032 | -- |
| NAGALA | wrong_all | -0.0562 | -0.0757 | -0.0375 | 0.0005 | 0.8606 | 0.0797 | -- |
| NAGALA | sign_core | -0.0008 | -0.0032 | 0.0017 | 0.0000 | 0.0476 | 0.0057 | -- |
| NAGALA | sign_ext | -0.0032 | -0.0073 | 0.0009 | 0.0001 | 0.0043 | 0.0070 | -- |
| NAGALA | lam_0.03 | -0.0011 | -0.0050 | 0.0025 | 0.0000 | 0.0928 | 0.0928 | -- |
| NAGALA | lam_0.1 | -0.0014 | -0.0050 | 0.0022 | 0.0000 | 0.0818 | 0.0818 | -- |
| NAGALA | lam_0.3 | -0.0007 | -0.0040 | 0.0027 | 0.0000 | 0.0609 | 0.0609 | -- |
| NAGALA | lam_1 | 0.0006 | -0.0033 | 0.0042 | 0.0000 | 0.0354 | 0.0354 | -- |
| NAGALA | lam_3 | -0.0020 | -0.0063 | 0.0018 | 0.0000 | 0.0253 | 0.0253 | -- |
| NAGALA | lam_10 | -0.0040 | -0.0078 | 0.0001 | 0.0001 | 0.0115 | 0.0115 | -- |
| NAGALA | lam_30 | -0.0046 | -0.0083 | -0.0004 | 0.0000 | 0.0058 | 0.0058 | -- |
| NAGALA | lam_100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0036 | 0.0036 | -- |
| RICH | loco_age_years | -0.0004 | -0.0010 | 0.0002 | 0.0000 | 0.0002 | 0.0001 | 0.0008 |
| RICH | loco_bmi_kgm2 | 0.0004 | -0.0000 | 0.0008 | -0.0000 | 0.0001 | 0.0000 | 0.0004 |
| RICH | loco_sbp_mmHg | -0.0001 | -0.0003 | 0.0002 | 0.0000 | 0.0130 | 0.0001 | 0.0714 |
| RICH | loco_family_history_dm | -0.0001 | -0.0004 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RICH | loco_tg_mmol | 0.0000 | -0.0007 | 0.0007 | -0.0000 | 0.0003 | 0.0001 | 0.0010 |
| RICH | loco_hdl_mmol | -0.0005 | -0.0010 | 0.0001 | 0.0000 | 0.0310 | 0.0001 | 0.3090 |
| RICH | wrong_age | -0.0522 | -0.0566 | -0.0472 | 0.0003 | 0.0874 | 0.0115 | -- |
| RICH | wrong_bmi | -0.0092 | -0.0113 | -0.0073 | 0.0001 | 0.1314 | 0.0042 | -- |
| RICH | wrong_all | -0.1201 | -0.1274 | -0.1125 | 0.0006 | 0.5807 | 0.0540 | -- |
| RICH | sign_core | 0.0002 | -0.0003 | 0.0008 | -0.0000 | 0.0299 | 0.0001 | -- |
| RICH | sign_ext | -0.0002 | -0.0008 | 0.0004 | 0.0000 | 0.0001 | 0.0018 | -- |
| RICH | lam_0.03 | -0.0001 | -0.0005 | 0.0003 | 0.0000 | 0.0261 | 0.0261 | -- |
| RICH | lam_0.1 | 0.0004 | -0.0002 | 0.0011 | -0.0000 | 0.0275 | 0.0275 | -- |
| RICH | lam_0.3 | -0.0001 | -0.0010 | 0.0007 | -0.0000 | 0.0195 | 0.0195 | -- |
| RICH | lam_1 | 0.0002 | -0.0005 | 0.0010 | 0.0000 | 0.0054 | 0.0054 | -- |
| RICH | lam_3 | 0.0001 | -0.0009 | 0.0009 | -0.0000 | 0.0046 | 0.0046 | -- |
| RICH | lam_10 | -0.0007 | -0.0017 | 0.0002 | 0.0000 | 0.0007 | 0.0007 | -- |
| RICH | lam_30 | 0.0002 | -0.0005 | 0.0010 | -0.0000 | 0.0004 | 0.0004 | -- |
| RICH | lam_100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| NHANES | loco_age_years | 0.0002 | -0.0001 | 0.0005 | -0.0000 | 0.0003 | 0.0004 | 0.0002 |
| NHANES | loco_bmi_kgm2 | -0.0008 | -0.0029 | 0.0014 | 0.0000 | 0.0004 | 0.0004 | 0.0008 |
| NHANES | loco_waist_cm | 0.0011 | -0.0003 | 0.0026 | -0.0000 | 0.0004 | 0.0005 | 0.0000 |
| NHANES | loco_sbp_mmHg | 0.0002 | -0.0004 | 0.0008 | -0.0000 | 0.0020 | 0.0004 | 0.0123 |
| NHANES | loco_family_history_dm | -0.0019 | -0.0039 | 0.0003 | 0.0001 | 0.0004 | 0.0004 | 0.0000 |
| NHANES | loco_tg_mmol | 0.0002 | -0.0004 | 0.0010 | -0.0000 | 0.0067 | 0.0003 | 0.0950 |
| NHANES | loco_hdl_mmol | -0.0001 | -0.0005 | 0.0003 | -0.0000 | 0.0013 | 0.0004 | 0.0073 |
| NHANES | loco_exercise_regular | -0.0001 | -0.0010 | 0.0009 | 0.0000 | 0.0029 | 0.0005 | 0.0222 |
| NHANES | wrong_age | -0.0270 | -0.0338 | -0.0209 | 0.0007 | 0.1224 | 0.0146 | -- |
| NHANES | wrong_bmi | -0.0010 | -0.0029 | 0.0011 | 0.0001 | 0.1366 | 0.0025 | -- |
| NHANES | wrong_all | -0.1096 | -0.1239 | -0.0973 | 0.0020 | 0.6978 | 0.1658 | -- |
| NHANES | sign_core | 0.0014 | 0.0003 | 0.0025 | -0.0000 | 0.0032 | 0.0005 | -- |
| NHANES | sign_ext | -0.0017 | -0.0038 | 0.0005 | 0.0001 | 0.0004 | 0.0017 | -- |
| NHANES | lam_0.03 | 0.0015 | 0.0002 | 0.0027 | -0.0001 | 0.0110 | 0.0110 | -- |
| NHANES | lam_0.1 | 0.0011 | -0.0002 | 0.0025 | -0.0000 | 0.0094 | 0.0094 | -- |
| NHANES | lam_0.3 | 0.0003 | -0.0005 | 0.0012 | -0.0000 | 0.0069 | 0.0069 | -- |
| NHANES | lam_1 | -0.0005 | -0.0025 | 0.0015 | 0.0000 | 0.0055 | 0.0055 | -- |
| NHANES | lam_3 | -0.0003 | -0.0023 | 0.0018 | -0.0000 | 0.0033 | 0.0033 | -- |
| NHANES | lam_10 | -0.0005 | -0.0025 | 0.0016 | 0.0000 | 0.0017 | 0.0017 | -- |
| NHANES | lam_30 | 0.0013 | -0.0002 | 0.0027 | -0.0000 | 0.0009 | 0.0009 | -- |
| NHANES | lam_100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 0.0005 | -- |

Empirical sign of each constrained predictor (adjusted logistic regression on the evaluation rows) and agreement of each model's held-out partial
dependence with the prior:

| cohort | model | feature | prior_sign | adjusted_coef | coef_lo | coef_hi | adjusted_evidence | pd_change_p10_to_p90 | pd_sign_agrees_with_prior |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | M2 | age_years | 1 | 0.4167 | 0.2861 | 0.5473 | supported | 0.0181 | yes |
| NAGALA | M2 | bmi_kgm2 | 1 | 0.3683 | 0.1283 | 0.6083 | supported | 0.0116 | yes |
| NAGALA | M2 | waist_cm | 1 | 0.1696 | -0.0943 | 0.4335 | uncertain | 0.0109 | yes |
| NAGALA | M2 | sbp_mmHg | 1 | -0.1210 | -0.4091 | 0.1672 | uncertain | -0.0015 | no |
| NAGALA | M2 | tg_mmol | 1 | 0.1920 | 0.0525 | 0.3314 | supported | 0.0103 | yes |
| NAGALA | M2 | hdl_mmol | -1 | -0.3832 | -0.5911 | -0.1753 | supported | -0.0119 | yes |
| NAGALA | CKINN | age_years | 1 | 0.4167 | 0.2861 | 0.5473 | supported | 0.0182 | yes |
| NAGALA | CKINN | bmi_kgm2 | 1 | 0.3683 | 0.1283 | 0.6083 | supported | 0.0108 | yes |
| NAGALA | CKINN | waist_cm | 1 | 0.1696 | -0.0943 | 0.4335 | uncertain | 0.0100 | yes |
| NAGALA | CKINN | sbp_mmHg | 1 | -0.1210 | -0.4091 | 0.1672 | uncertain | 0.0048 | yes |
| NAGALA | CKINN | tg_mmol | 1 | 0.1920 | 0.0525 | 0.3314 | supported | 0.0108 | yes |
| NAGALA | CKINN | hdl_mmol | -1 | -0.3832 | -0.5911 | -0.1753 | supported | -0.0118 | yes |
| RICH | M2 | age_years | 1 | 0.7937 | 0.7575 | 0.8298 | supported | 0.0414 | yes |
| RICH | M2 | bmi_kgm2 | 1 | 0.4676 | 0.4273 | 0.5080 | supported | 0.0167 | yes |
| RICH | M2 | sbp_mmHg | 1 | 0.1291 | 0.0845 | 0.1736 | supported | 0.0068 | yes |
| RICH | M2 | tg_mmol | 1 | 0.2406 | 0.2035 | 0.2777 | supported | 0.0095 | yes |
| RICH | M2 | hdl_mmol | -1 | -0.0200 | -0.0696 | 0.0295 | uncertain | -0.0014 | yes |
| RICH | CKINN | age_years | 1 | 0.7937 | 0.7575 | 0.8298 | supported | 0.0400 | yes |
| RICH | CKINN | bmi_kgm2 | 1 | 0.4676 | 0.4273 | 0.5080 | supported | 0.0167 | yes |
| RICH | CKINN | sbp_mmHg | 1 | 0.1291 | 0.0845 | 0.1736 | supported | 0.0081 | yes |
| RICH | CKINN | tg_mmol | 1 | 0.2406 | 0.2035 | 0.2777 | supported | 0.0103 | yes |
| RICH | CKINN | hdl_mmol | -1 | -0.0200 | -0.0696 | 0.0295 | uncertain | -0.0041 | yes |
| NHANES | M2 | age_years | 1 | 0.8322 | 0.7331 | 0.9313 | supported | 0.0697 | yes |
| NHANES | M2 | bmi_kgm2 | 1 | 0.3270 | 0.1953 | 0.4586 | supported | 0.0245 | yes |
| NHANES | M2 | waist_cm | 1 | 0.1698 | 0.0278 | 0.3117 | supported | 0.0227 | yes |
| NHANES | M2 | sbp_mmHg | 1 | 0.1438 | 0.0694 | 0.2182 | supported | 0.0183 | yes |
| NHANES | M2 | tg_mmol | 1 | 0.2259 | 0.1367 | 0.3152 | supported | 0.0185 | yes |
| NHANES | M2 | hdl_mmol | -1 | -0.3605 | -0.4525 | -0.2684 | supported | -0.0290 | yes |
| NHANES | CKINN | age_years | 1 | 0.8322 | 0.7331 | 0.9313 | supported | 0.0684 | yes |
| NHANES | CKINN | bmi_kgm2 | 1 | 0.3270 | 0.1953 | 0.4586 | supported | 0.0245 | yes |
| NHANES | CKINN | waist_cm | 1 | 0.1698 | 0.0278 | 0.3117 | supported | 0.0226 | yes |
| NHANES | CKINN | sbp_mmHg | 1 | 0.1438 | 0.0694 | 0.2182 | supported | 0.0179 | yes |
| NHANES | CKINN | tg_mmol | 1 | 0.2259 | 0.1367 | 0.3152 | supported | 0.0208 | yes |
| NHANES | CKINN | hdl_mmol | -1 | -0.3605 | -0.4525 | -0.2684 | supported | -0.0276 | yes |

| cohort | model | region | violation_rate | share_sparse |
|---|---|---|---|---|
| NAGALA | CKINN | held-out random rows | 0.0036 | -- |
| NAGALA | CKINN | held-out sparse rows | 0.0058 | 0.0502 |
| NAGALA | M1 | held-out random rows | 0.1037 | -- |
| NAGALA | M1 | held-out sparse rows | 0.1261 | 0.0502 |
| NAGALA | M2 | held-out random rows | 0.1029 | -- |
| NAGALA | M2 | held-out sparse rows | 0.1224 | 0.0502 |
| NAGALA-trained | CKINN | other cohort (RICH) | 0.0015 | -- |
| NAGALA-trained | CKINN | source (NAGALA) | 0.0003 | -- |
| NAGALA-trained | M1 | other cohort (RICH) | 0.1877 | -- |
| NAGALA-trained | M1 | source (NAGALA) | 0.1663 | -- |
| NAGALA-trained | M2 | other cohort (RICH) | 0.1598 | -- |
| NAGALA-trained | M2 | source (NAGALA) | 0.1432 | -- |
| NHANES | CKINN | held-out random rows | 0.0005 | -- |
| NHANES | CKINN | held-out sparse rows | 0.0013 | 0.0500 |
| NHANES | M1 | held-out random rows | 0.0137 | -- |
| NHANES | M1 | held-out sparse rows | 0.0161 | 0.0500 |
| NHANES | M2 | held-out random rows | 0.0124 | -- |
| NHANES | M2 | held-out sparse rows | 0.0137 | 0.0500 |
| RICH | CKINN | held-out random rows | 0.0000 | -- |
| RICH | CKINN | held-out sparse rows | 0.0004 | 0.0500 |
| RICH | M1 | held-out random rows | 0.0585 | -- |
| RICH | M1 | held-out sparse rows | 0.0670 | 0.0500 |
| RICH | M2 | held-out random rows | 0.0489 | -- |
| RICH | M2 | held-out sparse rows | 0.0541 | 0.0500 |
| RICH-trained | CKINN | other cohort (NAGALA) | 0.0009 | -- |
| RICH-trained | CKINN | source (RICH) | 0.0004 | -- |
| RICH-trained | M1 | other cohort (NAGALA) | 0.1054 | -- |
| RICH-trained | M1 | source (RICH) | 0.0623 | -- |
| RICH-trained | M2 | other cohort (NAGALA) | 0.1118 | -- |
| RICH-trained | M2 | source (RICH) | 0.0757 | -- |

## 10. Transport (P2) and secondary transport analyses (E)
| source | target | model | seeds | auroc_target_cv | auroc_external | transport_gap | gap_lo | gap_hi | abs_citl_external | slope_external | brier_external |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | RICH | M1 | 5 | 0.8476 | 0.8139 | 0.0337 | 0.0302 | 0.0373 | 0.4204 | 0.9542 | 0.0191 |
| NAGALA | RICH | M2 | 5 | 0.8479 | 0.8153 | 0.0326 | 0.0293 | 0.0360 | 0.5056 | 0.9736 | 0.0192 |
| NAGALA | RICH | CKINN | 5 | 0.8483 | 0.8267 | 0.0216 | 0.0190 | 0.0241 | 0.5998 | 1.0436 | 0.0192 |
| NAGALA | RICH | LR | 1 | 0.8446 | 0.8161 | 0.0285 | 0.0254 | 0.0319 | 0.6879 | 0.8785 | 0.0199 |
| NAGALA | RICH | XGB | 1 | 0.8465 | 0.8048 | 0.0417 | 0.0375 | 0.0460 | 0.5010 | 0.8647 | 0.0194 |
| NAGALA | RICH | XGB_mono | 1 | 0.8441 | 0.8201 | 0.0240 | 0.0205 | 0.0274 | 0.4695 | 0.8809 | 0.0194 |
| RICH | NAGALA | M1 | 5 | 0.8022 | 0.7883 | 0.0139 | 0.0014 | 0.0271 | 0.5557 | 0.8863 | 0.0229 |
| RICH | NAGALA | M2 | 5 | 0.8022 | 0.7889 | 0.0133 | 0.0001 | 0.0272 | 0.5445 | 0.8850 | 0.0229 |
| RICH | NAGALA | CKINN | 5 | 0.8019 | 0.7946 | 0.0072 | -0.0040 | 0.0186 | 0.5192 | 0.9152 | 0.0228 |
| RICH | NAGALA | LR | 1 | 0.8042 | 0.8044 | -0.0002 | -0.0098 | 0.0097 | 0.8147 | 1.1043 | 0.0230 |
| RICH | NAGALA | XGB | 1 | 0.7987 | 0.7901 | 0.0086 | -0.0060 | 0.0240 | 0.5067 | 0.9946 | 0.0228 |
| RICH | NAGALA | XGB_mono | 1 | 0.7987 | 0.7897 | 0.0090 | -0.0052 | 0.0236 | 0.6081 | 1.1053 | 0.0229 |

| source | target | model | events_in_recalibration_sample | brier_before | brier_after | citl_before | citl_after | slope_before | slope_after | auroc_unweighted_s0 | auroc_reweighted | abs_citl_unweighted_s0 | abs_citl_reweighted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | RICH | M1 | 14 | 0.0193 | 0.0188 | -0.4922 | 0.3348 | 0.9562 | 0.9722 | -- | -- | -- | -- |
| NAGALA | RICH | M2 | 14 | 0.0192 | 0.0188 | -0.6426 | 0.3444 | 1.1399 | 1.1117 | 0.8235 | 0.8189 | 0.6443 | 0.6255 |
| NAGALA | RICH | CKINN | 14 | 0.0193 | 0.0187 | -0.6824 | 0.3420 | 1.1464 | 1.1114 | 0.8242 | 0.8255 | 0.6841 | 0.6462 |
| NAGALA | RICH | LR | 14 | 0.0200 | 0.0188 | -0.6864 | 0.3015 | 0.8781 | 0.9323 | 0.8161 | 0.8285 | 0.6879 | 0.8210 |
| NAGALA | RICH | XGB_mono | 14 | 0.0194 | 0.0188 | -0.4679 | 0.3281 | 0.8809 | 1.0181 | -- | -- | -- | -- |
| RICH | NAGALA | M1 | 18 | 0.0233 | 0.0232 | 0.5614 | 0.3335 | 0.9107 | 0.7841 | -- | -- | -- | -- |
| RICH | NAGALA | M2 | 18 | 0.0234 | 0.0232 | 0.6996 | 0.3302 | 0.8154 | 0.7919 | 0.7954 | 0.7848 | 0.6766 | 0.3838 |
| RICH | NAGALA | CKINN | 18 | 0.0233 | 0.0232 | 0.6342 | 0.3308 | 0.7925 | 0.8009 | 0.7964 | 0.7729 | 0.6109 | 0.1579 |
| RICH | NAGALA | LR | 18 | 0.0235 | 0.0230 | 0.8373 | 0.3419 | 1.0943 | 0.8602 | 0.8044 | 0.7763 | 0.8147 | 0.6053 |
| RICH | NAGALA | XGB_mono | 18 | 0.0235 | 0.0232 | 0.6316 | 0.3539 | 1.0925 | 0.8310 | -- | -- | -- | -- |

| cohort | variant | auroc | auroc_minus_M2 | lo | hi | brier |
|---|---|---|---|---|---|---|
| NAGALA | M2 | 0.7927 | 0.0000 | 0.0000 | 0.0000 | 0.0225 |
| NAGALA | CKINN | 0.7945 | 0.0018 | -0.0029 | 0.0074 | 0.0225 |
| NAGALA | M2_pooled | 0.8047 | 0.0121 | 0.0030 | 0.0201 | 0.0223 |
| NAGALA | CKINN_pooled | 0.8016 | 0.0089 | -0.0003 | 0.0167 | 0.0223 |
| RICH | M2 | 0.8493 | 0.0000 | 0.0000 | 0.0000 | 0.0184 |
| RICH | CKINN | 0.8493 | -0.0001 | -0.0007 | 0.0006 | 0.0184 |
| RICH | M2_pooled | 0.8444 | -0.0050 | -0.0068 | -0.0032 | 0.0185 |
| RICH | CKINN_pooled | 0.8438 | -0.0055 | -0.0074 | -0.0038 | 0.0185 |

## 11. Sensitivity analyses (E)
| analysis | cohort | model | auroc | brier | ckinn_minus_M2 | ckinn_minus_M2_lo | ckinn_minus_M2_hi | ckinn_minus_M1 | ckinn_minus_M1_lo | ckinn_minus_M1_hi | citl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T3 | NAGALA | CKINN | 0.8795 | 0.0205 | 0.0039 | -0.0002 | 0.0078 | 0.0079 | 0.0009 | 0.0142 | -- |
| T3 | NAGALA | LR | 0.8794 | 0.0204 | -- | -- | -- | -- | -- | -- | -- |
| T3 | NAGALA | M1 | 0.8716 | 0.0206 | -- | -- | -- | -- | -- | -- | -- |
| T3 | NAGALA | M2 | 0.8756 | 0.0206 | -- | -- | -- | -- | -- | -- | -- |
| T3 | NAGALA | XGB | 0.8788 | 0.0204 | -- | -- | -- | -- | -- | -- | -- |
| T3 | NAGALA | XGB_mono | 0.8819 | 0.0205 | -- | -- | -- | -- | -- | -- | -- |
| T3 | RICH | CKINN | 0.9150 | 0.0159 | 0.0008 | 0.0001 | 0.0015 | 0.0014 | 0.0005 | 0.0023 | -- |
| T3 | RICH | LR | 0.9045 | 0.0162 | -- | -- | -- | -- | -- | -- | -- |
| T3 | RICH | M1 | 0.9136 | 0.0160 | -- | -- | -- | -- | -- | -- | -- |
| T3 | RICH | M2 | 0.9142 | 0.0159 | -- | -- | -- | -- | -- | -- | -- |
| T3 | RICH | XGB | 0.9142 | 0.0159 | -- | -- | -- | -- | -- | -- | -- |
| T3 | RICH | XGB_mono | 0.9114 | 0.0160 | -- | -- | -- | -- | -- | -- | -- |
| noFH | RICH | CKINN | 0.8462 | 0.0184 | -0.0012 | -0.0018 | -0.0004 | -- | -- | -- | -- |
| noFH | RICH | LR | 0.8450 | 0.0185 | -- | -- | -- | -- | -- | -- | -- |
| noFH | RICH | M2 | 0.8474 | 0.0184 | -- | -- | -- | -- | -- | -- | -- |
| v1label | NHANES | CKINN | 0.8240 | 0.0429 | 0.0002 | -0.0008 | 0.0013 | -- | -- | -- | -- |
| v1label | NHANES | LR | 0.8104 | 0.0434 | -- | -- | -- | -- | -- | -- | -- |
| v1label | NHANES | M2 | 0.8238 | 0.0429 | -- | -- | -- | -- | -- | -- | -- |
| survey | NHANES | CKINN | 0.8413 | 0.0284 | 0.0006 | -0.0003 | 0.0015 | 0.0053 | 0.0014 | 0.0089 | -0.2331 |
| survey | NHANES | LR | 0.8332 | 0.0286 | -- | -- | -- | -- | -- | -- | -0.2338 |
| survey | NHANES | M1 | 0.8359 | 0.0285 | -- | -- | -- | -- | -- | -- | -0.2124 |
| survey | NHANES | M2 | 0.8407 | 0.0284 | -- | -- | -- | -- | -- | -- | -0.2203 |
| survey | NHANES | XGB_mono | 0.8331 | 0.0285 | -- | -- | -- | -- | -- | -- | -0.2539 |

## 12. Post hoc additions for the 28-step protocol (PH; decided after the v2 results, no verdict depends on them)
Refits that repeat a main-run fit (largest absolute difference from the stored predictions; 0 means identical):

| cohort | model | max_abs_diff_vs_main_fit |
|---|---|---|
| NAGALA | CKINN | 0.000000 |
| NAGALA | GAM | 0.000000 |
| NAGALA | LGBM | 0.000000 |
| NAGALA | LGBM_mono | 0.000000 |
| NAGALA | LR | 0.000000 |
| NAGALA | LR_aug | 0.000000 |
| NAGALA | M1 | 0.000000 |
| NAGALA | M2 | 0.000000 |
| NAGALA | MI_LR | 0.000000 |
| NAGALA | RF | 0.000000 |
| NAGALA | SVM | 0.000000 |
| NAGALA | XGB | 0.000000 |
| NAGALA | XGB_aug | 0.000000 |
| NAGALA | XGB_mono | 0.000000 |
| NAGALA | cLR | 0.000000 |
| NHANES | CKINN | 0.000000 |
| NHANES | GAM | 0.000000 |
| NHANES | LGBM | 0.000000 |
| NHANES | LGBM_mono | 0.000000 |
| NHANES | LR | 0.000000 |
| NHANES | LR_aug | 0.000000 |
| NHANES | M1 | 0.000000 |
| NHANES | M2 | 0.000000 |
| NHANES | MI_LR | 0.000000 |
| NHANES | RF | 0.000000 |
| NHANES | SVM | 0.000000 |
| NHANES | XGB | 0.000000 |
| NHANES | XGB_aug | 0.000000 |
| NHANES | XGB_mono | 0.000000 |
| NHANES | cLR | 0.000000 |
| RICH | CKINN | 0.000000 |
| RICH | GAM | 0.000000 |
| RICH | LGBM | 0.000000 |
| RICH | LGBM_mono | 0.000000 |
| RICH | LR | 0.000000 |
| RICH | LR_aug | 0.000000 |
| RICH | M1 | 0.000000 |
| RICH | M2 | 0.000000 |
| RICH | MI_LR | 0.000000 |
| RICH | RF | 0.000000 |
| RICH | SVM | 0.000000 |
| RICH | XGB | 0.000000 |
| RICH | XGB_aug | 0.000000 |
| RICH | XGB_mono | 0.000000 |
| RICH | cLR | 0.000000 |

SVM (radial-kernel approximation, Platt scaling), run-averaged, and CKINN minus SVM:

| cohort | model | auroc | auroc_lo | auroc_hi | auprc | brier | slope | citl | ece | ckinn_minus_svm_auroc | diff_lo | diff_hi | auroc_loss_mcar30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | SVM | 0.7848 | 0.7598 | 0.8083 | 0.1048 | 0.0228 | 0.9748 | -0.0161 | 0.0045 | 0.0161 | 0.0049 | 0.0284 | 0.0252 |
| RICH | SVM | 0.8403 | 0.8340 | 0.8464 | 0.0935 | 0.0187 | 0.9992 | 0.0016 | 0.0054 | 0.0085 | 0.0064 | 0.0104 | 0.0073 |
| NHANES | SVM | 0.8125 | 0.8016 | 0.8248 | 0.1589 | 0.0378 | 1.0044 | 0.0057 | 0.0066 | 0.0131 | 0.0075 | 0.0186 | 0.0245 |

Violation rate for every model (log-odds change against the prior after +0.25 SD or a 0-to-1 change; Wilson intervals):

| cohort | model | n | violations | violation_rate | lo | hi |
|---|---|---|---|---|---|---|
| NAGALA | CKINN | 34,090 | 109 | 0.0032 | 0.0027 | 0.0039 |
| NAGALA | GAM | 34,090 | 11,115 | 0.3260 | 0.3211 | 0.3310 |
| NAGALA | LGBM | 34,090 | 5,176 | 0.1518 | 0.1481 | 0.1557 |
| NAGALA | LGBM_mono | 34,090 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NAGALA | LR | 34,090 | 8,252 | 0.2421 | 0.2375 | 0.2466 |
| NAGALA | LR_aug | 34,090 | 4,459 | 0.1308 | 0.1273 | 0.1344 |
| NAGALA | M1 | 34,090 | 3,309 | 0.0971 | 0.0940 | 0.1003 |
| NAGALA | M2 | 34,090 | 3,215 | 0.0943 | 0.0913 | 0.0975 |
| NAGALA | MI_LR | 34,090 | 8,250 | 0.2420 | 0.2375 | 0.2466 |
| NAGALA | RF | 34,090 | 9,536 | 0.2797 | 0.2750 | 0.2845 |
| NAGALA | SVM | 34,090 | 12,974 | 0.3806 | 0.3754 | 0.3857 |
| NAGALA | XGB | 34,090 | 9,284 | 0.2723 | 0.2676 | 0.2771 |
| NAGALA | XGB_aug | 34,090 | 8,040 | 0.2358 | 0.2314 | 0.2404 |
| NAGALA | XGB_mono | 34,090 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NAGALA | cLR | 34,090 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NHANES | CKINN | 32,731 | 16 | 0.0005 | 0.0003 | 0.0008 |
| NHANES | GAM | 32,731 | 2,396 | 0.0732 | 0.0704 | 0.0761 |
| NHANES | LGBM | 32,731 | 986 | 0.0301 | 0.0283 | 0.0320 |
| NHANES | LGBM_mono | 32,731 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NHANES | LR | 32,731 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NHANES | LR_aug | 32,731 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NHANES | M1 | 32,731 | 422 | 0.0129 | 0.0117 | 0.0142 |
| NHANES | M2 | 32,731 | 397 | 0.0121 | 0.0110 | 0.0134 |
| NHANES | MI_LR | 32,731 | 2,886 | 0.0882 | 0.0851 | 0.0913 |
| NHANES | RF | 32,731 | 8,155 | 0.2492 | 0.2445 | 0.2539 |
| NHANES | SVM | 32,731 | 5,912 | 0.1806 | 0.1765 | 0.1848 |
| NHANES | XGB | 32,731 | 228 | 0.0070 | 0.0061 | 0.0079 |
| NHANES | XGB_aug | 32,731 | 117 | 0.0036 | 0.0030 | 0.0043 |
| NHANES | XGB_mono | 32,731 | 0 | 0.0000 | 0.0000 | 0.0001 |
| NHANES | cLR | 32,731 | 0 | 0.0000 | 0.0000 | 0.0001 |
| RICH | CKINN | 27,513 | 2 | 0.0001 | 0.0000 | 0.0003 |
| RICH | GAM | 27,513 | 2,661 | 0.0967 | 0.0933 | 0.1003 |
| RICH | LGBM | 27,513 | 238 | 0.0087 | 0.0076 | 0.0098 |
| RICH | LGBM_mono | 27,513 | 0 | 0.0000 | 0.0000 | 0.0001 |
| RICH | LR | 27,513 | 536 | 0.0195 | 0.0179 | 0.0212 |
| RICH | LR_aug | 27,513 | 536 | 0.0195 | 0.0179 | 0.0212 |
| RICH | M1 | 27,513 | 1,492 | 0.0542 | 0.0516 | 0.0570 |
| RICH | M2 | 27,513 | 1,228 | 0.0446 | 0.0423 | 0.0471 |
| RICH | MI_LR | 27,513 | 2,514 | 0.0914 | 0.0880 | 0.0948 |
| RICH | RF | 27,513 | 1,443 | 0.0524 | 0.0499 | 0.0551 |
| RICH | SVM | 27,513 | 6,110 | 0.2221 | 0.2172 | 0.2270 |
| RICH | XGB | 27,513 | 153 | 0.0056 | 0.0047 | 0.0065 |
| RICH | XGB_aug | 27,513 | 163 | 0.0059 | 0.0051 | 0.0069 |
| RICH | XGB_mono | 27,513 | 0 | 0.0000 | 0.0000 | 0.0001 |
| RICH | cLR | 27,513 | 0 | 0.0000 | 0.0000 | 0.0001 |

BMI set to 24 and to 35 in the same held-out people: share whose predicted risk is lower at 35:

| cohort | model | pairs | risk_lower_at_bmi35 | share_lower | lo | hi | share_unchanged |
|---|---|---|---|---|---|---|---|
| NAGALA | LR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | cLR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | GAM | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | RF | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | XGB | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | XGB_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | LGBM | 5,000 | 18 | 0.0036 | 0.0023 | 0.0057 | 0.0000 |
| NAGALA | LGBM_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | M1 | 5,000 | 13 | 0.0026 | 0.0015 | 0.0044 | 0.0000 |
| NAGALA | M2 | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | CKINN | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | MI_LR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | LR_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | XGB_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NAGALA | SVM | 5,000 | 27 | 0.0054 | 0.0037 | 0.0078 | 0.0000 |
| RICH | LR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | cLR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | GAM | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | RF | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | XGB | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | XGB_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | LGBM | 5,000 | 1 | 0.0002 | 0.0000 | 0.0011 | 0.0000 |
| RICH | LGBM_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | M1 | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | M2 | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | CKINN | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | MI_LR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | LR_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | XGB_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| RICH | SVM | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | LR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | cLR | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | GAM | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | RF | 5,000 | 300 | 0.0600 | 0.0537 | 0.0669 | 0.0000 |
| NHANES | XGB | 5,000 | 1 | 0.0002 | 0.0000 | 0.0011 | 0.0000 |
| NHANES | XGB_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | LGBM | 5,000 | 50 | 0.0100 | 0.0076 | 0.0132 | 0.0000 |
| NHANES | LGBM_mono | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | M1 | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | M2 | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | CKINN | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | MI_LR | 5,000 | 4 | 0.0008 | 0.0003 | 0.0021 | 0.0000 |
| NHANES | LR_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | XGB_aug | 5,000 | 0 | 0.0000 | 0.0000 | 0.0008 | 0.0000 |
| NHANES | SVM | 5,000 | 91 | 0.0182 | 0.0148 | 0.0223 | 0.0000 |

Variants (each against the same model fitted without the change, repeat 0):

| cohort | variant | model | auroc_diff | auroc_diff_lo | auroc_diff_hi | brier_diff | citl_diff | tpr_sex_gap_diff | tpr_age_range_diff |
|---|---|---|---|---|---|---|---|---|---|
| NAGALA | dbp_constrained | CKINN | -0.0019 | -0.0048 | 0.0008 | 0.0001 | 0.0188 | -0.0303 | -0.0333 |
| NAGALA | wrong_hdl | CKINN | -0.0036 | -0.0110 | 0.0031 | -0.0000 | -0.0559 | -0.0195 | -0.0500 |
| NAGALA | m5_group_weights | CKINN | -0.0114 | -0.0189 | -0.0038 | 0.0002 | -0.0060 | 0.0257 | 0.0037 |
| NAGALA | class_weights | LR | -0.0024 | -0.0068 | 0.0010 | 0.1524 | -3.7233 | 0.0108 | 0.0407 |
| NAGALA | class_weights | M2 | -0.0004 | -0.0042 | 0.0039 | 0.1615 | -3.6997 | -0.0758 | -0.0574 |
| NAGALA | class_weights | CKINN | 0.0010 | -0.0038 | 0.0065 | 0.1676 | -3.8045 | -0.0735 | -0.0907 |
| NAGALA | tg_hdl_ratio | LR | 0.0006 | -0.0008 | 0.0019 | 0.0000 | -0.0031 | 0.0065 | 0.0000 |
| NAGALA | tg_hdl_ratio | M2 | 0.0015 | -0.0045 | 0.0078 | 0.0000 | 0.0335 | -0.0389 | -0.0370 |
| NAGALA | tg_hdl_ratio | CKINN | -0.0017 | -0.0070 | 0.0033 | 0.0001 | 0.0108 | -0.0303 | -0.0204 |
| RICH | dbp_constrained | CKINN | -0.0009 | -0.0017 | -0.0001 | 0.0000 | 0.0436 | -0.0038 | 0.0102 |
| RICH | wrong_hdl | CKINN | 0.0002 | -0.0005 | 0.0009 | -0.0000 | 0.0022 | -0.0047 | 0.0001 |
| RICH | m5_group_weights | CKINN | -0.0014 | -0.0025 | -0.0002 | 0.0000 | 0.0200 | 0.0012 | -0.0251 |
| RICH | class_weights | LR | 0.0004 | -0.0001 | 0.0008 | 0.1492 | -4.0057 | 0.0021 | 0.0020 |
| RICH | class_weights | M2 | -0.0011 | -0.0022 | -0.0001 | 0.1546 | -3.8850 | 0.0045 | -0.0026 |
| RICH | class_weights | CKINN | -0.0017 | -0.0029 | -0.0004 | 0.1566 | -3.8834 | -0.0028 | 0.0101 |
| RICH | tg_hdl_ratio | LR | -0.0001 | -0.0002 | -0.0000 | -0.0000 | 0.0018 | 0.0000 | -0.0016 |
| RICH | tg_hdl_ratio | M2 | 0.0001 | -0.0007 | 0.0009 | 0.0000 | -0.0108 | 0.0010 | -0.0071 |
| RICH | tg_hdl_ratio | CKINN | -0.0009 | -0.0019 | 0.0001 | 0.0000 | 0.0177 | 0.0110 | -0.0038 |
| NHANES | dbp_constrained | CKINN | -0.0003 | -0.0018 | 0.0012 | 0.0000 | -0.0245 | 0.0015 | -0.0037 |
| NHANES | wrong_hdl | CKINN | -0.0080 | -0.0124 | -0.0036 | 0.0002 | 0.0039 | 0.0023 | 0.0298 |
| NHANES | m5_group_weights | CKINN | 0.0011 | -0.0000 | 0.0023 | -0.0000 | 0.0360 | 0.0126 | -0.0077 |
| NHANES | class_weights | LR | -0.0000 | -0.0013 | 0.0012 | 0.1457 | -3.1650 | 0.0021 | 0.0064 |
| NHANES | class_weights | M2 | -0.0039 | -0.0066 | -0.0014 | 0.1437 | -3.1035 | -0.0054 | -0.0191 |
| NHANES | class_weights | CKINN | -0.0027 | -0.0051 | -0.0001 | 0.1453 | -3.1074 | 0.0023 | -0.0030 |
| NHANES | tg_hdl_ratio | LR | -0.0001 | -0.0003 | 0.0001 | 0.0000 | 0.0007 | -0.0002 | 0.0137 |
| NHANES | tg_hdl_ratio | M2 | -0.0021 | -0.0045 | 0.0003 | 0.0001 | 0.0269 | 0.0150 | -0.0311 |
| NHANES | tg_hdl_ratio | CKINN | -0.0011 | -0.0038 | 0.0014 | 0.0000 | 0.0363 | 0.0321 | -0.0271 |
| NHANES | history_questions | LR | 0.0059 | 0.0031 | 0.0090 | -0.0002 | -0.0004 | 0.0199 | -0.0455 |
| NHANES | history_questions | M2 | 0.0117 | 0.0076 | 0.0157 | -0.0005 | -0.0149 | 0.0305 | -0.0438 |
| NHANES | history_questions | CKINN | 0.0103 | 0.0059 | 0.0147 | -0.0004 | 0.0448 | 0.0288 | -0.0338 |

Pooled NAGALA plus RICH training with a cohort indicator:

| cohort | model | auroc_single | auroc_pooled | auroc_pooled_indicator | indicator_minus_pooled | lo | hi |
|---|---|---|---|---|---|---|---|
| NAGALA | M2 | 0.7927 | 0.8047 | 0.8061 | 0.0014 | -0.0049 | 0.0075 |
| NAGALA | CKINN | 0.7945 | 0.8016 | 0.8057 | 0.0041 | -0.0012 | 0.0095 |
| RICH | M2 | 0.8493 | 0.8444 | 0.8440 | -0.0004 | -0.0017 | 0.0010 |
| RICH | CKINN | 0.8493 | 0.8438 | 0.8457 | 0.0019 | 0.0005 | 0.0033 |

Train-by-test AUROC matrix (seed 0):

| model | within_RICH_cv | NAGALA_to_RICH | gap_NAGALA_to_RICH | gap_NAGALA_to_RICH_lo | gap_NAGALA_to_RICH_hi | within_NAGALA_cv | RICH_to_NAGALA | gap_RICH_to_NAGALA | gap_RICH_to_NAGALA_lo | gap_RICH_to_NAGALA_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| LR | 0.8446 | 0.8161 | 0.0285 | 0.0256 | 0.0318 | 0.8042 | 0.8044 | -0.0002 | -0.0103 | 0.0096 |
| cLR | 0.8446 | 0.8223 | 0.0223 | 0.0199 | 0.0250 | 0.8047 | 0.8044 | 0.0003 | -0.0093 | 0.0090 |
| GAM | 0.8459 | 0.8040 | 0.0419 | 0.0379 | 0.0460 | 0.7996 | 0.7933 | 0.0064 | -0.0081 | 0.0217 |
| RF | 0.8428 | 0.7933 | 0.0495 | 0.0453 | 0.0542 | 0.8038 | 0.7856 | 0.0182 | 0.0009 | 0.0358 |
| XGB | 0.8465 | 0.8048 | 0.0417 | 0.0378 | 0.0461 | 0.7987 | 0.7901 | 0.0086 | -0.0063 | 0.0248 |
| XGB_mono | 0.8441 | 0.8201 | 0.0240 | 0.0207 | 0.0275 | 0.7987 | 0.7897 | 0.0090 | -0.0053 | 0.0245 |
| LGBM | 0.8465 | 0.8035 | 0.0430 | 0.0387 | 0.0478 | 0.7976 | 0.7882 | 0.0093 | -0.0059 | 0.0261 |
| LGBM_mono | 0.8458 | 0.8202 | 0.0256 | 0.0217 | 0.0294 | 0.7950 | 0.7884 | 0.0067 | -0.0075 | 0.0234 |
| SVM | 0.8400 | 0.8004 | 0.0395 | 0.0354 | 0.0443 | 0.7796 | 0.7707 | 0.0089 | -0.0128 | 0.0316 |
| MI_LR | 0.8451 | 0.8148 | 0.0303 | 0.0270 | 0.0338 | 0.8042 | 0.8030 | 0.0012 | -0.0097 | 0.0109 |
| M1 | 0.8473 | 0.8139 | 0.0334 | 0.0297 | 0.0378 | 0.8031 | 0.7926 | 0.0105 | -0.0023 | 0.0237 |
| M2 | 0.8472 | 0.8235 | 0.0237 | 0.0206 | 0.0274 | 0.8014 | 0.7954 | 0.0061 | -0.0071 | 0.0205 |
| CKINN | 0.8481 | 0.8242 | 0.0240 | 0.0211 | 0.0272 | 0.8023 | 0.7964 | 0.0060 | -0.0067 | 0.0184 |

Intercept-only recalibration in the target: the intercept is estimated on a random sample of 1,000 target participants (target labels used; the same sample for every model) and evaluated on the remaining target participants. With few events in that sample the estimated shift is imprecise, so calibration-in-the-large after recalibration need not be near zero; the calibration-sample columns show why.

| source | target | model | intercept_shift | brier_before | brier_after | citl_before | citl_after | mean_predicted_after | observed_rate | calibration_n | calibration_events | calibration_observed_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | RICH | LR | -0.9827 | 0.0200 | 0.0188 | -0.6864 | 0.2963 | 0.0150 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | LR | 0.5151 | 0.0235 | 0.0231 | 0.8373 | 0.3223 | 0.0181 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | cLR | -0.9879 | 0.0199 | 0.0187 | -0.6957 | 0.2921 | 0.0150 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | cLR | 0.5162 | 0.0235 | 0.0231 | 0.8385 | 0.3223 | 0.0181 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | GAM | -0.5876 | 0.0188 | 0.0189 | -0.2584 | 0.3292 | 0.0144 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | GAM | 0.1995 | 0.0233 | 0.0232 | 0.5320 | 0.3325 | 0.0180 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | RF | -0.8473 | 0.0191 | 0.0189 | -0.5060 | 0.3413 | 0.0142 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | RF | 0.2370 | 0.0235 | 0.0233 | 0.5745 | 0.3375 | 0.0179 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | XGB | -0.8288 | 0.0195 | 0.0188 | -0.4994 | 0.3294 | 0.0145 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | XGB | 0.1937 | 0.0233 | 0.0232 | 0.5304 | 0.3366 | 0.0179 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | XGB_mono | -0.7918 | 0.0194 | 0.0188 | -0.4679 | 0.3240 | 0.0146 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | XGB_mono | 0.2994 | 0.0235 | 0.0233 | 0.6316 | 0.3321 | 0.0179 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | LGBM | -0.8775 | 0.0198 | 0.0189 | -0.5511 | 0.3264 | 0.0146 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | LGBM | 0.1704 | 0.0233 | 0.0232 | 0.5209 | 0.3504 | 0.0177 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | LGBM_mono | -1.2325 | 0.0217 | 0.0190 | -0.8537 | 0.3788 | 0.0140 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | LGBM_mono | 0.2082 | 0.0234 | 0.0233 | 0.5479 | 0.3397 | 0.0179 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | SVM | -0.8532 | 0.0193 | 0.0189 | -0.4971 | 0.3561 | 0.0141 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | SVM | 0.2594 | 0.0235 | 0.0233 | 0.5936 | 0.3342 | 0.0179 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | MI_LR | -1.0678 | 0.0204 | 0.0188 | -0.7837 | 0.2840 | 0.0151 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | MI_LR | 0.4919 | 0.0235 | 0.0231 | 0.8082 | 0.3163 | 0.0182 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | M1 | -0.8268 | 0.0193 | 0.0188 | -0.4922 | 0.3346 | 0.0144 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | M1 | 0.2387 | 0.0233 | 0.0232 | 0.5614 | 0.3226 | 0.0182 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | M2 | -0.9873 | 0.0192 | 0.0188 | -0.6426 | 0.3448 | 0.0142 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | M2 | 0.3719 | 0.0234 | 0.0232 | 0.6996 | 0.3277 | 0.0182 | 0.0246 | 1,000 | 18 | 0.0180 |
| NAGALA | RICH | CKINN | -1.0250 | 0.0193 | 0.0188 | -0.6824 | 0.3426 | 0.0142 | 0.0197 | 1,000 | 14 | 0.0140 |
| RICH | NAGALA | CKINN | 0.3024 | 0.0233 | 0.0232 | 0.6342 | 0.3318 | 0.0181 | 0.0246 | 1,000 | 18 | 0.0180 |

Older and lean subgroup (age 65 or more, BMI below 20) across lambda; not estimable below 20 events:

| cohort | lambda | n | events | estimable | mean_predicted | observed_rate | auroc | auroc_lo | auroc_hi | citl |
|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | 0.0000 | 32 | 1 | no | 0.0172 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 0.0300 | 32 | 1 | no | 0.0179 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 0.1000 | 32 | 1 | no | 0.0178 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 0.3000 | 32 | 1 | no | 0.0178 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 1.0000 | 32 | 1 | no | 0.0168 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 3.0000 | 32 | 1 | no | 0.0163 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 10.0000 | 32 | 1 | no | 0.0158 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 30.0000 | 32 | 1 | no | 0.0155 | 0.0312 | -- | -- | -- | -- |
| NAGALA | 100.0000 | 32 | 1 | no | 0.0173 | 0.0312 | -- | -- | -- | -- |
| RICH | 0.0000 | 902 | 28 | yes | 0.0347 | 0.0310 | 0.4984 | 0.3722 | 0.6129 | -0.1167 |
| RICH | 0.0300 | 902 | 28 | yes | 0.0345 | 0.0310 | 0.4939 | 0.3743 | 0.6038 | -0.1114 |
| RICH | 0.1000 | 902 | 28 | yes | 0.0334 | 0.0310 | 0.5011 | 0.3775 | 0.6106 | -0.0762 |
| RICH | 0.3000 | 902 | 28 | yes | 0.0352 | 0.0310 | 0.4926 | 0.3685 | 0.6036 | -0.1319 |
| RICH | 1.0000 | 902 | 28 | yes | 0.0351 | 0.0310 | 0.5144 | 0.3954 | 0.6255 | -0.1275 |
| RICH | 3.0000 | 902 | 28 | yes | 0.0352 | 0.0310 | 0.5262 | 0.4070 | 0.6394 | -0.1310 |
| RICH | 10.0000 | 902 | 28 | yes | 0.0368 | 0.0310 | 0.5189 | 0.3969 | 0.6315 | -0.1771 |
| RICH | 30.0000 | 902 | 28 | yes | 0.0346 | 0.0310 | 0.5272 | 0.4125 | 0.6387 | -0.1146 |
| RICH | 100.0000 | 902 | 28 | yes | 0.0336 | 0.0310 | 0.4992 | 0.3772 | 0.6104 | -0.0823 |
| NHANES | 0.0000 | 215 | 3 | no | 0.0200 | 0.0140 | -- | -- | -- | -- |
| NHANES | 0.0300 | 215 | 3 | no | 0.0196 | 0.0140 | -- | -- | -- | -- |
| NHANES | 0.1000 | 215 | 3 | no | 0.0206 | 0.0140 | -- | -- | -- | -- |
| NHANES | 0.3000 | 215 | 3 | no | 0.0213 | 0.0140 | -- | -- | -- | -- |
| NHANES | 1.0000 | 215 | 3 | no | 0.0204 | 0.0140 | -- | -- | -- | -- |
| NHANES | 3.0000 | 215 | 3 | no | 0.0207 | 0.0140 | -- | -- | -- | -- |
| NHANES | 10.0000 | 215 | 3 | no | 0.0209 | 0.0140 | -- | -- | -- | -- |
| NHANES | 30.0000 | 215 | 3 | no | 0.0212 | 0.0140 | -- | -- | -- | -- |
| NHANES | 100.0000 | 215 | 3 | no | 0.0214 | 0.0140 | -- | -- | -- | -- |

RQ7, participants with baseline fasting glucose below 5.6 mmol/L:

| cohort | tier | model | runs | n | events | auroc | auroc_lo | auroc_hi | tpr_at_fpr_0.10 | citl | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | T2 | LR | 5 | 10,243 | 112 | 0.7957 | 0.7489 | 0.8355 | 0.4857 | -0.6437 | 0.0108 |
| NAGALA | T2 | XGB_mono | 5 | 10,243 | 112 | 0.7857 | 0.7407 | 0.8260 | 0.4607 | -0.6228 | 0.0111 |
| NAGALA | T2 | M1 | 5 | 10,243 | 112 | 0.7894 | 0.7463 | 0.8283 | 0.4679 | -0.6342 | 0.0108 |
| NAGALA | T2 | M2 | 5 | 10,243 | 112 | 0.7930 | 0.7522 | 0.8312 | 0.4804 | -0.6518 | 0.0109 |
| NAGALA | T2 | CKINN | 5 | 10,243 | 112 | 0.7927 | 0.7505 | 0.8295 | 0.4768 | -0.6347 | 0.0108 |
| NAGALA | T3 | LR | 1 | 10,243 | 112 | 0.8320 | 0.7899 | 0.8737 | 0.5804 | -0.0573 | 0.0100 |
| NAGALA | T3 | XGB_mono | 1 | 10,243 | 112 | 0.8258 | 0.7761 | 0.8711 | 0.6071 | 0.0512 | 0.0101 |
| NAGALA | T3 | M1 | 1 | 10,243 | 112 | 0.8139 | 0.7695 | 0.8598 | 0.5714 | -0.1569 | 0.0101 |
| NAGALA | T3 | M2 | 1 | 10,243 | 112 | 0.8212 | 0.7778 | 0.8653 | 0.5893 | -0.1504 | 0.0101 |
| NAGALA | T3 | CKINN | 1 | 10,243 | 112 | 0.8292 | 0.7825 | 0.8720 | 0.6071 | -0.0924 | 0.0100 |
| RICH | T2 | LR | 5 | 148,504 | 1,029 | 0.8305 | 0.8196 | 0.8421 | 0.4947 | -0.9280 | 0.0072 |
| RICH | T2 | XGB_mono | 5 | 148,504 | 1,029 | 0.8250 | 0.8137 | 0.8371 | 0.4663 | -0.9325 | 0.0071 |
| RICH | T2 | M1 | 5 | 148,504 | 1,029 | 0.8315 | 0.8201 | 0.8429 | 0.5005 | -0.9121 | 0.0071 |
| RICH | T2 | M2 | 5 | 148,504 | 1,029 | 0.8297 | 0.8184 | 0.8416 | 0.4935 | -0.9342 | 0.0072 |
| RICH | T2 | CKINN | 5 | 148,504 | 1,029 | 0.8301 | 0.8186 | 0.8421 | 0.4927 | -0.9342 | 0.0072 |
| RICH | T3 | LR | 1 | 148,504 | 1,029 | 0.8092 | 0.7960 | 0.8216 | 0.4966 | -0.0319 | 0.0068 |
| RICH | T3 | XGB_mono | 1 | 148,504 | 1,029 | 0.8330 | 0.8215 | 0.8447 | 0.5005 | -0.0798 | 0.0068 |
| RICH | T3 | M1 | 1 | 148,504 | 1,029 | 0.8386 | 0.8268 | 0.8500 | 0.5219 | 0.0791 | 0.0067 |
| RICH | T3 | M2 | 1 | 148,504 | 1,029 | 0.8387 | 0.8276 | 0.8496 | 0.5335 | 0.0869 | 0.0068 |
| RICH | T3 | CKINN | 1 | 148,504 | 1,029 | 0.8415 | 0.8302 | 0.8521 | 0.5355 | 0.0729 | 0.0067 |

RICH one-step progression (baseline FPG below 5.6 mmol/L; outcome FPG of 5.6 mmol/L or more at the final visit, or diabetes):

| model | n | events | auroc | auroc_lo | auroc_hi | auprc | brier | slope | citl | ckinn_minus_M2 | ckinn_minus_M2_lo | ckinn_minus_M2_hi | ckinn_minus_M1 | ckinn_minus_M1_lo | ckinn_minus_M1_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | 148,504 | 17,635 | 0.7056 | 0.7017 | 0.7096 | 0.2410 | 0.0984 | 1.0033 | 0.0005 | -- | -- | -- | -- | -- | -- |
| XGB_mono | 148,504 | 17,635 | 0.7079 | 0.7042 | 0.7119 | 0.2461 | 0.0982 | 1.1737 | 0.0008 | -- | -- | -- | -- | -- | -- |
| M1 | 148,504 | 17,635 | 0.7084 | 0.7046 | 0.7125 | 0.2448 | 0.0981 | 0.9982 | 0.0295 | -- | -- | -- | -- | -- | -- |
| M2 | 148,504 | 17,635 | 0.7118 | 0.7080 | 0.7156 | 0.2573 | 0.0975 | 0.9814 | 0.0292 | -- | -- | -- | -- | -- | -- |
| CKINN | 148,504 | 17,635 | 0.7122 | 0.7083 | 0.7162 | 0.2547 | 0.0976 | 0.9535 | -0.0024 | 0.0004 | -0.0004 | 0.0013 | 0.0038 | 0.0027 | 0.0049 |

SHAP direction against the priors (number and share of constrained predictors whose SHAP dependence has the prior's sign; per predictor in table PH16) and agreement with integrated gradients:

| cohort | model | n_constrained | n_estimable | n_agree | share_agree |
|---|---|---|---|---|---|
| NAGALA | CKINN | 7 | 7 | 7 | 1.000 |
| NAGALA | LR | 7 | 7 | 5 | 0.714 |
| NAGALA | M1 | 7 | 7 | 6 | 0.857 |
| NAGALA | M2 | 7 | 7 | 7 | 1.000 |
| NAGALA | XGB | 7 | 7 | 5 | 0.714 |
| NAGALA | XGB_mono | 7 | 7 | 7 | 1.000 |
| NHANES | CKINN | 8 | 8 | 8 | 1.000 |
| NHANES | LR | 8 | 8 | 8 | 1.000 |
| NHANES | M1 | 8 | 8 | 8 | 1.000 |
| NHANES | M2 | 8 | 8 | 8 | 1.000 |
| NHANES | XGB | 8 | 8 | 8 | 1.000 |
| NHANES | XGB_mono | 8 | 8 | 8 | 1.000 |
| RICH | CKINN | 6 | 6 | 6 | 1.000 |
| RICH | LR | 6 | 6 | 6 | 1.000 |
| RICH | M1 | 6 | 6 | 5 | 0.833 |
| RICH | M2 | 6 | 6 | 6 | 1.000 |
| RICH | XGB | 6 | 6 | 6 | 1.000 |
| RICH | XGB_mono | 6 | 6 | 6 | 1.000 |

| cohort | model | spearman_mean_abs_shap_vs_mean_abs_ig |
|---|---|---|
| NAGALA | M1 | 0.936 |
| NAGALA | M2 | 0.868 |
| NAGALA | CKINN | 0.893 |
| RICH | M1 | 0.940 |
| RICH | M2 | 0.791 |
| RICH | CKINN | 0.846 |
| NHANES | M1 | 0.965 |
| NHANES | M2 | 0.903 |
| NHANES | CKINN | 0.935 |

Patient-level reasoning text for CKINN (texts are row-level and stay in the local cache):

| cohort | patients | share_consistent_with_all_priors | most_common_top_contributor | share_with_lowering_panel_counterfactual | median_largest_drop_in_probability |
|---|---|---|---|---|---|
| NAGALA | 50 | 0.940 | age_years | 0.960 | 0.001 |
| RICH | 50 | 1.000 | age_years | 0.900 | 0.002 |
| NHANES | 50 | 1.000 | age_years | 0.980 | 0.002 |

Scorecard (one row per cohort and model):

| cohort | model | auroc | auprc | brier | slope | citl | ece | sens | spec | ppv | npv | sex_gap_tpr_at_fpr_0.10 | violation_rate_perturbation | bmi_pair_share_lower | transport_gap_into_this_cohort |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAGALA | LR | 0.8053 | 0.1288 | 0.0222 | 0.9420 | 0.0058 | 0.0027 | 0.4631 | 0.9010 | 0.1036 | 0.9855 | 0.1035 | 0.2421 | 0.0000 | -0.0002 |
| NAGALA | cLR | 0.8057 | 0.1311 | 0.0222 | 0.9463 | 0.0039 | 0.0027 | 0.4651 | 0.9008 | 0.1039 | 0.9856 | 0.0962 | 0.0000 | 0.0000 | 0.0003 |
| NAGALA | GAM | 0.8015 | 0.1213 | 0.0224 | 1.1149 | 0.0037 | 0.0035 | 0.4597 | 0.9007 | 0.1025 | 0.9854 | 0.1233 | 0.3260 | 0.0000 | 0.0064 |
| NAGALA | RF | 0.8047 | 0.1122 | 0.0224 | 1.0884 | 0.0050 | 0.0029 | 0.4584 | 0.9016 | 0.1032 | 0.9854 | 0.1182 | 0.2797 | 0.0000 | 0.0182 |
| NAGALA | XGB | 0.8008 | 0.1080 | 0.0225 | 0.8971 | 0.0144 | 0.0029 | 0.4664 | 0.9002 | 0.1035 | 0.9856 | 0.1009 | 0.2723 | 0.0000 | 0.0086 |
| NAGALA | XGB_mono | 0.7983 | 0.1068 | 0.0226 | 0.8521 | 0.0319 | 0.0038 | 0.4570 | 0.8991 | 0.1006 | 0.9853 | 0.1207 | 0.0000 | 0.0000 | 0.0090 |
| NAGALA | LGBM | 0.7964 | 0.1031 | 0.0226 | 0.9002 | 0.0315 | 0.0029 | 0.4577 | 0.8998 | 0.1013 | 0.9853 | 0.1311 | 0.1518 | 0.0036 | 0.0093 |
| NAGALA | LGBM_mono | 0.7965 | 0.1086 | 0.0227 | 0.8627 | 0.0091 | 0.0037 | 0.4624 | 0.8997 | 0.1022 | 0.9855 | 0.1155 | 0.0000 | 0.0000 | 0.0067 |
| NAGALA | M1 | 0.7981 | 0.1105 | 0.0225 | 0.9716 | 0.0182 | 0.0031 | 0.4617 | 0.9002 | 0.1025 | 0.9855 | 0.1324 | 0.0971 | 0.0026 | 0.0105 |
| NAGALA | M2 | 0.7985 | 0.1097 | 0.0225 | 0.9165 | -0.0004 | 0.0030 | 0.4604 | 0.8989 | 0.1012 | 0.9854 | 0.1065 | 0.0943 | 0.0000 | 0.0061 |
| NAGALA | CKINN | 0.8009 | 0.1133 | 0.0224 | 0.9484 | 0.0141 | 0.0028 | 0.4624 | 0.8998 | 0.1022 | 0.9855 | 0.1130 | 0.0032 | 0.0000 | 0.0060 |
| NAGALA | SVM | 0.7848 | 0.1048 | 0.0228 | 0.9748 | -0.0161 | 0.0045 | 0.4557 | 0.9000 | 0.1012 | 0.9853 | 0.0614 | 0.3806 | 0.0054 | 0.0089 |
| RICH | LR | 0.8469 | 0.0947 | 0.0185 | 1.0021 | -0.0020 | 0.0025 | 0.4968 | 0.9002 | 0.0910 | 0.9889 | 0.1794 | 0.0195 | 0.0000 | 0.0285 |
| RICH | cLR | 0.8469 | 0.0947 | 0.0185 | 1.0017 | -0.0002 | 0.0026 | 0.4973 | 0.9003 | 0.0911 | 0.9889 | 0.1782 | 0.0000 | 0.0000 | 0.0223 |
| RICH | GAM | 0.8483 | 0.0990 | 0.0185 | 0.9934 | -0.0008 | 0.0017 | 0.5042 | 0.9008 | 0.0927 | 0.9891 | 0.1462 | 0.0967 | 0.0000 | 0.0419 |
| RICH | RF | 0.8430 | 0.0925 | 0.0185 | 1.3665 | 0.0012 | 0.0045 | 0.4952 | 0.9003 | 0.0907 | 0.9889 | 0.1658 | 0.0524 | 0.0000 | 0.0495 |
| RICH | XGB | 0.8481 | 0.0957 | 0.0184 | 1.0847 | 0.0004 | 0.0015 | 0.5026 | 0.9005 | 0.0922 | 0.9890 | 0.1530 | 0.0056 | 0.0000 | 0.0417 |
| RICH | XGB_mono | 0.8455 | 0.0929 | 0.0185 | 1.1890 | 0.0035 | 0.0030 | 0.4919 | 0.9004 | 0.0903 | 0.9888 | 0.1536 | 0.0000 | 0.0000 | 0.0240 |
| RICH | LGBM | 0.8479 | 0.0957 | 0.0184 | 1.1049 | -0.0003 | 0.0017 | 0.5041 | 0.9002 | 0.0921 | 0.9890 | 0.1599 | 0.0087 | 0.0002 | 0.0430 |
| RICH | LGBM_mono | 0.8474 | 0.0961 | 0.0185 | 1.0767 | 0.0014 | 0.0021 | 0.5004 | 0.9003 | 0.0917 | 0.9890 | 0.1517 | 0.0000 | 0.0000 | 0.0256 |
| RICH | M1 | 0.8494 | 0.0983 | 0.0184 | 0.9990 | 0.0076 | 0.0011 | 0.5040 | 0.9007 | 0.0925 | 0.9891 | 0.1673 | 0.0542 | 0.0000 | 0.0334 |
| RICH | M2 | 0.8486 | 0.0979 | 0.0184 | 0.9833 | -0.0123 | 0.0006 | 0.5077 | 0.9003 | 0.0928 | 0.9891 | 0.1656 | 0.0446 | 0.0000 | 0.0237 |
| RICH | CKINN | 0.8488 | 0.0987 | 0.0184 | 0.9774 | -0.0137 | 0.0007 | 0.5036 | 0.9002 | 0.0921 | 0.9890 | 0.1613 | 0.0001 | 0.0000 | 0.0240 |
| RICH | SVM | 0.8403 | 0.0935 | 0.0187 | 0.9992 | 0.0016 | 0.0054 | 0.5007 | 0.9002 | 0.0916 | 0.9890 | 0.1794 | 0.2221 | 0.0000 | 0.0395 |
| NHANES | LR | 0.8142 | 0.1553 | 0.0379 | 0.9988 | 0.0007 | 0.0037 | 0.4282 | 0.9024 | 0.1620 | 0.9728 | 0.0477 | 0.0000 | 0.0000 | -- |
| NHANES | cLR | 0.8142 | 0.1554 | 0.0379 | 1.0000 | 0.0002 | 0.0036 | 0.4274 | 0.9024 | 0.1618 | 0.9728 | 0.0501 | 0.0000 | 0.0000 | -- |
| NHANES | GAM | 0.8271 | 0.1702 | 0.0374 | 1.1083 | -0.0004 | 0.0041 | 0.4439 | 0.8999 | 0.1635 | 0.9735 | 0.0442 | 0.0732 | 0.0000 | -- |
| NHANES | RF | 0.8111 | 0.1570 | 0.0378 | 1.0650 | -0.0012 | 0.0027 | 0.4107 | 0.9011 | 0.1547 | 0.9720 | 0.0602 | 0.2492 | 0.0600 | -- |
| NHANES | XGB | 0.8156 | 0.1620 | 0.0377 | 1.2769 | 0.0041 | 0.0074 | 0.4272 | 0.8998 | 0.1582 | 0.9727 | 0.0713 | 0.0070 | 0.0002 | -- |
| NHANES | XGB_mono | 0.8155 | 0.1621 | 0.0377 | 1.2797 | 0.0053 | 0.0078 | 0.4244 | 0.9001 | 0.1576 | 0.9726 | 0.0668 | 0.0000 | 0.0000 | -- |
| NHANES | LGBM | 0.8186 | 0.1610 | 0.0377 | 1.0678 | 0.0078 | 0.0054 | 0.4246 | 0.9003 | 0.1580 | 0.9726 | 0.0473 | 0.0301 | 0.0100 | -- |
| NHANES | LGBM_mono | 0.8187 | 0.1637 | 0.0377 | 1.0495 | 0.0041 | 0.0053 | 0.4318 | 0.8999 | 0.1597 | 0.9729 | 0.0374 | 0.0000 | 0.0000 | -- |
| NHANES | M1 | 0.8201 | 0.1636 | 0.0376 | 0.9887 | 0.0208 | 0.0030 | 0.4350 | 0.9006 | 0.1618 | 0.9731 | 0.0645 | 0.0129 | 0.0000 | -- |
| NHANES | M2 | 0.8252 | 0.1688 | 0.0374 | 0.9390 | 0.0151 | 0.0028 | 0.4540 | 0.8996 | 0.1661 | 0.9740 | 0.0364 | 0.0121 | 0.0000 | -- |
| NHANES | CKINN | 0.8256 | 0.1692 | 0.0374 | 0.9465 | 0.0020 | 0.0027 | 0.4580 | 0.8991 | 0.1667 | 0.9741 | 0.0232 | 0.0005 | 0.0000 | -- |
| NHANES | SVM | 0.8125 | 0.1589 | 0.0378 | 1.0044 | 0.0057 | 0.0066 | 0.4272 | 0.9007 | 0.1593 | 0.9727 | 0.0705 | 0.1806 | 0.0182 | -- |

## 13. Limitations
- Version 1 results were seen before every v2 decision; v2 is a re-analysis of the same development and evaluation rows, not a new test set. All amendments are listed in protocol_v2.md, section 2.
- Task B labels are binary incident diabetes over follow-up windows that differ between cohorts (NAGALA 0.4 to 13.0 years, RICH 2.0 to 7.6 years; AUDIT F15); transport differences mix population shift with design. The censoring-aware analyses address the horizon only in part.
- Cross-population evidence is from Japan and China; NHANES is a detection task, its extract is a non-random subset of the survey (AUDIT F14), and survey-weighted estimates do not represent the US population.
- RICH family history is probably under-recorded at several sites (AUDIT F16); RICH drinking status is unused (AUDIT F17).
- Missingness experiments remove values at test time; the training augmentation of M2, CKINN, LR_aug and XGB_aug matches the primary MCAR scenario, which favours those models against the other comparators (AUDIT F10).
- H4 is now measured by across-seed attribution stability, which CKINN is not trained for; the violation rates remain a manipulation check and are not evidence of clinical validity. Agreement with clinical priors is not proof that a model is causally correct.
- Subgroup gaps at one common threshold reflect differences in risk between groups; sensitivity at matched specificity isolates ranking within groups but subgroups with few events have wide intervals.
- A 'seed' changes the network initialisation, the inner validation split and the batch order together, so the seed variance component includes inner-split variance. M1, which has no augmentation, sees its mini-batches in a different order from M2 and CKINN after the first epoch.
- Lambda was chosen at the top of the pre-declared grid in every cohort and tier: the development score barely changed across the grid and the declared rule takes the largest lambda within one standard error, so CKINN here is a strongly constrained network.
- Bootstrap intervals hold the fitted models fixed; the corrected repeated-CV t-test, which allows for training variability, gives wider intervals. Intervals for the IPCW time-dependent AUROC hold the censoring distribution fixed.
- The post hoc analyses (label PH; section 12 of report.md) were decided after the v2 results were seen; they describe the models further and cannot support a confirmatory claim. The reasoning texts and SHAP values are model behaviour, not causal effects or advice.
- Outputs are risk estimates from retrospective research data; they are not diagnoses and have not been validated for clinical use.
