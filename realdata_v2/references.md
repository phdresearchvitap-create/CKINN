# References (verification status)

Checked 2026-11-20 in PubMed. Scite could not be used (the account has no active Scite plan), so references outside PubMed carry `TODO(cite)` and must be checked before submission. Nothing here was cited from memory without a flag.

## Verified in PubMed

| Use in this study | Reference | PMID | DOI |
| --- | --- | --- | --- |
| NAGALA cohort (data source) | Okamura T, Hashimoto Y, Hamaguchi M, Obora A, Kojima T, Fukui M. Ectopic fat obesity presents the greatest risk for incident type 2 diabetes: a population-based longitudinal study. *Int J Obes (Lond)* 2019;43(1):139-148. | 29717276 | [10.1038/s41366-018-0076-3](https://doi.org/10.1038/s41366-018-0076-3) |
| Rich Healthcare Group cohort (data source) | Chen Y, Zhang X-P, Yuan J, et al. Association of body mass index and age with incident diabetes in Chinese adults: a population-based cohort study. *BMJ Open* 2018;8(9):e021768. | 30269064 | [10.1136/bmjopen-2018-021768](https://doi.org/10.1136/bmjopen-2018-021768) |
| Reporting guideline | Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. | 38626948 | [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) |
| Paired AUROC test | DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 1988;44(3):837-845. | 3203132 | none listed |
| Harrell's C | Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Stat Med* 1996;15(4):361-387. | 8668867 | [10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4](https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4) |
| Uno's C | Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med* 2011;30(10):1105-1117. | 21484848 | [10.1002/sim.4154](https://doi.org/10.1002/sim.4154) |
| IPCW Brier score | Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. *Stat Med* 1999;18(17-18):2529-2545. | 10474158 | see PubMed record |
| Decision curves | Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565-574. | 17099194 | [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361) |
| Calibration | Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17(1):230. | 31842878 | [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7) |
| Equivalence (TOST) | Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J Pharmacokinet Biopharm* 1987;15(6):657-680. | 3450848 | [10.1007/BF01068419](https://doi.org/10.1007/BF01068419) |

Note on the Rich cohort: the article describes 32 sites in 11 cities; the data file used here codes 15 site values. The study reports the 15 codes as found.

## Not verified (TODO(cite) before submission)

- Holm's sequentially rejective procedure (Scandinavian Journal of Statistics, 1979). TODO(cite)
- Nadeau and Bengio, corrected resampled t-test (Machine Learning, 2003). TODO(cite)
- Integrated gradients (Sundararajan, Taly and Yan, ICML 2017). TODO(cite)
- Multiple imputation by chained equations, and scikit-learn's `IterativeImputer`. TODO(cite)
- Cumulative/dynamic time-dependent AUROC with inverse-probability-of-censoring weights (the estimator implemented follows scikit-survival's `cumulative_dynamic_auc`, against which it is unit-tested). TODO(cite)
- NHANES 1999-2018 (National Center for Health Statistics): web citation, access date, survey-weight guidance, and the source of the harmonised extract used here. TODO(cite)
- Dryad deposits of the NAGALA and Rich data. TODO(cite)
- Diagnostic thresholds for diabetes (fasting plasma glucose 7.0 mmol/L, HbA1c 6.5%) and the 8-hour fasting requirement (American Diabetes Association Standards of Care). TODO(cite)
- Software: PyTorch, scikit-learn, XGBoost, LightGBM. TODO(cite)
