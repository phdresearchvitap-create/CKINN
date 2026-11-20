# TRIPOD+AI checklist (filled from what the code did)

Item wording is abbreviated and the numbering must be checked against the official TRIPOD+AI checklist before submission: TODO(verify).

| Section | Item (abbreviated) | Where addressed in this study |
|---|---|---|
| Title and abstract | Identify the study as developing and evaluating prediction models; summarise | To be written; report.md and methods_results_draft.md give the numbers |
| Background and objectives | Rationale, intended use, objectives | Proposal and protocol_v2.md section 1; outputs are research risk estimates, not diagnoses |
| Data sources | Source and dates of data | NAGALA, Rich Healthcare Group, NHANES 1999-2018 extract; file hashes in manifest.json |
| Participants | Eligibility, setting, treatments | Cell 17; NHANES exclusions (A11) in table03; no rows removed in NAGALA or RICH |
| Data preparation | Pre-processing and quality checks | AUDIT.md; cell 17; zeros treated as missing (A12); winsorising, log and z-score fitted in training folds |
| Outcome | Definition, timing, blinding | Incident diabetes as recorded by each cohort; NHANES laboratory definition; follow-up windows differ (AUDIT F15) |
| Predictors | Definition, timing, measurement | table05 feature sets; T2 excludes glucose and HbA1c; T3 sensitivity |
| Sample size | Justification | Fixed by the data; table04 events; minimum detectable effect in table11 |
| Missing data | Handling | Mask and augmentation (M2, CKINN); mean fill (M1); native handling (trees); multiple imputation baseline; tables 22-24 |
| Analytical methods | Model type, building, tuning, internal validation | protocol_v2.md sections 4-6; tables 06-07; repeated CV with paired seeds |
| Class imbalance | Methods used | None (no resampling); output bias initialised at the training prevalence |
| Fairness | Approaches to fairness | H5 and tables 20-21 (sex, age band, site, ethnic group); minimum-events rule |
| Model output | Form of predictions | Probabilities; thresholds from inner validation splits for operating points |
| Training versus evaluation | Differences between data used | Development rows tune only; evaluation rows by cross-validation; transport and site analyses external |
| Performance measures | Metrics and their uncertainty | AUROC, AUPRC, Brier, scaled Brier, slope, calibration-in-the-large, ECE, net benefit, censoring-aware metrics; bootstrap intervals |
| Model updating / recalibration | If done | Recalibration in the small in the target (table32, target labels used, exploratory) |
| Ethics | Approval and consent | Public de-identified data; check each source's terms before submission: TODO(verify) |
| Funding and conflicts | Sources and roles | To be completed by the authors |
| Protocol and registration | Where available | protocol_v2.md with amendment log and configuration hash; not registered |
| Data and code availability | Access | Data from public sources; this notebook and library (ckv2) reproduce every table |
| Patient and public involvement | Details | None |
| Participant flow and characteristics | Numbers, events, missingness | tables 01-04 |
| Model specification | Full model or how to obtain it | Configuration and chosen hyperparameters in manifest.json; code in ckv2 |
| Model performance | With intervals, by subgroup | tables 08-34; figures 1-6 |
| Interpretation and limitations | Overall interpretation, limitations, usability | report.md section 12; methods_results_draft.md |
