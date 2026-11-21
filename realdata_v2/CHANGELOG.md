# Changelog

## v2 + post hoc block (2026-11-21)

Added after the v2 results had been seen, to cover the 28-step protocol where the data allow it; logged in `protocol_v2.md` section 10 and labelled PH everywhere. No v2 verdict, configuration or hash changes.

- `ckv2/posthoc.py`: permutation-sampling SHAP (log-odds scale, additive by construction), model-agnostic perturbation violations, BMI 24/35 pairs, reasoning-text inputs and a fixed reasoning-text template, balanced and group weights, Wilson intervals, task builders for every post hoc analysis, predictor audit table, sign-table references.
- Engine: test sets can set a predictor to a fixed value; tasks can run perturbation, SHAP and reasoning probes.
- Models: SVM by Nystroem radial-kernel approximation with a linear SVM and Platt scaling.
- Data: TG/HDL ratio in all cohorts; NHANES "told of prediabetes" and "told of hypertension"; RICH final-visit fasting glucose (outcome of the progression analysis only, never an input).
- Notebook: cells PH1 to PH4 (fits; reproducibility, SVM, violations, variants, pooled indicator; transport matrix, recalibration, stress test, RQ7, progression, group AUPRC; SHAP, reasoning summary, audit, references, scorecard, two figures); report section 12.
- Tests: 38 (SHAP exactness on a linear model, perturbation violations, reasoning wording, weights, Wilson, progression label, SVM).
- Documents: `ERRORS_OBJECTIVE1.md`, `NOVELTY_SEARCH.md`, sign-table references in `references.md`, corrected `COVERAGE_28_STEPS.md`.

## v2 (2026-11-20)

Analysis changes are amendments A1 to A19 of `protocol_v2.md` (decided after the v1 results were seen, before any v2 result). This list covers the code.

### Fixed
- CKINN, M2 and M1 were trained with different seeds, inner splits and (RICH) training subsamples because the fold seed included the model name. Seeds now depend only on cohort, tier, repeat, fold and seed index; a unit test checks CKINN with lambda 0 equals M2 exactly (AUDIT F2, A1).
- NHANES label included people on insulin or diabetes pills and FPG values after less than 8 hours of fasting (AUDIT F13, A11).
- RICH ALT and AST zeros were kept as values (AUDIT F17, A12).
- The pooled-training ablation standardised test folds with their own statistics; it now uses training statistics only (AUDIT F12, A17).
- M1 was described as median-imputed; it fills the training mean on the transformed scale, and the description now says so (AUDIT F9).
- The outputs archive contained full copies of the raw data; raw and row-level files now live only in the cache, and cell 41 fails if one appears among the archived outputs (AUDIT F18, A18).
- Bootstrap p-values were a normal approximation to 300 draws; they are now read from 2,000 paired draws (AUDIT F7, A6).
- `NULL` verdict strings were read back as missing values by pandas; tables are now read with `keep_default_na=False`.
- Removed the no-op `... or None` lines and the `except Exception: pass` that hid bootstrap failures.

### Added
- Repeated stratified cross-validation (5 x 5) with paired network seeds and a 5-seed study; variance components (A2).
- Process-parallel, checkpointed task engine (`ckv2/engine.py`): every fit is a deterministic task written atomically to disk; re-running resumes.
- Equal tuning budget (32 evaluations) for M1, M2 and CKINN; finer lambda grid with a one-standard-error rule (A4).
- Competitors with monotone constraints (XGBoost, LightGBM), sign-constrained logistic regression, spline additive model, multiple imputation plus LR (A13).
- Hypothesis families A (vs M2) and B (vs M1) with Holm correction; redefined H3, H4 and H5 (A7 to A10); DeLong, corrected resampled t-test, TOST, minimum detectable effect, RICH cluster bootstrap (A6).
- Censoring-aware metrics (Harrell's C, Uno's C, IPCW AUROC) and 3-year IPCW models (A14); calibration intervals, ECE and reliability bands; decision curves; fixed-sensitivity and fixed-specificity operating points (A15).
- Non-circular constraint analyses: leave one constraint out, wrong-sign controls, alternative sign tables, lambda path, empirical signs against held-out partial dependence, sparse-region and other-cohort violation rates (A9).
- Missingness at three rates for MCAR, MAR and MNAR, RICH structural patterns (A16); survey-weighted NHANES sensitivity (A19).
- 28 unit tests on toy arrays, including agreement with scikit-learn, lifelines and scikit-survival.
- Figures as vector PDF and 300 dpi PNG; LaTeX tables; generated Methods and Results draft; TRIPOD+AI checklist; manifest with package versions and input, library and output hashes.

### Removed
- RBF SVM (capped at 8,000 rows in v1); v1 results for it remain in the v1 record.
- The fairness-weighted CKINN variant; its v1 results remain in the v1 record.

## v1 (2026-11-20)
Original study (`CKINN_real_data_study.ipynb`, frozen-protocol hash prefix `d5840db6d905283e`). Kept unchanged as the record.
