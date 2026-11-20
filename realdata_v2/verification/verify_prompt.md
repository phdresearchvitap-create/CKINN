You are an independent verifier. You did not write this study's code and must not read or import its library (`ckv2`, `notebook_cells.py`). Write your own short Python script with numpy, pandas and scikit-learn only, and recompute key numbers from the saved predictions, then compare with the stored tables.

Python: /tmp/claude-0/-home-user-CKINN/552669ee-251b-5795-80d6-31c2a57a4763/scratchpad/venv_colab/bin/python
Saved predictions (row-level; never print rows, only aggregates): /tmp/claude-0/-home-user-CKINN/552669ee-251b-5795-80d6-31c2a57a4763/scratchpad/v2full/outputs_v2_full/cache/preds/
- `<COHORT>_runs.npz` for NAGALA, RICH, NHANES: key `y` (labels); keys `"<model>|r<r>|s<s>|<tag>"` = out-of-fold predicted probabilities for model, repeat r and seed index s; tag `none` = unmodified test data, other tags are test-time missingness scenarios (e.g. `mcar30`). Main runs are (r, s) = (0,0),(1,1),(2,2),(3,3),(4,4). Seed study: (0, s) for s = 1..4 for M1, M2, CKINN. Keys `"THRSPEC|<model>|r<r>|s<s>"` = per-row threshold chosen on the inner validation split; `"FOLD|..."` = fold id.
- `transport.npz`: `ext|<src>|<tgt>|<model>|s<s>` (source-trained model applied to all target rows), `ref|<tgt>|<model>|s<s>` (5-fold CV inside the target, repeat 0), `y_NAGALA`, `y_RICH`.
Stored tables: /tmp/claude-0/-home-user-CKINN/552669ee-251b-5795-80d6-31c2a57a4763/scratchpad/v2full/outputs_v2_full/tables/ (read CSVs with keep_default_na=False, na_values=['']).

Recompute and compare (report each as match / mismatch with both values; tolerance 1e-6 unless stated):
1. Table 08 (`table08_performance.csv`): for every main model in each cohort, run-averaged AUROC (sklearn roc_auc_score per main run, then the mean over the 5 runs), run-averaged Brier, run-averaged AUPRC (average_precision_score), and scaled Brier = 1 - Brier / (prev * (1 - prev)).
2. Table 09 (`table09_hypothesis_endpoints.csv`): the point estimates of H1 (CKINN minus M2 and CKINN minus M1 run-averaged AUROC) and H2 Brier differences, per cohort.
3. Calibration slope of CKINN and M2 per run via an unpenalised logistic regression of y on logit(p) (statsmodels is not available; use scikit-learn LogisticRegression with C=1e10), averaged over runs; compare with table 08 `slope` (tolerance 1e-3).
4. Table 31 (`table31_transport.csv`): for M2 and CKINN, seed-averaged AUROC of `ref` and `ext` predictions and the transport gap, for both directions.
5. Table 22 (`table22_missingness_auroc.csv`): run-averaged AUROC under `mcar30` for M2 and CKINN in each cohort.
6. Table 12 (`table12_variance_components.csv`): SD across the 5 seeds of repeat 0 of the AUROC of CKINN, and SD across the 5 main runs.
7. An independent paired percentile bootstrap (your own code, 1,000 draws, seed of your choice) of the run-averaged CKINN minus M2 AUROC difference in NAGALA: report your 95% interval next to the stored `ci_lo`/`ci_hi` (they will differ slightly because the draws differ; flag a difference larger than 0.003).
8. Sanity: every prediction in [0, 1], no NaN; the number of rows equals the `n` column of table 08; within each main run every row has exactly one fold id.

Do not edit any file in the repository. Write your script to /tmp/claude-0/-home-user-CKINN/552669ee-251b-5795-80d6-31c2a57a4763/scratchpad/verify/verify.py and its output to verify_output.txt in the same folder. Finish with a short report: list of checks, match or mismatch, and any concern.
