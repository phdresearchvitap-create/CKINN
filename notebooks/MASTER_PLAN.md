# CKINN notebook master plan, version 2 (88 cells)

Prepared 2026-11-11. It replaces the 44-cell plan of the same date (kept in git history). Sources, in order of authority: the locked protocol (`protocol/protocol_v1.0_LOCKED.md`) with `protocol/deviations.md`; the 26-page proposal; the two raw workbooks; the existing repository. The All-in-One PDF and the pasted "92-cell" prompt were treated as documents under review.

No model was trained to write this plan. All counts below are descriptive checks on the raw files.

The number 88 was not a target. It is what remains after every proposal requirement was given a cell, each stage was given a verifying gate, and cells that could not contain executable code were removed.

Companion files: `protocol/proposal_coverage.csv` (the requirement-by-requirement register, 86 rows) and `protocol/amendment_A1_PROPOSED.md` (everything in this plan that is not in the locked protocol).

Platform (from 2026-11-12, deviation D-7): Google Colab on the investigator's Windows laptop, project folder `My Drive/CKINN_Research`. Notebook cell 0 connects Google Drive and installs scikit-survival and lifelines; XGBoost, LightGBM, PyTorch and SHAP ship with Colab. Built notebook: `notebooks/CKINN_colab.ipynb`; the same cells as text: `notebooks/CKINN_colab_cells.txt`.

## 1. Status codes

| Code | Meaning | May support C1 to C4? |
| --- | --- | --- |
| L | Locked protocol v1.0, run exactly as locked | Yes |
| D | Logged deviation (D-1 site 14; D-2 all rows; D-3 zero-coded laboratory values) | Yes |
| A | Amendment A1, exploratory, proposed; the cell refuses to run until the item is approved in cell 3 | No |
| G | Gated: needs data or clinical raters that do not exist yet; the cell runs a real readiness check and stops | No |

Full coverage in this plan means every proposal requirement has a cell and a status. It does not mean every experiment can be completed with RICH and NAGALA; section 7 states which cannot.

## 2. Assessment of the 44-cell plan

Sound and retained: the data forensics, harmonization in mmol/L, the leakage blocklist, the toolkit with toy tests, the locked analyses R1 to R5 and S1 to S4, the lock gate before cross-cohort work, and the mechanical decision rules.

Wrong or incomplete, and corrected here:

| # | Fault in the 44-cell plan | Correction |
| --- | --- | --- |
| 1 | It stated that no intermediate glycaemic state is recorded. This is wrong for RICH, which records fasting glucose at the final visit. | Cells 13, 79, 80 |
| 2 | Logistic regression, random forest, LightGBM and SVM (proposal p. 17) were dismissed as "not run". | Cells 32 to 35, 45 |
| 3 | AUPRC, calibration intercept and accuracy (proposal pp. 15, 22) were missing. | Cells 25 to 27 |
| 4 | Ablation Models 4 and 5 (proposal p. 18) and "population alignment" (p. 19) were read as evaluation only; no trained component existed. | Cells 72, 75, 76 |
| 5 | The knowledge graph (pp. 24 to 25) was listed as out of scope with no design. | Cells 77, 78 |
| 6 | The physiological model, the third cohort and clinical expert agreement were named but had no module. | Cells 68, 81, 82 |
| 7 | H1 (superiority over the MLP) was folded into C1 (non-inferiority to the best baseline). They are different claims. | Cell 46 |
| 8 | The recalibration sample of 1,000 was not checked for events. It holds about 24 events in NAGALA and 20 in RICH (6 to 8 by 3 years). | Cell 59 |
| 9 | Gates were markdown notes that verified nothing. | Gates are code cells |
| 10 | Several proposal inputs (LDL, alcohol, weight, height) had no stated status. | Cells 15, 51 |

## 3. Errors in the pasted 92-cell prompt

| # | Error | What is done instead |
| --- | --- | --- |
| 1 | Its own count is wrong: its list has 12 markdown and 80 code cells, not 11 and 81. | Not followed |
| 2 | Wrong file names. The workbooks are `DatasetIntJObesity.xlsx` and `RCHealthCareData-20180820.xlsx`. | Correct names, verified by hash |
| 3 | Cell 80 demands a current-diabetes classifier. Neither cohort contains a prevalent case (no baseline glucose of 7.0 mmol/L or more, no HbA1c of 6.5% or more), and the proposal itself sets this option aside (p. 4: "I would choose B or C"). | Recorded as not identifiable (cells 13, 81) |
| 4 | Cells 79, 81 and 85 are code cells for branches with no data, while the prompt also forbids placeholder code. | Executable readiness and intake checks (cells 81, 82) |
| 5 | Cell 81 treats progression as wholly unobservable. | One-step progression is run in RICH (cells 79, 80) |
| 6 | Cell 36 asks for a LightGBM survival model. LightGBM has no built-in Cox objective. | LightGBM enters as a censoring-weighted classifier (cell 34) |
| 7 | Cell 32 adds an unpenalized Cox model that the proposal does not ask for and that is indistinguishable from ridge 0.01 with 211,833 rows and eight predictors. | Not added |
| 8 | "Population alignment" (proposal p. 19) is absent; cross-population is treated only as evaluation. | Cell 75 |
| 9 | Eleven markdown gates that cannot fail. | Eleven code gates that write a signed record |
| 10 | It asks for the first executable cell in the same response as the plan. | Plan first, as instructed; cell 1 follows on request |

Adopted from the prompt because they are right: AUPRC, the recalibration feasibility check, H1 reported separately, classifiers under an explicit censoring policy, random survival forest, fairness-aware training, violation magnitude, accumulated local effects, the expert-review rubric, and a requirement register.

## 4. Proposal coverage (summary of `protocol/proposal_coverage.csv`)

| Proposal section (page) | Requirement | Cells | Status |
| --- | --- | --- | --- |
| Title (1, 26) | "CKINN", not "PINN"; risk prediction | 1, 88 | L |
| 1, 5 Inputs (1 to 2, 5 to 6) | Every listed input has a recorded status | 15 | L, A, G |
| 2, 6, 10 Clinical priors (2, 6 to 7, 10) | Soft directional priors, "generally", subject to data | 19, 38, 39 | L |
| 3A Current classification (3) | Not chosen by the proposal; no prevalent cases | 13, 81 | Not identifiable |
| 3B Future risk (3 to 4) | 3-year incident risk | 43 to 46 | L |
| 3C Progression (4) | Prediabetes to diabetes; healthy to prediabetes | 13, 79, 80 | A; full multistate G |
| 4 Not diagnosis (4 to 5) | Wording | 1, 67, 88 | L |
| 7, 21 Architecture and framework (7 to 8, 19 to 20) | Cleaning, missing data, features, alignment, network, loss, risk, calibration, fairness, XAI, explanation | 16 to 22, 37 to 41, 63 to 67, 75 | L, A |
| 8, 9 Loss (8 to 9) | Data loss plus lambda times clinical loss, weights w | 23, 38, 39 | L |
| 11, 28 Physiological model (10 to 11, 26) | Glucose-insulin dynamics | 81 | G |
| 12 Cross-population (11 to 12) | Population A to B; hospitals A to C to D | 53 to 61 | L |
| 14, 15 Explainability (13 to 14) | SHAP, Integrated Gradients, counterfactual, perturbation, monotonicity, clinical reasoning, no medical advice | 63 to 67 | A |
| 16 Fairness (14 to 15) | Groups; sensitivity, specificity, calibration, FNR, FPR, AUROC, AUPRC | 20, 70 to 73 | L (T2 subgroups), A |
| 17 Calibration (15 to 16) | Curve, intercept, slope, Brier | 26, 50, 59, 60 | L, A (intercept) |
| 18 Datasets (16 to 17) | Two populations; third external cohort "if available" | 57, 58, 82 | L; third cohort G |
| 19 Comparison (17 to 18) | LR, RF, XGBoost, LightGBM, SVM, MLP, CKINN | 30 to 35, 43 to 45 | L, A |
| 20 Ablation (18 to 19) | Models 1 to 5; "what actually improves performance" | 46 to 49, 76 | L, A |
| 22 RQ1 to RQ7 (20 to 21) | See section 5 | 85 | L, A |
| 23 H1 to H5 (21) | See section 5 | 85 | L, A |
| 24 Evaluation (22 to 23) | Discrimination, classification, calibration, generalization, fairness, explainability incl. expert agreement | 25 to 27, 66, 68 | L, A, G |
| 25 Novelty (23 to 24) | Positioning for the paper | 88 | Writing |
| 26 Knowledge graph (24 to 25) | Graph plus network plus XAI | 77, 78 | A |
| 27 Four layers (25 to 26) | Prediction, knowledge, generalization, explanation; five evaluation axes | 85 | L, A |

## 5. Research questions, hypotheses and claims

Each line is a different kind of claim. One being supported never implies another.

| Item | Kind of claim | Evidence | Cells | Status |
| --- | --- | --- | --- | --- |
| RQ1 | Predictive accuracy | R1, all models | 43 to 46 | L |
| RQ2, H1 | Superiority of CKINN over MLP | CKINN minus MLP, Nadeau-Bengio interval | 46 | A (secondary) |
| C1 | Non-inferiority to the best locked baseline, margin 0.01 | R1, T2 | 46, 84 | L |
| RQ3, H3, C2 | Better transport | R2, R3, R4 | 53 to 60, 84 | L |
| RQ4, H2, C3 | Better calibration under transport | R3 raw predictions | 57 to 60, 84 | L |
| RQ5, H4, C4 | Fewer violations of the priors | Violation rate, R1 T2 | 46, 84 | L |
| RQ5, H4 | Explanation consistency and faithfulness | Stability, fidelity, cross-cohort agreement | 63 to 66 | A |
| RQ5 | Clinical plausibility | Independent expert ratings | 68 | G |
| RQ6, H5 | More consistent group performance | Group gaps with intervals; fairness-aware variant | 70 to 73 | A |
| RQ7 | Risk before diagnostic thresholds | Tiers without glucose; normoglycaemic stratum | 43, 44, 79 | L, A |

## 6. Controlled comparisons

All neural variants share the network (64, 64), optimizer, folds, preprocessing and metrics; each differs from its reference in one component only.

| Comparison | Differs by | Where | Metrics | Status |
| --- | --- | --- | --- | --- |
| MLP vs CKINN | Penalty | R1 to R4, all locked tiers | Protocol section 9 | L |
| MLP vs MLP + one prior | One prior added (BMI, TG, SBP, age; FPG at T3) | R1, repeat 0, T2 and T3 | Section 9; violation rate per variable | A |
| CKINN vs CKINN minus one prior | One prior removed | Same | Same | A |
| CKINN along lambda (0, 0.1, 1, 10) | Penalty weight | Same | Scaled Brier, C-index, violation rate and magnitude | A |
| CKINN vs XGB-mono | Network against monotone trees | R1 to R4 | Section 9 | L |
| CKINN vs Cox, XGB | Model family | R1 to R4 | Section 9 | L |
| CKINN vs RSF, LR, RF, XGBoost, LightGBM, SVM | Model family; classifiers use the censoring-weighted 3-year endpoint | R1 repeat 0 and R3, T2 and T3 | 3-year risk metrics | A |
| CKINN vs aligned CKINN (Model 4) | Importance weights from target covariates | R3, T2 and T3 | Section 9 | A |
| CKINN vs fairness-aware CKINN (Model 5) | Group-loss parity penalty | R1 repeat 0 and R3, T2 | Section 9; group gaps | A |
| CKINN vs graph-structured CKINN | Connectivity mask | R1 repeat 0 and R3, T2 and T3 | Section 9 | A |
| Model 1 vs Model 2 (MLP + XAI) | Nothing predictive; reported as identical predictions plus explanation metrics | Cell 76 | Explanation metrics | A |

## 7. What can and cannot be done with RICH and NAGALA

| Component | Verdict | Reason |
| --- | --- | --- |
| Incident 3-year risk, both cohorts | Executable | Event and time recorded |
| Both transport directions | Executable, imprecise in NAGALA | 101 NAGALA events by 3 years; 17 in women; 5 to 7 in each female age band |
| Target recalibration on 1,000 | Executable, imprecise | About 24 (NAGALA) and 20 (RICH) events in the sample |
| HbA1c, waist, exercise, family history priors | Within one cohort only | Each exists in one cohort |
| Prediabetes to diabetes | Executable in both cohorts | Baseline glucose 5.6 to 6.99 mmol/L: RICH 26,247 (2,896 events); NAGALA by glucose or HbA1c 3,172 (271 events) |
| Healthy to prediabetes | Executable in RICH only, as status at the final visit | Of 185,586 below 5.6 mmol/L at baseline, 20,785 were at 5.6 to 6.99 at the final visit and 1,278 had diabetes. Transition time is unknown; NAGALA has no final-visit glucose, so no external check |
| Full multistate model with transition times | Blocked | Needs dated intermediate visits |
| Current-diabetes classification | Not identifiable | Zero prevalent cases; not chosen by the proposal |
| Glucose-insulin differential-equation model | Blocked | No insulin, C-peptide or glucose-tolerance data; one glucose value per person in NAGALA, two in RICH |
| Fairness by ethnicity or socio-economic group | Blocked | Not recorded |
| Third external cohort; pooled training on A + B then test on C | Blocked | Needs a third cohort with incident diabetes, follow-up time and the tier predictors |
| Clinical expert agreement | Blocked | Needs at least two independent clinicians |
| Generalization to Indian or other populations | Not claimable | Two East Asian screening cohorts |

A blocked component is never replaced by simulated data or assumed labels.

## 8. The cells

Every code cell writes its result and a manifest entry (code hash, protocol hash, data hashes, configuration). A stored file is reused only if its manifest entry matches the current configuration and its row counts and checksums validate; otherwise the unit is re-run. Each gate cell re-checks the stage's outputs and writes `results/gates/gate_X.json`; the next stage starts only when the gate passes.

### Stage A. Governance (cells 1 to 6)

| Cell | Type | What it does | Why (proposal page; protocol section) | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | M | Scope: title, RQ1 to RQ7, H1 to H5, C1 to C4, the 3-year estimand, tiers, status codes, list of extensions. | Fixes what may be claimed (pp. 1, 20 to 21, 26; s. 1). | None | L |
| 2 | C | Versions, device, one thread, seeds, folders, protocol constants. | Reproducibility (s. 12). | Versions printed | L |
| 3 | C | Verifies the lock hash; lists deviations; records, item by item, which parts of amendment A1 you have approved. | No locked setting changes silently; exploratory cells cannot run unapproved. | Hash matches; approvals table | L |
| 4 | C | Hashes both workbooks, reads them once, asserts sheet names, shapes and exact headers, caches unchanged copies. | Provenance (s. 2). | 15,464 by 31; 211,833 by 25 | L |
| 5 | C | Loads the coverage register and checks every requirement has a cell and a status. | Nothing in the proposal can drop out unnoticed. | 86 rows, none unassigned | L |
| 6 | C | Gate A. | Stop point. | Pass | L |

### Stage B. Raw-data forensics (cells 7 to 14)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 7 | C | Identifiers, duplicates, types, missingness per column, code tables (sex, site, smoking, drinking, alcohol, family history, NAGALA phenotype). | One row per participant; coding faults visible (p. 6: assess missingness). | No duplicates; HDL 44.64%, AST 58.20% missing in RICH | L |
| 8 | C | Exact zeros and implausible values in every laboratory variable, by site. Deletes nothing. | A zero triglyceride is a code for "not measured" (D-3). | RICH zeros: TG 860, HDL 102, ALT 2, AST 1; NAGALA HDL 11 | D |
| 9 | C | Rebuilds each event flag. RICH: diagnosis column against final-visit glucose; checks that follow-up ends at diagnosis for the diagnosis-only outcome. | Defines the label (s. 3; p. 4). | 373; 4,174 = 770 + 2,870 + 534; none unexplained | L |
| 10 | C | Follow-up, censoring and support at 2, 3, 4, 5 years; reverse Kaplan-Meier; maximum follow-up by site. | Confirms the horizon; shows a binary 3-year label is invalid (s. 6; D-1). | 50.1% of RICH censored before 3 years; sites 14 and 16 below 3 years | L |
| 11 | C | Design difference: RICH has no follow-up under 2 years and no event before 2.0 years; NAGALA has 61. States the unconditional and conditional risk estimands. | Transport must compare like with like. | 0 and 61 events before 2 years; 40 NAGALA events between 2 and 3 years | L |
| 12 | C | NAGALA sex coding (counts, derived height, event rate); agreement of mg/dL and mmol/L pairs. | Sex is in every tier; no unit conversion is needed. | 8,430 and 7,034; 170.8 and 158.3 cm | L |
| 13 | C | Glycaemic audit: baseline ceilings (6.99 and 6.05 mmol/L; HbA1c 6.4%); prevalent diabetes; baseline strata; RICH state at the final visit. | Eligibility differs between cohorts; decides which proposal options (p. 3 to 4) are identifiable. | Zero prevalent cases; counts of section 7 | L |
| 14 | C | Gate B. | Stop point. | Pass | L |

### Stage C. Harmonization (cells 15 to 21)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 15 | C | Feature registry: every input named in the proposal, with source column, unit, missingness, tier, prior, or the reason it is unavailable. | "Do not blindly put every variable in" (p. 6). | Registry file; every proposal input present | L |
| 16 | C | One schema for both cohorts in mmol/L; male = 1; time in years; event; diagnosis-only event; D-3; glycaemic state variables. Saves files, hashes and a flow ledger. | Everything downstream reads these files (s. 3, 4). | Rows and events unchanged | L, D |
| 17 | C | Tiers T1 to T4, extras E2 and E3, extended extras E4; blocklist asserting that final-visit glucose, diagnosis flag, event, time, identifier and site never enter a predictor set. | Leakage is impossible by construction (p. 6; s. 4). | Assertion passes | L; E4 is A |
| 18 | C | Table 1 by cohort and event status; standardized differences; missingness by tier and site. | Describes the shift that transport must survive (p. 12). | Table saved | L |
| 19 | C | Kaplan-Meier 3-year risk by quintile of each constrained variable; curves by cohort. | Shows whether each prior holds descriptively (pp. 6 to 7; s. 0). | No RICH event before 2 years | L |
| 20 | C | Events by sex, age band, their intersection, site and cohort, in total and by 3 years. | Decides which group comparisons are estimable (p. 15). | NAGALA women: 17 events by 3 years | L |
| 21 | C | Gate C. | Stop point. | Pass | L |

### Stage D. Statistical toolkit, tested on toy arrays only (cells 22 to 29)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 22 | C | Fold-wise preprocessing (median or mode, missingness indicators, standardization); repeated 5-fold, leave-one-site-out, shift and inner splits. | Test data never inform preprocessing (s. 5, 8). | Changing test rows leaves fitted statistics unchanged | L |
| 23 | C | Cox partial likelihood with ties; Breslow hazard; risk 1 - exp(-H0(3) exp(lp)); conditional risk 1 - exp(-(H0(3) - H0(2)) exp(lp)). | The estimand (s. 6); the landmark formula. | Matches hand calculation | L; conditional is A |
| 24 | C | Kaplan-Meier, reverse Kaplan-Meier, censoring weights. | Needed by every 3-year metric. | Toy tests | L |
| 25 | C | C-index, time-dependent AUC, weighted Brier and scaled Brier; censoring-weighted average precision. | Discrimination (pp. 15, 22; s. 9). | Toy tests | L; AUPRC is A |
| 26 | C | Calibration slope and distance, observed-to-expected ratio, decile table, net benefit; intercept on the complementary log-log scale. | Calibration (pp. 15 to 16, 22; s. 9). | Toy tests | L; intercept is A |
| 27 | C | Censoring-weighted sensitivity, specificity, PPV, NPV, F1, accuracy, false-negative and false-positive rates at fixed thresholds. | Classification metrics (pp. 15, 22). | Agrees with plain formulas when nothing is censored | A |
| 28 | C | Nadeau-Bengio interval; paired bootstrap; violation rate and magnitude; the unit runner with manifest validation. | Inference for the decision rules (s. 9, 11). | Toy tests | L; magnitude is A |
| 29 | C | Gate D. | Stop point. | All tests pass | L |

### Stage E. Baseline models (cells 30 to 36)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 30 | C | Cox regression, ridge 0.01; one smoke-test fold. | Survival counterpart of logistic regression (p. 17; s. 6). | Risks in [0, 1] | L |
| 31 | C | Boosted Cox, and boosted Cox with the monotone constraints. | XGBoost family; conventional monotone model (p. 17; s. 6, 7). | Smoke test | L |
| 32 | C | Random survival forest, fixed settings; time and memory measured on one fold. | Random-forest family for censored data (p. 17). | Feasibility report; if it cannot run, that is the result | A |
| 33 | C | Defines the 3-year classification endpoint: event by 3 years is 1, followed beyond 3 years is 0, censored earlier gets weight 0, the rest are weighted by inverse censoring probability. | Classifiers have no likelihood for censoring; nobody censored early is called a non-event. | RICH: 105,778 with known status, 106,055 with weight 0 | A |
| 34 | C | Logistic regression, random forest, XGBoost and LightGBM classifiers on that endpoint. | The proposal's named families (p. 17). | Smoke test | A |
| 35 | C | SVM: linear, and radial kernel by Nystroem approximation, with weighted Platt scaling fitted inside the training fold. | SVM (p. 17); an exact kernel SVM does not scale to these rows. | Probabilities in [0, 1] | A |
| 36 | C | Gate E. | Stop point. | Pass | L |

### Stage F. MLP and CKINN (cells 37 to 42)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 37 | C | MLP with Cox loss: hidden (64, 64), dropout 0.1, AdamW, early stopping. | Unconstrained reference (p. 18; s. 6). | Trains on toy data | L |
| 38 | C | Prior registry: variable, direction, weight (all 1), tiers and cohorts where it applies. | The knowledge layer (pp. 6 to 7; s. 7). No prior on an absent feature. | Matches protocol section 7 | L |
| 39 | C | CKINN: the same network plus lambda times the sum over priors of the mean squared negative part of s times the derivative of the log hazard, by automatic differentiation on standardized inputs, at observed covariate values only. | The proposed model (p. 9; s. 6). | Loss finite | L |
| 40 | C | Tests: lambda 0 reproduces the MLP exactly; penalty gradient matches finite differences; penalty is zero for a monotone toy network; risk conversion. | Proves the only difference between MLP and CKINN is the penalty. | All pass | L |
| 41 | C | Lambda rule (inner 80/20; 0, 0.1, 1, 10; largest positive value within 0.002 scaled Brier of the best, else 0.1); checkpoint writer; one smoke-test fold in RICH. | Selection uses training data only (s. 6). | Smoke test | L |
| 42 | C | Gate F. | Stop point. | Pass | L |

### Stage G. Internal comparison and ablation in RICH (cells 43 to 52)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 43 | C | R1 for Cox, XGB, XGB-mono: 3 repeats of 5-fold, T1 to T4. | RQ1; baselines for C1. Replaces the superseded files. | 60 units per model | L |
| 44 | C | R1 for MLP and CKINN on the same folds; lambda log; out-of-fold predictions and checkpoints. | RQ2; inputs to C1 and C4. | 60 units per model | L |
| 45 | C | R1 (repeat 0, T2 and T3) for the forest and the five classifiers. | Full comparison of p. 17. They do not enter the "best baseline" of C1. | Units complete | A |
| 46 | C | Summary by model and tier; Nadeau-Bengio intervals for C1 and C4; H1 (CKINN minus MLP) reported separately. | The numbers on which C1, C4 and H1 are judged. | Table complete | L; H1 is A |
| 47 | C | MLP plus one prior at a time. | Which prior, alone, changes anything (p. 18). | Units complete | A |
| 48 | C | CKINN with one prior removed at a time. | Which prior the full model depends on (p. 18). | Units complete | A |
| 49 | C | Lambda path: accuracy, calibration, violation rate and magnitude; CKINN against XGB-mono. | The cost and effect of constraint strength. | Table and plot | A |
| 50 | C | Calibration plots, decision curves, intervals; AUPRC and threshold metrics. | pp. 15 to 16, 22. | Figures saved | L; added metrics A |
| 51 | C | S1 diagnosis-only outcome; S2 and R5 complete cases and missingness; S4 extras (family history, smoking; HbA1c, waist, exercise, fatty liver, GGT); E4 adds LDL and alcohol. | Robustness; the only test of the family-history, HbA1c and exercise priors. | 1,304 events in S1 | L; E4 is A |
| 52 | C | Gate G. | Stop point. | Pass | L |

### Stage H. Transport (cells 53 to 62)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 53 | C | R2 leave-one-site-out, T1 to T3, 12 test sites; subgroup metrics at T2. | Hospitals A to C, then D (p. 12; s. 8). | 12 sites per tier | L, D |
| 54 | C | R4: train under 45, test 45 and over; train BMI under 25, test 25 and over. | Real covariate shift (s. 8). | 6 units | L |
| 55 | C | Summary tables and site-level plot. | Evidence for C2. | Saved | L |
| 56 | C | Pre-external gate: lock hash, R1 complete, checkpoints hashed, code frozen. | NAGALA outcomes meet a RICH-trained model only after this. | Pass | L |
| 57 | C | R3 RICH to NAGALA, T1 to T4, unchanged predictions. | RQ3, RQ4 (s. 8). | Refuses to run on hash mismatch | L |
| 58 | C | R3 NAGALA to RICH, T1 to T3, unchanged predictions. | Both directions are required by the decision rule (s. 11). | Units complete | L |
| 59 | C | Counts events in the 1,000-participant recalibration sample and reports the precision of the slope; then runs the locked recalibration and evaluates on the rest. | Recalibrated results are model updating, not external validation, and are tabulated separately. | Event counts printed before fitting | L; audit is A |
| 60 | C | Paired bootstrap (200 resamples) on raw predictions; subgroups at T2; S3 without sex; calibration and decision-curve figures. | The intervals named in section 11. | Both directions | L |
| 61 | C | Two-year landmark (conditional risk) and RICH restricted to baseline glucose below 6.1 mmol/L. | Aligns follow-up design and eligibility. | 12,513 at risk; 204,640 with 2,238 events | A |
| 62 | C | Gate H. | Stop point. | Pass | L |

### Stage I. Explainability (cells 63 to 69)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 63 | C | Integrated Gradients on the log hazard and on 3-year risk (baseline: training medians); SHAP on 3-year risk where the package installs. | pp. 13 to 14. | Attributions sum to the output difference | A |
| 64 | C | Partial-dependence, individual and accumulated-local-effect curves for each constrained variable, MLP against CKINN. | Monotonicity analysis (p. 13). | Curves saved | A |
| 65 | C | What-if changes: one variable, within the 1st to 99th percentile for the same sex and age band; age and sex never altered. | Counterfactual and perturbation (pp. 13 to 14). | Feasibility rule enforced | A |
| 66 | C | Fidelity (replacing the top-attributed features changes the output more than replacing random ones); stability across folds; agreement between methods and between cohorts; violation rate and magnitude. | "SHAP consistency, feature stability" (p. 22); H4. | Table saved | A |
| 67 | C | Patient-level report template: risk, main factors, consistency with the priors, a what-if line; wording as model behaviour. | "Clinical reasoning explanation"; no advice (p. 14). | Wording check | A |
| 68 | C | Builds a blinded review pack and rubric; computes agreement only from rating files that exist. | "Clinical expert agreement" (p. 22). | Pack written; analysis blocked without ratings | G |
| 69 | C | Gate I. | Stop point. | Pass | L |

### Stage J. Fairness (cells 70 to 74)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 70 | C | By sex, age band, RICH site and cohort: discrimination, calibration, threshold metrics, with intervals and an adequacy flag. | pp. 14 to 15; RQ6. | Underpowered groups flagged | L (T2), A |
| 71 | C | Gaps between groups for MLP and CKINN, with bootstrap intervals. | H5. | Table saved | A |
| 72 | C | Fairness-aware CKINN: adds mu times the variance across groups of the group Cox loss; mu chosen in the inner split. | Model 5 (p. 18). Measurement alone is not optimization. | Toy test; units complete | A |
| 73 | C | Accuracy, calibration and gap trade-offs for MLP, CKINN and the fairness-aware variant. | What fairness costs or gains. | Table and plot | A |
| 74 | C | Gate J. | Stop point. | Pass | L |

### Stage K. Further proposal components (cells 75 to 83)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 75 | C | Aligned CKINN: training rows weighted by the estimated odds of belonging to the target cohort, from covariates only; effective sample size reported. | "Population alignment" (p. 19); Model 4 (p. 18). | No target outcome used | A |
| 76 | C | Table of Models 1 to 5 on identical folds. | The mandatory ablation (p. 18). | Five rows | A |
| 77 | C | Knowledge graph: nodes, edges, signs, feature-to-node map; every edge needs a reference verified in PubMed or Scite or is flagged unverified. | pp. 24 to 25. | No unverified edge enters the model | A |
| 78 | C | Graph-structured CKINN: connections follow the graph; same penalty; compared with CKINN. | "Clinical data + knowledge graph + network" (p. 25). | Toy test; units complete | A |
| 79 | C | Metrics within baseline strata: below 5.6 mmol/L (and HbA1c below 5.7% in NAGALA), and prediabetes range. | RQ7; prediabetes to diabetes (p. 4). | From saved predictions | A |
| 80 | C | Healthy to prediabetes in RICH: status at the final visit among those below 5.6 mmol/L, modelled with follow-up length as offset; linear, MLP and CKINN. | Healthy to prediabetes (p. 4). | 185,586 at risk; no external validation possible | A |
| 81 | C | Readiness register: for the full multistate model, the current-diabetes classifier, the glucose-insulin model and extra fairness groups, lists the variables required, tests the workbooks for them, and records the blocking reason. | pp. 3, 4, 10 to 11, 15, 26. | Readiness file | G |
| 82 | C | Third-cohort intake: validates a supplied file against the feature registry and, if it passes, runs it through the cross-cohort function, including pooled training. | pp. 16 to 17. | "No cohort supplied" until one is | G |
| 83 | C | Gate K. | Stop point. | Pass | L |

### Stage L. Decisions and export (cells 84 to 88)

| Cell | Type | What it does | Why | Output and check | Status |
| --- | --- | --- | --- | --- | --- |
| 84 | C | Applies protocol section 11 to C1 to C4. | No judgement after results. | Four verdicts with intervals | L |
| 85 | C | Evidence matrix for RQ1 to RQ7 and H1 to H5, filled from result files, each labelled confirmatory, exploratory or blocked. | Every question answered or marked unanswerable (pp. 20 to 21). | 12 rows | L |
| 86 | C | Audit: units, manifests, hashes, deviations, approvals; every register row has evidence or a blocking reason. | The reported study is the planned one. | No gap | L |
| 87 | C | Tables, figures and number macros in the format of `scripts/make_report.py`; results bundle. | Numbers in the paper come from `results/` only. | Bundle written | L |
| 88 | M | Limitations and what is not claimed. | Required for thesis and paper. | None | L |

Totals: 88 cells, 86 code and 2 markdown; 11 gates.

## 9. Explainability verification

| Item | Definition |
| --- | --- |
| Targets | Log hazard for constraint checks and attributions; 3-year absolute risk for curves, SHAP, what-if and reports. Each output names its target. |
| Integrated Gradients | Straight path from training medians, 64 steps; check that attributions sum to the output difference. |
| SHAP | Model-agnostic explainer on 3-year risk, 200 background rows from training data, 1,000 participants (seed 0). |
| Stability | Spearman correlation of feature rank (mean absolute attribution) between folds. |
| Cross-cohort consistency | The same correlation between RICH test data and NAGALA for the RICH-trained model. |
| Fidelity | Change in output when the 1, 2 or 3 top-attributed features are set to the training median, minus the change for the same number of random features. |
| Violation rate and magnitude | Share of participants, and mean size, of a change against the prior after +0.25 SD. |
| What-if rule | One variable; values inside the 1st to 99th percentile of the same sex and age band; BMI, SBP, triglycerides and glucose only. |
| Expert rubric | Five items scored 1 to 5 (direction plausible; main factors appropriate; no implausible factor; wording clear and not advice; overall confidence); 60 cases by two models, blinded; weighted kappa. Reported only from real ratings by at least two clinicians. |
| Limits | Attributions describe the model, not causes; a lower violation rate is not clinical benefit. |

## 10. Approval and execution

Cells and parts marked L or D need no approval (cells 11 and 13 are descriptive counts with no model). Cells and parts marked A need your approval of the matching item in `protocol/amendment_A1_PROPOSED.md` (and your supervisor's, where you require it). Nothing marked A can change a C1 to C4 verdict. One cell is issued at a time; its output is checked against the column above before the next is issued.
