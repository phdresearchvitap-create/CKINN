# Coverage of the 28-step protocol and the original proposal by the real-data notebook

**Updated 2026-11-21 after the post hoc block (cells PH1 to PH4, label PH).** The first table gives the current status of every step; the original review (before the post hoc block) follows unchanged below it, so the change can be traced. Post hoc analyses were decided after the v2 results had been seen; they cannot support a confirmatory claim and do not change any v2 verdict (`protocol_v2.md`, section 10).

## Current status (after the post hoc block)

| Status | Steps | Count |
| --- | --- | --- |
| Covered (v2 or post hoc) | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27 | 22 |
| Partly covered (the rest is not possible on real data or needs people) | 12, 13, 25 | 3 |
| Not covered (a decision, not an analysis) | 26 | 1 |
| Not possible on real data | 11 | 1 |
| Not in the PDF | 28 | 1 |

| Step | Status now | What closed it, and what remains |
| --- | --- | --- |
| 1 | Covered | protocol_v2.md (unchanged). |
| 2 | Covered (PH) | BMI 24 versus 35 in the same held-out people and model-agnostic violation rates for every model (tables PH05, PH06), beside the derivative-based rates of M1, M2, CKINN. Caveat: BMI 35 with an unchanged waist is an implausible combination (ERRORS B8). |
| 3 | Covered | A (NHANES) and B (NAGALA, RICH) in v2; B with glucose and HbA1c is tier T3; C as a one-step progression in RICH (table PH13). |
| 4 | Covered | Unchanged. |
| 5 | Covered (PH) | Predictor audit table with missingness, leakage risk, fairness relevance and prior sign (PH19); proposal variables the data lack (PH20). |
| 6 | Covered (PH) | One PubMed-verified reference per row (references.md, PH21), found after the table was frozen; DBP constrained in a post hoc CKINN variant (PH07). The headline table still leaves DBP unconstrained, as frozen. |
| 7 | Covered | Unchanged. |
| 8 | Covered (PH) | LR, M2 and CKINN with balanced class weights against unweighted (PH07, variant `class_weights`). |
| 9 | Covered | Unchanged; the grid differs from the card's. |
| 10 | Covered (PH) | Older-and-lean subgroup across the lambda path (PH11); not estimable where it has fewer than 20 events. |
| 11 | Not possible on real data | No simulated physiological state exists. |
| 12 | Partly | Train-by-test matrix for every model, seed 0 (PH09). The S1+S2-to-S3 run remains impossible: NHANES has a different outcome. |
| 13 | Partly | The card's primary endpoint (on S3 at 30% missingness) is not computable on real data; v2 froze H1 instead. |
| 14 | Covered (PH) | SHAP for LR, XGBoost, monotone XGBoost, M1, M2, CKINN (PH15 to PH17, PH23), partial-dependence plots (figure PH1), integrated gradients (v2). Fidelity to generator weights is impossible on real data. |
| 15 | Covered (PH) | Reasoning text for 50 people per cohort with a sign-consistency flag and a blood-panel counterfactual worded as model behaviour (summary PH18; texts in the local cache only). |
| 16 | Covered (PH) | AUPRC and false-negative rate by group (PH14); socio-economic groups do not exist in the data. |
| 17 | Covered | v2, plus intercept-only recalibration (PH10), estimated on 1,000 target participants; with few events in that sample the shift is imprecise, and PH10 shows the sample. |
| 18 | Covered | Unchanged. |
| 19 | Covered (PH) | SVM added (PH01, PH03), 5 repeats of 5-fold CV. |
| 20 | Covered (PH) | M4 pooled training with a cohort indicator (PH08, within the two training cohorts only), M5 group weights and an HDL wrong-sign control (PH07). |
| 21 | Covered (PH) | TG/HDL ratio with its own sign (PH07); standardisation stays on training statistics only (ERRORS B16). |
| 22 | Covered (PH) | RQ7 among people with baseline fasting glucose below 5.6 mmol/L (PH12). |
| 23 | Covered | Unchanged. |
| 24 | Covered (PH) | One scorecard (PH22); clinical expert agreement is not possible without clinicians. |
| 25 | Partly | Dated PubMed search logged (NOVELTY_SEARCH.md); a systematic search in Embase, Scopus, IEEE Xplore, ACM and arXiv is still needed. |
| 26 | Not covered | The decision to put a knowledge graph in or out of paper 1 is the investigator's and supervisor's. |
| 27 | Covered (PH) | Layer 4 (reasoning text) and the scorecard added. |
| 28 | Not in the PDF | The 28-step PDF stops in step 27; the proposal's section 28 is the title. |

Proposal-only items now: one-step progression done (RICH, PH13); NHANES "told of prediabetes" and "told of hypertension" tried as inputs (PH07, exploratory); glucose-insulin equations and knowledge graph not attempted; no Indian cohort; expert agreement impossible without clinicians.

---

# Original review (before the post hoc block)

Prepared 2026-11-21 from: `CKINN_OBJ1.pdf` (original proposal, 26 pages), `CKINN_28-Step_Protocol.pdf` (11 pages), `CKINN_vs_MLP_real_data_final.ipynb` (44 cells, byte-identical to the repository's v2 clean notebook, no outputs), and the stored v2 results in `outputs_v2_full/`. Statuses come from reading the notebook code and the stored tables, not from the notebook's own claims. Code occurrences were also searched (for example SHAP, SVM, counterfactual, class weights: none found in the notebook).

## What the documents are

- **CKINN_OBJ1.pdf** is a ChatGPT transcript with 28 numbered sections: idea, tasks, variables, clinical sign table, architecture, loss, soft constraints, physiology, cross-population transport, explainability, fairness, calibration, datasets, model comparison, ablation, framework, research questions RQ1 to RQ7, hypotheses H1 to H5, evaluation, novelty, knowledge graph, final architecture and title.
- **CKINN_28-Step_Protocol.pdf** turns each section into a card (protocol text, actions, data used, "done when"). It is written for the **synthetic** cohorts S1_JP, S2_CN and S3_US, not for real data. Its own warning box lists what synthetic data cannot show (progression, ethnicity and socio-economic data, physiology results, expert agreement). **The PDF is incomplete:** page 11 ends in the middle of step 27 and there is no card for step 28 (the page footer says "Page 11 of 11"). Re-export it if step 28 matters.
- **The real-data notebook** does not follow the 28 steps; it follows `protocol_v2.md`, which was built from the audit of the first real-data study. So the question below is how far the notebook happens to cover each step.

Mapping used: S1_JP = NAGALA, S2_CN = RICH, S3_US = NHANES. One mismatch matters throughout: the 28-step protocol treats S3 as an external test set for the same task, but NHANES supports a different task (detecting undiagnosed diabetes, not incident diabetes) with a different label, so it cannot serve as an external set for NAGALA and RICH models.

## Summary

| Status | Steps | Count |
| --- | --- | --- |
| Covered | 1, 4, 7, 9, 17, 18, 23 | 7 |
| Partly covered | 2, 3, 5, 6, 8, 12, 13, 14, 16, 19, 20, 21, 22, 24, 27 | 15 |
| Not covered (possible on real data) | 10, 15, 25, 26 | 4 |
| Not possible on real data | 11 | 1 |
| Not in the PDF | 28 | 1 |

## Step by step

| Step | What the protocol asks | What the real-data notebook does | Status |
| --- | --- | --- | --- |
| 1 Basic idea | One-page spec of X, y and constraints before training | `protocol_v2.md` (frozen, hash checked in cell 16) | Covered |
| 2 Key difference | Copy 5,000 held-out rows, change BMI 24 to 35, count pairs whose risk falls; report for MLP, XGBoost and CKINN | Violation rate from input derivatives on all constrained predictors (cells 19, 33), reported for M1, M2 and CKINN only. The BMI 24 to 35 pair test is not run; XGBoost and logistic regression violation rates are not reported | Partly |
| 3 What to predict | A current status (primary), B future risk with FPG and HbA1c as inputs, C progression skipped | A (NHANES) and B (NAGALA, RICH) done. B headline **excludes** glucose and HbA1c (tier T2); the tier with them (T3) is a sensitivity analysis. C not done, although RICH has a final-visit fasting glucose that would allow a one-step progression analysis | Partly |
| 4 Clinical distinction | Output called a risk, not a diagnosis; label definition stated; no diagnostic verdict | Done in code, report and limitations; NHANES label defined (FPG 7.0 mmol/L after 8 h fasting or HbA1c 6.5%) | Covered |
| 5 Input variables | Audit table: availability, leakage risk, missing %, fairness relevance; unavailable variables listed as limitations | Missingness table, feature-set table and leakage guard exist (tables 02 and 05). No single audit table with leakage-risk and fairness columns; unavailable variables are only partly listed | Partly |
| 6 Clinical knowledge layer | Sign table frozen: risk up for age, BMI, waist, SBP, **DBP**, family history, TG (+ FPG, HbA1c for incident); down for HDL, exercise; one literature citation per row | Table frozen, but **DBP is not constrained** (it appears only in the extended-table sensitivity run, where it changes AUROC by -0.003 at most, intervals include zero). **No citation per row** exists in `references.md` | Partly |
| 7 Architecture | Group encoders (value + mask), 2 to 3 hidden layers with dropout, sigmoid head, random feature- and group-level masking | Done: encoders per group, value and mask inputs, dropout, panel-level and feature-level masking in training | Covered |
| 8 Formulation | BCE; report with and without class weights; no resampling; calibrated on held-out split | BCE and held-out calibration done; no resampling. **No class-weighted versus unweighted comparison** | Partly |
| 9 Clinical knowledge loss | Squared hinge on sign-adjusted autograd gradient, observed entries only, weighted sum, lambda in {0, 0.01, 0.1, 1, 10} on validation only | Same loss, observed entries only, all weights 1. The lambda grid differs: {0.03, 0.1, 0.3, 1, 3, 10, 30, 100} plus 0, chosen by a one-standard-error rule; lambda 100 was chosen in every cohort. The lambda path is reported (table 28) | Covered, with the grid changed |
| 10 Soft constraints | Penalty only, no monotone architecture; stress-test the built-in exception (age 65 and over, BMI under 20) over lambda | Penalty only. **Stress test not run**; on real data the same subgroup (older, lean) could be examined as an exploratory check | Not covered |
| 11 Physiology | Optional oracle auxiliary head, simulation only; do not call the model physiology-informed | Correctly absent; no physiology claim | Not possible on real data |
| 12 Cross-population | S1 to S2, S2 to S1, S1+S2 to S3, shared features, change in AUROC and calibration versus within-cohort; matrix for every model | NAGALA to RICH and RICH to NAGALA on 10 shared predictors, gap against the target's own cross-validated AUROC (H3). Pooled NAGALA+RICH training is only evaluated inside each cohort. The S3 external run is not possible. Transport run for LR, XGBoost, monotone XGBoost, M1, M2 and CKINN, not for every model | Partly |
| 13 Research question | Five outcomes plus missingness; one frozen primary endpoint: AUROC and calibration slope on S3 after training on S1+S2 at 30% panel missingness | Outcomes and missingness covered. The stated primary endpoint is **not reproducible** (see mapping note); v2 froze H1 (AUROC non-inferiority, family A versus M2) before results | Partly |
| 14 Explainability | SHAP and Integrated Gradients on CKINN, MLP and XGBoost; partial-dependence plots per constrained column; attribution fidelity (synthetic only) | Integrated Gradients (stability across seeds, direction consistency) and partial-dependence slopes (table 29) done. **No SHAP, no LIME, no counterfactual or perturbation analysis, no partial-dependence plots** (tables only). Fidelity to generator weights is impossible on real data | Partly |
| 15 Beyond SHAP | Clinical-reasoning text template, top three contributors, "consistent with constraints" flag, counterfactuals on blood-panel values for 50 patients | **Not implemented** | Not covered |
| 16 Fairness | Sensitivity, specificity, FNR, FPR, AUROC and AUPRC by sex, age band, cohort and site; one threshold fixed on training data | Sex, age band, RICH site and NHANES ethnic group (descriptive); AUROC, sensitivity at matched specificity, true and false positive rates at the training-derived threshold, calibration (tables 20, 21). **AUPRC by group is not computed.** No socio-economic data exist | Partly |
| 17 Calibration | Curve, intercept, slope, Brier with bootstrap intervals; mean predicted versus observed after transport; optional intercept-only recalibration | Curves with bands, slope, calibration-in-the-large, ECE, Brier with intervals; transport calibration reported; recalibration in the small uses intercept **and** slope, not intercept only | Covered |
| 18 Dataset strategy | Benchmark, independent population, external set; five replicates; loaders ready for NAGALA, RICH, NHANES; Pima not used | Real files loaded and hash-checked; five repeats of 5-fold cross-validation with paired seeds instead of five synthetic replicates; Pima not used | Covered |
| 19 Comparison | LR, RF, XGBoost, LightGBM, **SVM**, MLP, CKINN; identical masks; tune on training folds; mean over five replicates | All present except the **SVM, which v2 dropped** (it remains only in the v1 record). Added: monotone XGBoost and LightGBM, constrained LR, spline additive model, multiple-imputation LR. Same test-time masks for all models | Partly |
| 20 Ablation | M1 MLP; M2 + mask; M3 + constraints; M4 + pooled S1+S2 training **with cohort indicator**; M5 + group-reweighted loss; wrong-sign control on HDL | M1, M2, M3 (= CKINN) done with gains and intervals. M4 pooled without a cohort indicator. **M5 dropped in v2** (v1 had it as CKINN_fair, which did not help). Wrong-sign controls are for age, BMI and all signs, **not HDL** | Partly |
| 21 Framework | Cleaning, mask engine, TG-to-HDL ratio with its own sign, per-cohort standardisation, threshold table, single script | Unit harmonisation, winsorising, mask engine, threshold table (table 17), one notebook. No engineered TG/HDL ratio. Standardisation is from training statistics only (per-cohort own-statistics standardisation was removed deliberately: it used test-fold statistics) | Partly |
| 22 Research questions | RQ1 to RQ7, each pointing to a table or figure | RQ1 to RQ6 map to tables below. **RQ7** (incident diabetes in people with baseline FPG under 5.6 mmol/L) is not analysed | Partly |
| 23 Hypotheses | H1 to H5 by paired bootstrap, accept only if the interval excludes zero | H1 to H5 tested in two families with Holm correction (stricter than the card); H4 and H5 redefined (see AUDIT.md) | Covered |
| 24 Evaluation framework | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier, slope, intercept; stability and fidelity; expert agreement; one scorecard | All metrics present (tables 08, 17); attribution stability yes; fidelity and expert agreement not possible without a generator or clinicians. **No single scorecard table** across metrics and models | Partly |
| 25 Novelty | Fresh dated literature search before submission, novelty paragraph | **Not done.** `references.md` holds the data and methods references only | Not covered |
| 26 Knowledge graph | Optional second paper; record the decision in or out of paper 1 | **No decision recorded.** The condition in the card ("only if M3 shows the constraints leave accuracy on the table") is ambiguously worded; ask the supervisor how to read it | Not covered |
| 27 Final architecture | Four layers (baselines, penalties, transport, reasoning text) and one scorecard on accuracy, calibration, generalisation, fairness, explanation | Layers 1 to 3 present; layer 4 (reasoning text) missing; no single scorecard | Partly |
| 28 Final title | (Original proposal: title and central question.) Missing from the 28-step PDF | The notebook title differs from the proposal title ("CKINN versus M1 and M2 on real data, version 2") | Not in the PDF |

## Research questions and where the real-data results sit

| RQ | Table or figure in `outputs_v2_full/` |
| --- | --- |
| RQ1 accuracy | tables 08 and 18 |
| RQ2 knowledge versus unconstrained | table 09 family A (H1, H2), tables 13 and 28 |
| RQ3 cross-population | tables 31 to 33, figure 1 (H3) |
| RQ4 calibration on an external population | tables 31, 32, figure 2 |
| RQ5 consistent explanations | tables 27 to 30, figure 6 (H4) |
| RQ6 disparities across groups | tables 20 and 21 (H5) |
| RQ7 elevated risk before diagnostic thresholds | none |

## Items in the original proposal that go beyond the 28 steps

- **Section 3C, progression** (healthy to prediabetes to diabetes). Not done; RICH's final-visit glucose would allow one step.
- **Section 5 variables.** Weight and height enter only through BMI. Ethnicity, previous prediabetes, cardiovascular disease, hypertension, medications, diet and sleep are absent from NAGALA and RICH. **The NHANES extract does contain** a prediabetes question (DIQ160, asked from 2005-06 onwards), a hypertension question (BPQ020, all cycles) and a cholesterol question (BPQ080); none is used in the primary analysis. Of the activity questions, PAQ650 (vigorous recreational activity) is used as `exercise_regular`; PAQ605 and PAQ620 are not. Prior prediabetes is on the proposal's own sign table, so this is a real gap for the detection task; note that a prediabetes diagnosis implies an earlier glucose test, which must be argued as a legitimate predictor.
- **Section 7 and 21, patient and clinical explanation, clinical decision support.** Not implemented (the threshold table is the only decision-support item).
- **Section 11 and 26, glucose-insulin differential equations and knowledge graph.** Not attempted; correctly gated.
- **Section 12, population examples.** The proposal names Indian, European and American populations. The data are Japanese, Chinese and American (NHANES, different task). There is **no Indian cohort**, which a reviewer in your setting is likely to raise.
- **Section 24, clinical expert agreement.** Needs real clinician ratings; none exist.

## Data present in this session

NAGALA (15,464 participants, 373 incident cases), Rich Healthcare Group (211,833, 4,174), the NHANES extract (101,316 rows; 37,522 adults 20 and over in cycles 1999 to 2018; 29,410 labelled under the corrected definition, 1,241 positive), the v1 results archive, and the executed v2 results. The earlier synthetic-study notebook (`CKINN_Objective1_Synthetic_Study.ipynb`) is also uploaded; it is the study the 28-step protocol was written for, and I did not compare it with the 28 steps in this review.

## Which gaps can be closed with the data you have

Cheap and mostly additive (each would be recorded in `protocol_v2.md` section 10 as post hoc or exploratory, because the protocol is frozen): the BMI 24 to 35 pair test and violation rates for XGBoost and logistic regression (step 2); a DBP-constrained headline run and per-row citations for the sign table (step 6); class-weighted versus unweighted results (step 8); the older-and-lean stress test over lambda (step 10); SHAP, counterfactual and perturbation analyses and partial-dependence plots (step 14); the 50-patient reasoning text with sign-consistency flag (step 15); AUPRC by group (step 16); the SVM, an M5 group-weighted run and an HDL wrong-sign control (steps 19 and 20); a TG/HDL ratio with its own sign (step 21); RQ7 for people with baseline FPG under 5.6 mmol/L (step 22); a single scorecard (steps 24 and 27); NHANES prediabetes and hypertension as inputs (section 5). Not closable without new data or people: novelty search needs a dated PubMed or Scite search, the knowledge-graph decision is yours and your supervisor's, and expert agreement needs clinicians.
