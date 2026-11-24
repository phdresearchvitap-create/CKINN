# Conformance of the protocol v4 notebook with the research proposal (CKINN_OBJ1, sections 1–28)

Status values:

* **Implemented:** the analysis follows the proposal.
* **Adapted:** the proposal's intent is followed within the limits of the data, and the reason is given.
* **Not possible:** the data cannot support the item.

The last column records what changed from the previous notebook (CKINN_Objective1_Final, protocol v2).

| § | Proposal requirement | Status | Implementation in v4 | Change from the previous notebook |
|---|---|---|---|---|
| 1–2 | Patient data + clinical knowledge → diabetes risk; clinically plausible relationships | Implemented | Knowledge loss on six clinical directions; direction audits (stage EXPLAIN) | Previously two directions, inactive in practice |
| 3A | Current diabetes classification | Adapted | NHANES: HbA1c ≥ 6.5% among participants without a reported diagnosis, trained 2017–2020, frozen, evaluated 2021–2023 | CKINN now constrained by age, BMI and blood pressure |
| 3B | Future risk P(D_{t+1} = 1 \| X_t) | Implemented | Recorded incident diabetes within three years (IPCW) | Unchanged |
| 3C | Progression healthy → prediabetes → diabetes | Not possible | No repeated dated glycaemic states in the files | Unchanged |
| 4 | Risk prediction, not diagnosis | Implemented | Outputs worded as risk estimates; text explanations say they are not advice | Unchanged |
| 5 | Clinically justified inputs; leakage, missingness and availability assessed | Adapted | Eight inputs shared by both cohorts plus missingness indicators. HbA1c, family history, physical activity and waist are not shared, so they cannot be used across populations | Unchanged |
| 6 | Clinical sign table (BMI, FPG, HbA1c, family history, hypertension, TG ↑; physical activity, HDL ↓; previous prediabetes, age ↑) | Implemented for shared factors | Age ↑, BMI ↑, SBP ↑, FPG ↑, TG ↑, HDL ↓. Unshared factors are listed as not constrained | **Changed:** previously FPG and BMI only |
| 7 | Demographic, clinical and laboratory branches → network → risk head + constraints → XAI | Implemented | Branched network; knowledge loss; explanation layer (M2) | Unchanged |
| 8–9 | L_total = BCE + λ Σ w_j max(0, −∂p̂/∂x_j)² | Implemented | IPCW BCE + λ × mean wrong-way squared probability derivative over six factors, w_j = 1, observed in-support values | **Changed:** λ grid {0 … 10⁵} so the penalty can act; previously λ ≤ 100, inactive |
| 10 | Soft constraints: respect knowledge, allow exceptions from strong evidence | Implemented | λ chosen by a paired one-standard-error rule: the largest λ the data do not clearly reject | **Changed:** previously a joint search tied λ to architecture trials |
| 11 | Physiological glucose–insulin knowledge | Not possible | No insulin or dynamic measurements | Unchanged |
| 12 | Train on populations or hospitals, test on a new one | Implemented | RICH → NAGALA; NAGALA → RICH; leave-one-site-out across RICH sites with random-effects pooling | **Added:** reverse direction; leave-one-site-out for CKINN with pooling |
| 13 | Generalisation, calibration, fairness and explainability across populations | Implemented | Every analysis is reported in both directions where data allow | Extended |
| 14 | SHAP, integrated gradients, counterfactuals, perturbation, monotonicity analysis; patient contribution shares | Implemented | Permutation SHAP (seed-averaged) for M1, CKINN, LR and XGB; integrated gradients; finite-step and gradient direction checks; counterfactuals; contribution shares in the reasoning text | **Added:** SHAP for the networks, counterfactuals, contribution shares |
| 15 | Clinical-reasoning explanation and counterfactual, never medical advice | Implemented | Template text for the highest-risk participants, stored privately; summary table exported | **Added** |
| 16 | Group sensitivity, specificity, calibration, FNR, FPR, AUROC, AUPRC (sex, age, geography, socio-economic) | Adapted | Sex, age groups, glycaemic stratum, cohort and site. Socio-economic variables are absent | Intervals for M1 and CKINN; gaps table |
| 17 | Calibration curve, intercept, slope, Brier | Implemented | Decile curves with 95% intervals (distinct markers), intercept, slope, ECE, Brier, IPA | **Fixed:** overlapping curves; **added** intervals |
| 18 | Dataset A + B → training; Dataset C → external validation | Adapted | Two longitudinal cohorts: each direction plus site-level validation. NHANES has no follow-up, so a third incident-outcome cohort does not exist | Reverse direction added |
| 19 | LR, RF, XGBoost, LightGBM, SVM, MLP, CKINN | Implemented | All, plus ridge Cox; six tuning trials each | **Changed:** previously two trials for comparators |
| 20 | Ablation M1–M5 | Implemented | M1, M2, M3 = CKINN, CKINN-FPG-BMI (sign-table ablation), M4 (site balancing), M5 (fairness) | M3 redefined as the six-direction CKINN |
| 21 | Cleaning, missing data, feature engineering, population alignment, network, constraints, XAI | Implemented | Audit, training-only imputation, harmonisation and eligibility alignment (FPG < 6.1), network, constraints, XAI | Unchanged |
| 22 | RQ1–RQ7 | Implemented, RQ7 adapted | Mapped in `proposal_questions_evidence.csv` | Regenerated for v4 |
| 23 | H1–H5 | Implemented | H1 confirmatory (paired Brier); H2–H5 descriptive with intervals | Extended estimates |
| 24 | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier, calibration, external validation, group metrics, SHAP consistency, feature stability, expert agreement | Implemented except expert agreement | All metrics with 95% intervals; SHAP direction agreement; stability across three seeds. No clinician panel | **Added:** intervals, seed stability |
| 25 | Honest novelty statement | Implemented | Contribution framed as a knowledge-constrained network evaluated across populations, with null and positive findings reported | — |
| 26 | Clinical knowledge graph | Not possible | Requires a curated evidence graph | Unchanged |
| 27 | Four layers: prediction, clinical knowledge, generalisation, explainability | Implemented | Stages INTERNAL–EXTERNAL, CKINN, REVERSE/LOSO, EXPLAIN | — |
| 28 | Title and central question | Implemented | Notebook title and introduction | — |
