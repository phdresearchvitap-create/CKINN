# CKINN Objective 1 · Protocol v4 (aligned with the research proposal CKINN_OBJ1)

Written 2026-11-24, before any analysis under this protocol was run. Its SHA-256 and UTC time are recorded in `PROTOCOL_v4.sha256`.

## Why a new version, and what was seen before it

The proposal (CKINN_OBJ1, sections 6–9) specifies CKINN as a neural network trained with the clinical knowledge loss

    L_total = L_data + λ Σ_j w_j max(0, −s_j ∂p̂/∂x_j)²

over the factors of its clinical sign table. The earlier operational protocol (v2) restricted the penalty to two factors (FPG and BMI) and to λ ≤ 100.

Two development runs under v2 (Colab runs 96ccd0057ac85891 and 48d5d0f8032418c2) showed three things:

* The unconstrained network already had no wrong-way FPG or BMI gradients, so the penalty was exactly zero and the CKINN and MLP predictions were nearly identical.
* For the sign-table factors that the unconstrained network contradicted (HDL, SBP), the penalty bound only at λ ≥ 10³ (exploratory analysis E9).
* The constraints changed external error most when training data were scarce (exploratory analysis E10).

These results were seen before this protocol was written. They informed two choices: the upper range of the λ grid, and the use of the full shared sign table. Every v4 result must be reported together with this history.

Some evidence in v4 had not been examined under any version before this protocol was written:

* the reverse direction (development in NAGALA, frozen evaluation in RICH);
* leave-one-site-out validation of the six-direction CKINN across RICH sites;
* seed variation (three network seeds);
* the SHAP, counterfactual, reasoning and stability analyses of the six-direction CKINN.

## Central research question (proposal section 28)

Can clinical knowledge constraints improve the accuracy, calibration, fairness, explainability and cross-population generalisation of neural networks for early type 2 diabetes risk prediction?

## Data and outcome (unchanged from v2)

* **Task A (future risk, proposal §3B).** Recorded incident type 2 diabetes within three years of a baseline examination.
  * Cohorts: RICH (China, 15 site codes) and NAGALA (Japan).
  * Eligibility: age ≥ 20, positive follow-up, binary event, baseline FPG < 6.1 mmol/L (NAGALA additionally HbA1c < 6.5%).
  * Weighting: inverse probability of censoring weights (reverse Kaplan–Meier). Cases get 1/G(T−), controls 1/G(3), unknown status 0.
* **Task B (current status, proposal §3A).** HbA1c ≥ 6.5% among NHANES participants without a reported diagnosis (as coded in the extract), trained on 2017–2020 and evaluated on 2021–2023.

## Predictors and the clinical sign table (proposal §5–§6)

* **Shared predictors:** age, sex, BMI, systolic blood pressure, ALT, fasting plasma glucose, triglycerides and HDL cholesterol, with missingness indicators. Preprocessing is fitted on training rows only.
* **Sign-table factors available in both cohorts (CKINN constraints):** increasing age ↑, BMI ↑, systolic blood pressure ↑ (hypertension), fasting glucose ↑, triglycerides ↑, HDL ↓. All weights w_j = 1. Sex and ALT have no prior.
* **Sign-table factors not shared by both cohorts:** HbA1c, family history, physical activity, previous prediabetes. These are not constrained, and the paper states this.
* **Task B:** the available sign-table factors are age ↑, BMI ↑ and systolic blood pressure ↑.

## Models (proposal §19–§20)

| Name | Definition |
| --- | --- |
| LR, RF, XGB, LGB, SVM | Classical and machine-learning comparators with IPCW sample weights, 6 tuning trials each (equal budget) |
| Cox | Ridge Cox truncated at 3 years; survival benchmark, not in the proposal list |
| M1 (MLP) | Branched network (demographic, body/pressure and laboratory branches), λ = 0; base architecture tuned by inner cross-validation over 6 trials |
| M2 (MLP + XAI) | M1 predictions with the explanation layer |
| M3 = CKINN | M1 base architecture and epochs plus the six-direction knowledge loss; λ selected by the rule below |
| CKINN-FPG-BMI | As M3 with only FPG and BMI constrained (the v2 specification); ablation of the sign table |
| M4 | M3 plus site-balanced training weights (cross-population component) |
| M5 | M4 plus a sex loss-gap penalty, coefficient 0.25 (fairness component) |

**λ selection (paired one-standard-error rule).**

* Grid: λ ∈ {0, 10, 10², 10³, 10⁴, 10⁵}.
* For each λ, compute the pooled inner-validation IPCW Brier contributions on the inner splits of the development data.
* Select the largest λ whose mean contribution exceeds that of the best λ by no more than one paired standard error. The standard error comes from participant-level differences on the same validation rows.

The rule encodes the proposal's soft constraint (§10): the model respects the clinical knowledge unless the data give clear evidence against it. M4 and M5 inherit the selected λ. The networks' final models average three seeds.

## Validation designs (proposal §12, §18)

1. **Internal:** nested 5 × 2 cross-validation in RICH, for every model.
2. **Population A → B:** freeze in RICH, evaluate in NAGALA. This is the primary direction.
3. **Population B → A:** nested 5 × 2 cross-validation in NAGALA, freeze in NAGALA, evaluate in RICH.
4. **Hospitals A, B, C → D:** leave-one-site-out across RICH sites with at least 20 known cases and 20 known controls. For each held-out site, M1, CKINN, LR and XGB are refitted on the other sites with the configurations selected on all RICH participants. Per-site paired differences are pooled with a DerSimonian–Laird random-effects model.
5. **Not possible:** the proposal's "Dataset A + Dataset B → Dataset C". NHANES has no follow-up, so no third cohort with incident outcomes exists.

## Evaluation (proposal §16, §17, §24)

* **Metrics:** AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, FPR, FNR, Brier score, index of prediction accuracy, calibration intercept and slope, ECE, calibration curves, and net benefit at 0.5% to 10%.
* **Intervals:** 1,000 participant bootstrap draws, censoring weights re-estimated per draw, conditional on the fitted models.
* **Thresholds:** the 90th percentile of known controls in the development inner out-of-fold predictions.
* **Fairness:** sex, age bands (20–39, 40–59, 60+), baseline FPG < 5.6 versus ≥ 5.6 and (in RICH) site. Groups need at least 20 known cases and 20 known controls.
* **Explainability:**
  * permutation SHAP for M1, CKINN, LR and XGB on 500 NAGALA participants, with RICH training background;
  * agreement of the SHAP direction with the sign table;
  * wrong-way gradient shares and finite-step direction checks for all six factors;
  * integrated gradients with a completeness check;
  * feature-ranking stability across the three seeds;
  * counterfactual risk changes when modifiable factors move to the RICH training median;
  * template-based clinical reasoning explanations, worded as model behaviour and never as advice.

## Hypotheses and decisions (proposal §23)

* **H1, primary:** Brier(CKINN) − Brier(M1) in NAGALA (population A → B), with its paired 95% interval.
  * Supported if the whole interval is below zero.
  * Contradicted if the whole interval is above zero.
  * Otherwise inconclusive.

  AUROC and AUPRC differences are secondary.
* **H2:** paired differences in |calibration intercept|, |calibration slope − 1| and ECE.
* **H3:** CKINN − M1 Brier and AUROC differences in population B → A and in the pooled leave-one-site-out estimate.
* **H4:** wrong-way gradient shares, SHAP direction agreement and seed stability, CKINN against M1.
* **H5:** group gaps in sensitivity, specificity, FNR, AUROC and calibration, CKINN against M1.
* **RQ7:** performance among people with baseline FPG < 5.6 mmol/L.

H2 to H5 are descriptive, and there is no multiplicity correction because no confirmatory claim is made for them.

## Exploratory (labelled as such)

* Training-size curve: RICH subsamples of 1,000, 3,000, 10,000 and 30,000, 5 draws each, M1 against CKINN with the same selection rule, scored in NAGALA.
* NAGALA restricted to at least two years of follow-up, matching the RICH source design.
* NAGALA with HDL treated as missing.
* Frozen specifications refitted on broader RICH eligibility (FPG < 7.0) and with ALT zeros retained.

## Not measurable with these data

Progression through prediabetes (§3C), physiological glucose–insulin constraints (§11), clinician agreement with explanations (§24), a clinical knowledge graph (§26) and clinical decision support or benefit.
