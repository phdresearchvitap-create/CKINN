import sys
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
A, R, B = RUN / "A" / "v4", RUN / "R" / "v4", RUN / "B_as_coded" / "v4"


def rd(p):
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if Path(p).exists() else None


def fmt(v, d=4):
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "NA"
        return f"{v:.3e}" if v != 0 and abs(v) < 1e-3 else f"{v:.{d}f}"
    return str(v)


def table(df, cols, d=4):
    df = df[[c for c in cols if c in df.columns]]
    out = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return out + "".join("| " + " | ".join(fmt(v, d) for v in r) + " |\n" for r in df.itertuples(index=False))


sections = ["# Results digest, protocol v4 (generated from the run's tables by make_digest.py)\n",
            f"Run folder: `{RUN.name}`. Intervals are 95% participant-bootstrap percentile intervals, conditional on the fitted models.\n"]


def add(title, path, cols, d=4, q=None):
    df = rd(path)
    if df is None:
        sections.append(f"## {title}\n\nNot produced in this run.\n")
        return
    if q is not None:
        df = q(df)
    sections.append(f"## {title}\n\n" + table(df, cols, d))


MAIN = ["M1", "CKINN", "CKINN_FPG_BMI", "M4", "M5", "LR", "RF", "XGB", "LGB", "SVM", "Cox", "reference"]
order = lambda df: df.assign(_o=df.model.map({m: i for i, m in enumerate(MAIN)})).sort_values("_o").drop(columns="_o")
add("1. Internal validation in RICH (pooled out-of-fold)", A / "internal_pooled_oof.csv",
    ["model", "n", "events", "brier", "auroc", "average_precision", "cal_intercept", "cal_slope"], q=order)
add("2. Selected λ on all RICH participants (one-standard-error rule)", A / "lambda_path_final.csv",
    ["model", "lambda_value", "inner_cv_brier", "excess_over_best", "paired_se", "within_one_se", "selected"], 6)
add("3. External validation RICH → NAGALA", A / "external_metrics_ci.csv",
    ["model", "n", "events", "brier", "brier_lo", "brier_hi", "auroc", "auroc_lo", "auroc_hi", "average_precision", "average_precision_lo",
     "average_precision_hi", "cal_intercept", "cal_intercept_lo", "cal_intercept_hi", "cal_slope", "cal_slope_lo", "cal_slope_hi",
     "sensitivity", "specificity", "ppv", "npv", "f1"], q=order)
add("4. H1/H2: paired CKINN − M1 in NAGALA", A / "contrast_ckinn_vs_m1.csv", ["metric", "CKINN", "M1", "CKINN_minus_M1", "lo", "hi", "h1_decision"], 6)
add("5. CKINN against every model in NAGALA", A / "contrast_ckinn_vs_all.csv", ["comparison", "metric", "estimate", "lo", "hi"], 6)
add("6. Index of prediction accuracy in NAGALA", A / "ipa.csv", ["model", "brier", "ipa", "lo", "hi"], q=order)
add("7. Net benefit per 1,000 in NAGALA", A / "net_benefit.csv", ["model", "threshold", "nb_1000", "lo_1000", "hi_1000"], 3,
    q=lambda d: d.assign(nb_1000=1000 * d.net_benefit, lo_1000=1000 * d.lo, hi_1000=1000 * d.hi))
add("8. Reverse direction NAGALA → RICH: internal (NAGALA out-of-fold)", R / "internal_pooled_oof.csv",
    ["model", "n", "events", "brier", "auroc", "average_precision", "cal_intercept", "cal_slope"], q=order)
add("9. Reverse direction NAGALA → RICH: external", R / "external_metrics_ci.csv",
    ["model", "n", "events", "brier", "brier_lo", "brier_hi", "auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope"], q=order)
add("10. H3: paired CKINN − M1 in RICH (NAGALA-trained)", R / "contrast_ckinn_vs_m1.csv", ["metric", "CKINN", "M1", "CKINN_minus_M1", "lo", "hi"], 6)
add("11. H3: leave-one-site-out, per site", A / "loso_ckinn_vs_m1.csv", ["site", "metric", "n", "events", "CKINN_minus_M1", "lo", "hi"], 6)
add("12. H3: leave-one-site-out, random-effects pooled", A / "loso_pooled.csv", ["metric", "k", "pooled", "lo", "hi", "tau2", "i2", "sites_ckinn_better"], 6)
add("13. H4: wrong-way gradient share (all participants)", A / "directions_gradient.csv", ["cohort", "model", "feature", "constrained", "eligible", "wrong_way_share"], 3,
    q=lambda d: d[d.model.isin(["M1", "CKINN", "CKINN_FPG_BMI"])])
add("14. H4: SHAP importance (share of total)", A / "shap_importance.csv", ["model", "feature", "mean_abs_shap", "share_of_total", "rank"])
add("15. H4: SHAP direction agreement with the sign table", A / "shap_direction.csv", ["model", "feature", "prior_sign", "spearman_value_vs_shap", "agrees_with_prior"], 3)
add("16. H4: seed stability of SHAP importance", A / "shap_stability.csv", ["model", "seed_a", "seed_b", "spearman"], 3)
add("17. H4: counterfactuals (modifiable factor at development median)", A / "counterfactual_summary.csv",
    ["model", "factor", "applicable", "mean_risk_change", "share_risk_decreased", "share_risk_increased"])
add("18. H5/RQ7: group performance", A / "fairness_groups.csv", ["stage", "model", "group", "n", "cases", "status", "sensitivity", "specificity", "fnr", "auroc", "auroc_lo", "auroc_hi", "cal_intercept"], 3,
    q=lambda d: d[d.model.isin(["M1", "CKINN"])])
add("19. H5: group gaps", A / "fairness_gaps.csv", ["stage", "model", "group_type", "supported_groups", "sensitivity_gap", "specificity_gap", "fnr_gap", "auroc_gap", "abs_cal_intercept_gap"], 3)
add("20. Sensitivity analyses", A / "sensitivity.csv", ["analysis", "model", "brier", "brier_lo", "brier_hi", "auroc", "cal_intercept", "cal_slope"])
add("21. Exploratory: training-size curve", A / "size_curve_summary.csv",
    ["size", "draws", "cases", "mean_brier_minus_M1", "sd_brier_minus_M1", "draws_ckinn_lower_brier", "mean_auroc_minus_M1", "sd_auroc_minus_M1"], 6)
add("22. NHANES current status (2017–2020 → 2021–2023)", B / "external_metrics_ci.csv", ["model", "n", "events", "brier", "auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope"], q=order)
add("23. Proposal questions and hypotheses", A / "proposal_questions_evidence.csv", ["item", "decision", "estimate"])
OUT.write_text("\n".join(sections), encoding="utf-8")
print("wrote", OUT)
