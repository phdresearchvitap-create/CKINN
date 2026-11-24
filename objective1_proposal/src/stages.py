from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
from data import CORE, NONLAB, PLUS, UNITS, PERIODS, R, J, csv, js, restore, digest, key, allowed
from stats import support, ReverseKM
from models import Net, Prep, seed_all
from runner import VARIANTS, env_info
from analysis import (
    external_contrast,
    figures,
    subgroups,
    explain,
    updating,
    evidence,
    export,
    internal_summary,
)
from verify import run_checks, Checks
import unittest

GROUPS = {
    "AUDIT": list(range(5, 14)) + [33],
    "CHECKS": [14, 17, 20],
    "INTERNAL": [21, 22],
    "FINAL": [23, 24],
    "EXTERNAL": [25, 26, 27],
    "ABLATIONS": [28],
    "SENSITIVITY": [29],
    "EXPLAIN": [30],
    "SUBGROUPS": [31],
    "UPDATE": [32],
    "SCREEN": [33, 34, 35],
    "REPORT": [36, 37, 38, 39],
}
TRAIN_STAGES = set(range(21, 33)) | {34, 35}
SELECTION_COLS = ["model", "lambda_value", "epochs", "source_cv_brier", "candidate_index", "threshold"]
METRIC_COLS = [
    "model",
    "n",
    "events",
    "brier",
    "auroc",
    "average_precision",
    "predicted",
    "observed",
    "cal_slope",
    "sensitivity",
    "specificity",
]


def show(path):
    p = Path(path)
    if not p.exists():
        print(f"Unavailable in this run: {p.name}")
        return
    if p.suffix == ".csv":
        display(pd.read_csv(p).head(25))
    elif p.suffix == ".png":
        display(Image(filename=str(p)))
    else:
        print(p.read_text(encoding="utf8")[:5000])


def support_table(st):
    result = []
    for name in ["RICH", "NAGALA"]:
        d = st.data(name)
        for site, part in [("ALL", d)] + list(d.groupby("site")):
            row = dict(cohort=name, site=site)
            try:
                row.update(support(part), status="estimated")
            except ValueError as e:
                row.update(
                    n=len(part),
                    events=int(((part.event == 1) & (part.time <= 3)).sum()),
                    controls=int((part.time > 3).sum()),
                    unknown=int(((part.event == 0) & (part.time <= 3)).sum()),
                    G3=float(ReverseKM(part.time, part.event).at(3)),
                    at_risk_3=int((part.time >= 3).sum()),
                    status=str(e),
                )
            result.append(row)
    csv(st.table / "A" / "ipcw_support.csv", result)
    return pd.DataFrame(result)


def test_cases(names):
    suite = unittest.TestSuite((Checks(n) for n in names))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Stage check failed")


def run(n, st, package):
    opt = st.cfg["options"]
    stage = st.cfg["stage"]
    if stage != "ALL" and stage not in GROUPS:
        raise ValueError("Unknown RUN_STAGE: " + stage)
    if stage != "ALL" and n not in GROUPS[stage]:
        return
    if st.mode == "AUDIT" and n in TRAIN_STAGES:
        print("Skipped in AUDIT mode.")
        return
    if n == 5:
        print(pd.Series(st.identity["data"]).to_string())
        st.audit()
    elif n == 6:
        info = json.loads((st.auditpath / "inventory.json").read_text(encoding="utf8"))
        for sheet in info["workbooks"]:
            print(sheet)
        print(info["zip_members"], info["consistency_report"])
    elif n == 7:
        for f in ["structure", "raw_features", "formula_cache", "unit_checks"]:
            show(st.auditpath / (f + ".csv"))
    elif n == 8:
        display(pd.DataFrame([dict(input=k, RICH=R[k], NAGALA=J[k], unit=UNITS[k]) for k in PLUS]))
    elif n == 9:
        show(st.auditpath / "cohort_flow.csv")
    elif n == 10:
        display(support_table(st))
    elif n == 11:
        print("Primary inputs", allowed(CORE, "A"), "Screening", allowed(NONLAB, "B"))
        if st.mode != "AUDIT":
            st.split(st.data("RICH"), st.b["outer"], "A_RICH_nested", st.seed)
    elif n == 12:
        print(
            Prep.__doc__ or "Preprocessor loaded. Imputation and scale are fitted inside each training split."
        )
    elif n == 13:
        print("A case at exactly year 3 is included. A censoring at exactly year 3 has unknown status.")
    elif n == 14:
        test_cases(["test_boundaries_ties_hand_weights", "test_loss_denominator_and_penalty"])
    elif n == 15:
        print("The primary Brier denominator is the entire evaluation cohort.")
    elif n == 16:
        display(Image(filename=str(Path(package) / "architecture.png")))
    elif n == 17:
        test_cases(
            ["test_loss_denominator_and_penalty", "test_derivative_sign_masks_and_raw_finite_difference"]
        )
    elif n == 18:
        print(st.cfg["models"], "Comparator candidate budget", st.b["comparator_trials"])
    elif n == 19:
        print(st.b, "Resume directory", st.private)
    elif n == 20:
        js(st.report / "software_checks.json", run_checks())
    elif n == 21:
        display(st.internal().head())
    elif n == 22:
        show(st.table / "A" / "A_RICH_nested.csv")
        display(internal_summary(st))
    elif n == 23:
        display(st.selection_table("A")[SELECTION_COLS])
    elif n == 24:
        print(st.freeze("A"))
    elif n == 25:
        display(st.score("A")[METRIC_COLS])
    elif n == 26:
        display(pd.Series(external_contrast(st)))
        if opt["ablations"]:
            display(pd.Series(external_contrast(st, matched=True)))
    elif n == 27:
        figures(st, "A")
        show(st.fig / "A" / "calibration_roc_pr.png")
    elif n == 28:
        if opt["ablations"]:
            p = st.table / "A" / "external_metrics.csv"
            if not p.exists():
                raise RuntimeError("Complete FINAL and EXTERNAL before ABLATIONS")
            df = pd.read_csv(p)
            if not set(VARIANTS).issubset(df.model):
                raise ValueError("Component fits missing from frozen manifest")
            csv(st.table / "A" / "ablations.csv", df[df.model.isin(VARIANTS + ["M1", "M2", "M3"])])
            display(pd.read_csv(st.table / "A" / "ablations.csv")[METRIC_COLS])
        else:
            print("Component analyses disabled in this run.")
    elif n == 29:
        for option, src, site in [
            ("site_holdout", "RICH", True),
            ("broader_rich", "RICH_broad", False),
            ("alt_zero", "RICH_alt0", False),
        ]:
            if opt[option]:
                display(st.internal(source=src, site=site).head())
            else:
                print("Disabled", option)
        print("Site-IPCW sensitivity is fitted in stage 21 when enabled:", opt["site_ipcw"])
    elif n == 30:
        if opt["explain"]:
            explain(st, "A")
            show(st.table / "A" / "directions.csv")
        else:
            print("Explanations disabled.")
    elif n == 31:
        if opt["subgroups"]:
            display(subgroups(st, "A").head())
        else:
            print("Subgroups disabled.")
    elif n == 32:
        if opt["target_update"]:
            display(pd.DataFrame(updating(st)))
        else:
            print("Target updating disabled.")
    elif n == 33:
        show(st.auditpath / "cohort_flow.csv")
        print(
            "NHANES analysis label",
            st.tag("B"),
            "Producer recode evidence:",
            opt["producer_recode_evidence"] or "unavailable",
        )
    elif n == 34:
        if opt["screening"] == "off":
            print("Screening disabled.")
        else:
            print(st.freeze("B"))
            display(st.selection_table("B", frozen=True)[SELECTION_COLS])
    elif n == 35:
        if opt["screening"] == "off":
            print("Screening disabled.")
        else:
            display(st.score("B")[METRIC_COLS])
            display(pd.Series(external_contrast(st, "B")))
            figures(st, "B")
            if opt["subgroups"]:
                subgroups(st, "B")
            if opt["explain"]:
                explain(st, "B")
    elif n == 36:
        for p in sorted(st.table.rglob("*.csv")):
            print(p.relative_to(st.table))
        if (st.log / "fits.jsonl").exists():
            fits = pd.read_json(st.log / "fits.jsonl", lines=True)
            csv(
                st.table / "fit_budget.csv",
                fits.groupby(["task", "model", "status"], dropna=False)
                .agg(fits=("model", "size"), seconds=("seconds", "sum"))
                .reset_index(),
            )
    elif n == 37:
        evidence(st)
        show(st.report / "summary.md")
    elif n == 38:
        evidence(st)
        show(st.report / "run_manifest.json")
    elif n == 39:
        print(export(st, package))
