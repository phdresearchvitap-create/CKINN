# %% [markdown]
# # CKINN: clinical knowledge-informed neural networks for explainable, fair and cross-population prediction of type 2 diabetes risk
#
# **Cell 1. Scope of the study (status L).** This notebook follows `notebooks/MASTER_PLAN.md` (88 cells). The confirmatory plan is `protocol/protocol_v1.0_LOCKED.md`; nothing in this notebook may change it.
#
# **What is predicted.** The absolute risk of incident diabetes within 3 years, in adults without diabetes at baseline. The outcome is diagnosis- or biomarker-defined. The output is a risk estimate, not a diagnosis, and no statement here is medical advice.
#
# **Data (real only).** RICH (China, 211,833 adults, 15 site codes) and NAGALA (Japan, 15,464 adults). No synthetic, simulated or augmented data are used at any stage.
#
# **Estimand.** For a participant with baseline predictors x, risk(3 | x) = 1 - exp(-H0(3) exp(lp(x))), where lp is the model's log relative hazard and H0 is the Breslow baseline cumulative hazard estimated on training data only.
#
# **Predictor tiers.** T1: age, sex, BMI, SBP, DBP. T2 (headline): T1 + total cholesterol, triglycerides, ALT. T3: T2 + fasting glucose. T4: T3 + HDL, AST.
#
# **Research questions of the proposal (pp. 20 to 21).** RQ1 accurate prediction of future risk; RQ2 value of clinical knowledge over an unconstrained network; RQ3 cross-population generalization; RQ4 calibration in an external population; RQ5 clinically consistent explanations; RQ6 disparities across demographic groups; RQ7 risk before diagnostic thresholds are reached.
#
# **Hypotheses of the proposal (p. 21).** H1 CKINN predicts better than the MLP; H2 constraints improve calibration; H3 constraints improve cross-population generalization; H4 CKINN explanations are more clinically consistent; H5 performance is more consistent across groups.
#
# **Locked claims (protocol sections 1 and 11).** C1 discrimination, non-inferiority to the best baseline (margin 0.01 in C-index, tier T2); C2 better transport across cohorts; C3 better calibration under transport; C4 fewer violations of the clinical priors than the MLP. A claim that fails its rule is reported as not supported.
#
# **Status of every later cell.** L: locked protocol. D: logged deviation (D-1 to D-3). A: exploratory item of amendment A1, which runs only after it is approved in cell 3 and can never enter a C1 to C4 verdict. G: gated on data or clinical raters that do not exist yet; the cell runs a readiness check and stops.
#
# **Not claimable from these data.** Generalization beyond two East Asian health-screening cohorts; causal effects; a physiological differential-equation model; a full multistate progression model; fairness by ethnicity or socio-economic group.
#
# **Session rule.** In every new session run cells 2, 3 and 4 first. They take seconds once the raw cache exists.

# %%
# CELL 2 - environment, seeds, folders, protocol constants, audit helpers        [status L; protocol section 12]
# Why: every later cell takes its settings and its bookkeeping from here. Nothing is read from the workbooks yet.
import os
os.environ.setdefault("CKINN_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ["CKINN_THREADS"]          # takes full effect in a fresh kernel

import sys, json, hashlib, platform, random, subprocess, datetime, importlib
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)


# ---------------------------------------------------------------- project root and folders
def find_root():
    """The folder that holds protocol/protocol_v1.0_LOCKED.md (the repository root)."""
    start = Path(os.environ.get("CKINN_ROOT", Path.cwd())).resolve()
    for p in [start, *start.parents]:
        if (p / "protocol" / "protocol_v1.0_LOCKED.md").exists():
            return p
    raise FileNotFoundError("Repository root not found. Open this notebook from inside the CKINN repository "
                            "(it must contain protocol/protocol_v1.0_LOCKED.md) or set CKINN_ROOT.")


ROOT = find_root()
DIR = {k: ROOT / v for k, v in {
    "raw": "data/raw", "processed": "data/processed", "cache": "data/processed/raw_cache",
    "results": "results", "parts": "results/parts", "preds": "results/preds", "audit": "results/audit",
    "gates": "results/gates", "figs": "results/figs", "tables": "results/tables", "protocol": "protocol"}.items()}
for _k, _p in DIR.items():
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def git_commit():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def cell_source_hash():
    """SHA-256 of the source of the cell being run (None outside IPython)."""
    try:
        return sha256_text(get_ipython().user_ns["In"][-1])
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path, datetime.date, datetime.datetime)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_jsonable, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- protocol constants (checked against the locked text in cell 3)
PROTOCOL_FILE = DIR["protocol"] / "protocol_v1.0_LOCKED.md"
PROTOCOL_SHA = sha256_file(PROTOCOL_FILE)
CFG = read_json(ROOT / "configs" / "protocol_v1.json")
CFG_SHA = sha256_file(ROOT / "configs" / "protocol_v1.json")

SEED = int(CFG["seed"])                       # section 12
H = float(CFG["horizon"])                     # section 6: fixed 3-year horizon
BASE = ["age", "sex_male", "bmi", "sbp", "dbp"]
TIERS = {"T1": BASE,                          # section 4
         "T2": BASE + ["tc", "tg", "alt"],
         "T3": BASE + ["tc", "tg", "alt", "fpg"],
         "T4": BASE + ["tc", "tg", "alt", "fpg", "hdl", "ast"]}
HEADLINE_TIER = "T2"                          # section 11
PRIORS = {"age": +1, "bmi": +1, "sbp": +1, "tg": +1, "fpg": +1, "hdl": -1,      # section 7: +1 risk up, -1 risk down
          "famhx": +1, "hba1c": +1, "exercise": -1}
EXTRAS = CFG["extras"]                        # section 4: cohort-specific, within cohort only
SMALL_SITES = (0, 16)                         # section 8: pooled into training, never tested
NO_TEST_SITES = (14,)                         # deviation D-1: follow-up shorter than 3 years
AGE_BANDS = [(0, 40, "under 40"), (40, 55, "40 to 54"), (55, 200, "55 and over")]      # section 9
HP = {                                        # section 6; no other hyperparameter is tuned
    "cox_ridge": 0.01,
    "xgb_trees": int(CFG["models"]["xgb_n"]), "xgb_depth": 3, "xgb_eta": 0.05,          # depth and rate: fixed code defaults
    "hidden": (64, 64), "dropout": 0.1, "lr": 2e-3, "weight_decay": 1e-4, "batch": 8192,
    "epochs": int(CFG["models"]["epochs"]), "patience": 4, "val_frac": 0.10,
    "lambda_grid": (0.0, *[float(x) for x in CFG["models"]["ckinn_grid"]]), "lambda_tol": float(CFG["models"]["tol"]),
    "lambda_fallback": 0.1, "inner_frac": 0.20}
EVAL = {                                      # sections 8 and 9
    "cv_folds": int(CFG["cv"]["folds"]), "cv_repeats": int(CFG["cv"]["repeats"]),
    "boot_n": int(CFG["cross"]["boot"]["n"]), "boot_max_n": int(CFG["cross"]["boot"]["max_n"]),
    "recal_n": int(CFG["cross"]["recal_n"]), "viol_delta_sd": 0.25, "viol_n": 5000,
    "net_benefit_thresholds": [round(0.01 * k, 2) for k in range(1, 11)], "c1_margin": 0.01}

# ---------------------------------------------------------------- seeds
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- packages: (import name, first stage that needs it, required or amendment-only)
PACKAGES = [("numpy", "A", "required"), ("pandas", "A", "required"), ("openpyxl", "A", "required"),
            ("matplotlib", "C", "required"), ("scipy", "D", "required"), ("sklearn", "D", "required"),
            ("lifelines", "D", "cross-check"), ("xgboost", "E", "required"), ("torch", "F", "required"),
            ("sksurv", "E", "amendment A1.2"), ("lightgbm", "E", "amendment A1.2"), ("shap", "I", "amendment A1.8")]


def package_table():
    rows = []
    for name, stage, need in PACKAGES:
        try:
            ver = getattr(importlib.import_module(name), "__version__", "installed")
        except Exception:
            ver = None
        rows.append({"package": name, "version": ver or "NOT INSTALLED", "first needed in stage": stage, "need": need})
    return pd.DataFrame(rows)


PKG = package_table()
DEVICE = "cpu"                                 # all fits run on the CPU with one thread so that results are repeatable
_accel = "torch not installed"
if "torch" in sys.modules or (PKG.set_index("package").loc["torch", "version"] != "NOT INSTALLED"):
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ["CKINN_THREADS"]))
    _accel = ("cuda available" if torch.cuda.is_available() else
              "mps available" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "none")

ENV = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
       "threads": int(os.environ["CKINN_THREADS"]), "device_used": DEVICE, "accelerator_seen": _accel,
       "packages": dict(zip(PKG["package"], PKG["version"]))}


# ---------------------------------------------------------------- audit trail: every code cell opens an Audit and closes it
class Audit:
    """Collects the checks of one cell, prints PASS or FAIL for each, writes results/audit/<cell>.json, appends one
    line to results/manifest.jsonl, and raises if any check failed. A stored record is evidence only together
    with its hashes: later cells and gates compare them with the current files."""

    def __init__(self, cell, title, status="L"):
        self.cell = f"cell_{int(cell):02d}" if isinstance(cell, int) else str(cell)
        self.title, self.status, self.rows, self.t0 = title, status, [], datetime.datetime.now()
        print(f"[{self.cell}] {title}   (status {status})\n")

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append({"check": name, "ok": ok, "detail": str(detail)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail != "" else ""))
        return ok

    def close(self, outputs=None, extra_path=None):
        rec = {"unit": f"nb_{self.cell}", "title": self.title, "status": self.status, "time": now(),
               "seconds": round((datetime.datetime.now() - self.t0).total_seconds(), 1),
               "passed": all(r["ok"] for r in self.rows), "checks": self.rows, "outputs": outputs or {},
               "source_sha": cell_source_hash(), "commit": git_commit(), "protocol_sha": PROTOCOL_SHA,
               "config_sha": CFG_SHA, "data_sha": globals().get("DATA_SHA", {}), "env": ENV}
        write_json(DIR["audit"] / f"{self.cell}.json", rec)
        if extra_path is not None:
            write_json(extra_path, rec)
        with open(DIR["results"] / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: rec[k] for k in ("unit", "time", "passed", "commit", "protocol_sha", "config_sha",
                                                    "source_sha")}, default=_jsonable) + "\n")
        n_fail = sum(not r["ok"] for r in self.rows)
        print(f"\n[{self.cell}] {'ALL CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED'}"
              f" ({len(self.rows)} checks, {rec['seconds']} s). Record: results/audit/{self.cell}.json")
        if n_fail:
            raise AssertionError(f"{self.cell}: {n_fail} check(s) failed. Fix this cell before running the next one.")
        return rec


def load_record(cell):
    """Record of an earlier cell, or None. Valid only if it passed and was made under the current protocol and config."""
    p = DIR["audit"] / (f"cell_{int(cell):02d}.json" if isinstance(cell, int) else f"{cell}.json")
    if not p.exists():
        return None
    r = read_json(p)
    r["valid"] = bool(r.get("passed")) and r.get("protocol_sha") == PROTOCOL_SHA and r.get("config_sha") == CFG_SHA
    return r


def require_gate(letter):
    """Stop unless the named gate passed under the current protocol and configuration."""
    p = DIR["gates"] / f"gate_{letter}.json"
    if not p.exists():
        raise RuntimeError(f"Gate {letter} has not been run. Run its gate cell first.")
    g = read_json(p)
    if not (g.get("passed") and g.get("protocol_sha") == PROTOCOL_SHA and g.get("config_sha") == CFG_SHA):
        raise RuntimeError(f"Gate {letter} is not valid for the current protocol or configuration. Re-run stage {letter}.")
    return g


# ---------------------------------------------------------------- report and checks
A = Audit(2, "Environment, seeds, folders, protocol constants")
print(f"  repository root : {ROOT}")
print(f"  python          : {ENV['python']} on {ENV['platform']}")
print(f"  threads / device: {ENV['threads']} / {DEVICE} (accelerator seen: {_accel})")
print(f"  seed            : {SEED}     horizon: {H} years     headline tier: {HEADLINE_TIER}")
print(f"  protocol sha256 : {PROTOCOL_SHA}")
print(f"  config sha256   : {CFG_SHA}\n")
print(PKG.to_string(index=False), "\n")

_have = dict(zip(PKG["package"], PKG["version"]))
A.check("Python is 3.9 or newer", sys.version_info >= (3, 9), ENV["python"])
A.check("packages needed in stages A and B are installed (numpy, pandas, openpyxl)",
        all(_have[p] != "NOT INSTALLED" for p in ("numpy", "pandas", "openpyxl")))
A.check("one computation thread is set", ENV["threads"] == 1, f"CKINN_THREADS={ENV['threads']}")
A.check("output folders exist and are writable", all(os.access(p, os.W_OK) for p in DIR.values()))
A.check("tiers are nested T1 < T2 < T3 < T4", all(set(TIERS[a]) < set(TIERS[b]) for a, b in (("T1", "T2"), ("T2", "T3"), ("T3", "T4"))),
        {k: len(v) for k, v in TIERS.items()})
A.check("headline tier T2 contains no glucose", "fpg" not in TIERS[HEADLINE_TIER] and "hba1c" not in TIERS[HEADLINE_TIER])
A.check("every prior has direction +1 or -1", all(v in (+1, -1) for v in PRIORS.values()), PRIORS)
_missing_later = [f"{r.package} (stage {r['first needed in stage']}, {r.need})" for _, r in PKG.iterrows() if r.version == "NOT INSTALLED"]
if _missing_later:
    print("\n  Not installed yet, needed later: " + "; ".join(_missing_later))
_ = A.close(outputs={"root": str(ROOT), "missing_later": _missing_later, "tiers": TIERS, "priors": PRIORS, "hp": HP, "eval": EVAL})

# %%
# CELL 3 - protocol lock, deviations register, amendment approvals               [status L; protocol sections 6 to 12]
# Why: proves the locked plan has not changed, that the settings used here are the locked ones, and records which
# exploratory items of amendment A1 the investigator has approved. Cells marked A refuse to run without approval.

# ---- EDIT ONLY THIS BLOCK: set an item to True when you (and your supervisor, where required) approve it. ----------
A1_APPROVED = {
    "A1.1": False,   # additional metrics (AUPRC, calibration intercept, threshold metrics, violation magnitude)
    "A1.2": False,   # additional comparators (survival forest; LR, RF, XGBoost, LightGBM, SVM classifiers)
    "A1.3": False,   # H1 (CKINN better than MLP) reported separately from C1
    "A1.4": False,   # prior ablations (one prior added, one removed, lambda path)
    "A1.5": False,   # extended extras E4 (LDL, alcohol)
    "A1.6": False,   # event-count audit before the locked recalibration
    "A1.7": False,   # two-year landmark and eligibility-aligned sensitivity
    "A1.8": False,   # explainability analyses
    "A1.9": False,   # clinical expert review (gated on real ratings)
    "A1.10": False,  # subgroup audit
    "A1.11": False,  # fairness-aware CKINN
    "A1.12": False,  # population alignment
    "A1.13": False,  # knowledge-graph CKINN
    "A1.14": False,  # glycaemic strata and one-step progression
    "A1.15": False,  # gated branches (readiness checks only)
}
# ---------------------------------------------------------------------------------------------------------------------
import re

A = Audit(3, "Protocol lock, deviations, amendment approvals")

# 1. lock
lock_expected = (DIR["protocol"] / "LOCK.sha256").read_text().split()[0]
A.check("locked protocol matches protocol/LOCK.sha256", PROTOCOL_SHA == lock_expected, PROTOCOL_SHA[:16] + "...")
PROTOCOL_TEXT = PROTOCOL_FILE.read_text(encoding="utf-8")

# 2. every active setting must be stated in the hashed protocol text
LOCKED_PHRASES = [
    ("horizon 3 years", H == 3.0, "Risk at horizon h = 3 years"),
    ("tier T1", TIERS["T1"] == BASE, "| T1 | age, sex, BMI, SBP, DBP |"),
    ("tier T2", TIERS["T2"] == BASE + ["tc", "tg", "alt"], "| T2 | T1 + total cholesterol, triglycerides, ALT |"),
    ("tier T3", TIERS["T3"] == TIERS["T2"] + ["fpg"], "| T3 | T2 + FPG |"),
    ("tier T4", TIERS["T4"] == TIERS["T3"] + ["hdl", "ast"], "| T4 | T3 + HDL, AST (RICH: heavy missingness) |"),
    ("headline tier T2", HEADLINE_TIER == "T2", "Pre-specified decision rules (headline tier T2)"),
    ("Cox ridge 0.01", HP["cox_ridge"] == 0.01, "Cox regression, ridge penalty 0.01"),
    ("hidden layers (64, 64)", HP["hidden"] == (64, 64), "hidden layers (64, 64)"),
    ("AdamW learning rate 2e-3", HP["lr"] == 2e-3, "AdamW, learning rate 2e-3"),
    ("weight decay 1e-4", HP["weight_decay"] == 1e-4, "weight decay 1e-4"),
    ("dropout 0.1", HP["dropout"] == 0.1, "dropout 0.1"),
    ("batch 8192", HP["batch"] == 8192, "batch 8192"),
    ("up to 40 epochs", HP["epochs"] == 40, "up to 40 epochs"),
    ("early-stopping patience 4", HP["patience"] == 4, "early stopping (patience 4)"),
    ("10% validation split", HP["val_frac"] == 0.10, "on a 10% validation split"),
    ("lambda grid {0, 0.1, 1, 10}", HP["lambda_grid"] == (0.0, 0.1, 1.0, 10.0), "fit lambda in {0, 0.1, 1, 10}"),
    ("lambda tolerance 0.002", HP["lambda_tol"] == 0.002, "within 0.002 of the best value"),
    ("lambda fallback 0.1", HP["lambda_fallback"] == 0.1, "if none qualifies, lambda = 0.1"),
    ("inner 80/20 split", HP["inner_frac"] == 0.20, "inner 80/20 split, fixed seed"),
    ("5-fold CV, 3 repeats", (EVAL["cv_folds"], EVAL["cv_repeats"]) == (5, 3), "repeated stratified 5-fold CV in RICH, 3 repeats"),
    ("sites 0 and 16 never tested", SMALL_SITES == (0, 16), "sites 0 and 16 are pooled into training and never tested"),
    ("recalibration on 1,000", EVAL["recal_n"] == 1000, "recalibration on a random 1,000 target participants"),
    ("bootstrap 200 resamples", EVAL["boot_n"] == 200, "paired bootstrap, 200 resamples"),
    ("bootstrap subsample 40,000", EVAL["boot_max_n"] == 40000, "at most 40,000 target participants"),
    ("violation check +0.25 SD on 5,000", (EVAL["viol_delta_sd"], EVAL["viol_n"]) == (0.25, 5000), "finite differences, +0.25 SD, 5,000 test patients"),
    ("net benefit at 1% to 10%", EVAL["net_benefit_thresholds"] == [round(0.01 * k, 2) for k in range(1, 11)], "net benefit at 1% to 10% thresholds"),
    ("age bands", [b[2] for b in AGE_BANDS] == ["under 40", "40 to 54", "55 and over"], "age band (under 40, 40 to 54, 55 and over)"),
    ("C1 margin 0.01", EVAL["c1_margin"] == 0.01, "exceeds -0.01"),
    ("seed rule", SEED == 0, "CV seed = repeat index; model seed = 1000 x repeat + fold (other analyses: seed 0)"),
]
for name, value_ok, phrase in LOCKED_PHRASES:
    A.check(f"setting '{name}' equals the locked text", value_ok and (phrase in PROTOCOL_TEXT),
            "" if phrase in PROTOCOL_TEXT else "phrase not found in the locked protocol")
_prior_rows = {"BMI": "bmi", "FPG": "fpg", "Triglycerides": "tg", "SBP": "sbp", "Age": "age", "HDL": "hdl",
               "Family history": "famhx", "HbA1c": "hba1c", "Exercise": "exercise"}
_prior_ok = all(re.search(rf"\| {re.escape(lab)} \| {'up' if PRIORS[v] > 0 else 'down'}", PROTOCOL_TEXT) for lab, v in _prior_rows.items())
A.check("all nine prior directions equal protocol section 7", _prior_ok and len(PRIORS) == 9)

# 3. deviations
DEV_TEXT = (DIR["protocol"] / "deviations.md").read_text(encoding="utf-8")
DEVIATIONS = re.findall(r"^## (D-\d+) \((.*?)\)\s*\n(.*?)(?=\n## |\Z)", DEV_TEXT, flags=re.S | re.M)
print("\n  Deviations on file:")
for d_id, d_when, d_body in DEVIATIONS:
    print(f"   {d_id}  ({d_when})  {' '.join(d_body.split())[:118]}...")
A.check("deviations D-1 to D-5 are on file", [d[0] for d in DEVIATIONS] == [f"D-{i}" for i in range(1, 6)], [d[0] for d in DEVIATIONS])

# 4. amendment A1: items and approvals
AMEND_FILE = DIR["protocol"] / "amendment_A1_PROPOSED.md"
AMEND_SHA = sha256_file(AMEND_FILE)
A1_ITEMS = dict(re.findall(r"^\*\*(A1\.\d+) (.*?)\.\*\*", AMEND_FILE.read_text(encoding="utf-8"), flags=re.M))
A.check("amendment A1 lists items A1.1 to A1.15", list(A1_ITEMS) == [f"A1.{i}" for i in range(1, 16)], len(A1_ITEMS))
A.check("the approval block names exactly the items of the amendment", set(A1_APPROVED) == set(A1_ITEMS))
A.check("approval values are True or False", all(isinstance(v, bool) for v in A1_APPROVED.values()))

APPROVAL_FILE = DIR["protocol"] / "amendment_A1_approvals.json"
_old = read_json(APPROVAL_FILE) if APPROVAL_FILE.exists() else {"items": {}}
APPROVALS = {"amendment_sha": AMEND_SHA, "updated": now(), "items": {}}
for item, title in A1_ITEMS.items():
    prev = _old["items"].get(item, {})
    ok = bool(A1_APPROVED.get(item, False))
    same_text = prev.get("amendment_sha") == AMEND_SHA
    APPROVALS["items"][item] = {
        "title": title, "approved": ok,
        "approved_on": (prev.get("approved_on") if (ok and prev.get("approved") and same_text) else (now() if ok else None)),
        "amendment_sha": AMEND_SHA if ok else None}
write_json(APPROVAL_FILE, APPROVALS)


def require_approval(item):
    """Called by every cell marked A. Stops unless the item is approved for the current text of the amendment."""
    rec = read_json(APPROVAL_FILE)["items"].get(item)
    if not rec or not rec["approved"] or rec["amendment_sha"] != sha256_file(AMEND_FILE):
        raise PermissionError(f"{item} is not approved. It is exploratory (amendment A1). Set it to True in cell 3, "
                              f"re-run cell 3, then re-run this cell.")
    return True


print("\n  Amendment A1 (exploratory; none of these can enter a C1 to C4 verdict):")
print(pd.DataFrame([{"item": k, "approved": v["approved"], "approved on": v["approved_on"] or "", "title": v["title"]}
                    for k, v in APPROVALS["items"].items()]).to_string(index=False))
_ = A.close(outputs={"lock_sha": lock_expected, "deviations": [d[0] for d in DEVIATIONS], "amendment_sha": AMEND_SHA,
                     "approved_items": [k for k, v in A1_APPROVED.items() if v]})

# %%
# CELL 4 - raw workbooks: identity by SHA-256, one read, structural assertions, validated cache     [status L; protocol section 2]
# Why: proves the analysis starts from the two published files, unchanged. The workbooks are opened read-only and are
# never written. The first run reads the Excel files (RICH takes one to three minutes); later runs load a cache that
# is accepted only if its recorded source hash and its content hash both still match.
import time

A = Audit(4, "Raw workbooks: hashes, shapes, headers, cache")

RAW_SPEC = {   # expected structure of the published files; header_sha is the SHA-256 of the header names joined by newlines
    "NAGALA": {"file": "DatasetIntJObesity.xlsx", "sheets": ["Sheet 1"], "shape": (15464, 31),
               "header_sha": "db9dbc8f949105a4578326f421d54262641c5ff6645241fc69c36c22a980962d", "first": "Patient Number", "last": "Incident DM 0/1"},
    "RICH": {"file": "RCHealthCareData-20180820.xlsx", "sheets": ["RC"], "shape": (211833, 25),
             "header_sha": "88782dd81f55586b5753d6f87b2a8dc00b106561f48c5dcb3c8ec28341849bcc", "first": "id", "last": "family histroy of diabetes(1,Yes;0,No)"},
}
CHECKSUMS = {line.split()[1]: line.split()[0]
             for line in (DIR["raw"] / "CHECKSUMS.sha256").read_text().splitlines() if line.strip()}


def excel_letter(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def content_sha(df):
    """Order-sensitive hash of every value, the index and the column names of a data frame."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update("\n".join(map(str, df.columns)).encode("utf-8"))
    return h.hexdigest()


def locate_by_hash(want_sha, preferred_name):
    """Find the workbook by content, whatever it is called: data/raw first, then the notebook folder and the root."""
    seen = []
    for folder in (DIR["raw"], Path.cwd(), ROOT, ROOT / "notebooks"):
        for p in sorted(folder.glob("*.xlsx"), key=lambda q: q.name != preferred_name):
            if p.resolve() in seen:
                continue
            seen.append(p.resolve())
            if sha256_file(p) == want_sha:
                return p
    return None


RAW, DATA_SHA, RAW_PATH, _inventory = {}, {}, {}, []
for cohort, spec in RAW_SPEC.items():
    want = CHECKSUMS[spec["file"]]
    path = locate_by_hash(want, spec["file"])
    if not A.check(f"{cohort}: workbook with the published SHA-256 was found", path is not None,
                   path.name if path else f"put {spec['file']} in data/raw"):
        continue
    RAW_PATH[cohort], DATA_SHA[cohort] = path, want
    cache, meta_p = DIR["cache"] / f"{cohort.lower()}_raw.pkl", DIR["cache"] / f"{cohort.lower()}_raw.json"
    df, source = None, "cache"
    if cache.exists() and meta_p.exists():
        meta = read_json(meta_p)
        if meta.get("source_sha") == want:
            cand = pd.read_pickle(cache)
            if content_sha(cand) == meta.get("content_sha") and list(cand.shape) == meta.get("shape"):
                df = cand
    if df is None:
        t0 = time.time()
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheets = list(xl.sheet_names)
        df = xl.parse(sheets[0])
        xl.close()
        source = f"Excel, read in {time.time() - t0:.0f} s"
        df.to_pickle(cache)
        write_json(meta_p, {"cohort": cohort, "source_file": path.name, "source_sha": want, "sheets": sheets,
                            "shape": list(df.shape), "header_sha": sha256_text("\n".join(map(str, df.columns))),
                            "content_sha": content_sha(df), "pandas": pd.__version__, "written": now()})
    meta = read_json(meta_p)
    RAW[cohort] = df
    print(f"\n  {cohort}: {path.name}  ({path.stat().st_size / 1e6:.1f} MB; loaded from {source})")
    A.check(f"{cohort}: the raw file still has the published SHA-256 after reading", sha256_file(path) == want)
    A.check(f"{cohort}: single worksheet named {spec['sheets']}", meta["sheets"] == spec["sheets"], meta["sheets"])
    A.check(f"{cohort}: {spec['shape'][0]:,} rows by {spec['shape'][1]} columns", df.shape == spec["shape"], df.shape)
    A.check(f"{cohort}: header names are exactly those of the published file",
            sha256_text("\n".join(map(str, df.columns))) == spec["header_sha"])
    A.check(f"{cohort}: first and last columns", (df.columns[0], df.columns[-1]) == (spec["first"], spec["last"]))
    A.check(f"{cohort}: no column is entirely empty and no row is entirely empty",
            (df.notna().sum() > 0).all() and (df.notna().sum(axis=1) > 0).all())
    A.check(f"{cohort}: every column is numeric", all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes),
            df.dtypes.astype(str).value_counts().to_dict())
    for i, c in enumerate(df.columns):
        _inventory.append({"cohort": cohort, "excel_column": excel_letter(i), "header": c, "dtype": str(df[c].dtype),
                           "non_missing": int(df[c].notna().sum()), "missing": int(df[c].isna().sum())})

INVENTORY = pd.DataFrame(_inventory)
INVENTORY.to_csv(DIR["audit"] / "raw_inventory.csv", index=False, encoding="utf-8")
print("\n", INVENTORY.to_string(index=False))


def get_raw(cohort):
    """A copy of the unchanged raw table. Later cells never modify RAW itself."""
    return RAW[cohort].copy()


_ = A.close(outputs={"files": {k: v.name for k, v in RAW_PATH.items()}, "data_sha": DATA_SHA,
                     "shapes": {k: list(v.shape) for k, v in RAW.items()},
                     "content_sha": {k: content_sha(v) for k, v in RAW.items()},
                     "inventory": "results/audit/raw_inventory.csv"})

# %%
# CELL 5 - proposal coverage register: every requirement of the 26-page proposal has a cell and a status   [status L]
# Why: makes it impossible for a proposal requirement to drop out unnoticed. The final audit (cell 86) re-reads this
# register and demands, for every row, either a result file or a recorded blocking reason.
A = Audit(5, "Proposal coverage register")

REG_FILE = DIR["protocol"] / "proposal_coverage.csv"
REG = pd.read_csv(REG_FILE, dtype=str, keep_default_na=False)
REG_SHA = sha256_file(REG_FILE)
N_CELLS, N_PAGES, N_SECTIONS = 88, 26, 28
VERDICTS = {"existing data; locked", "existing data; amendment", "existing data; locked and amendment",
            "needs data", "needs expertise", "not identifiable", "writing"}


def _ints(s):
    return [int(x) for x in s.split(";") if x.strip()]


def _pages(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _sections(s):
    return [int(re.sub(r"[^0-9]", "", x)) for x in s.split(";") if re.sub(r"[^0-9]", "", x)]


A.check("register has the expected columns", list(REG.columns) == ["id", "page", "section", "requirement", "verdict", "cells", "status", "note"])
A.check("86 requirements, identifiers unique", len(REG) == 86 and REG["id"].is_unique, len(REG))
A.check("no requirement lacks a description, a verdict, a cell or a status",
        not (REG[["requirement", "verdict", "cells", "status"]] == "").any().any())
A.check("every verdict is one of the seven allowed", set(REG["verdict"]) <= VERDICTS, sorted(set(REG["verdict"]) - VERDICTS))
A.check("every status is L, D, A or G", all(set(s.split(";")) <= {"L", "D", "A", "G"} for s in REG["status"]))
A.check(f"every cell reference lies in 1 to {N_CELLS}", all(1 <= c <= N_CELLS for s in REG["cells"] for c in _ints(s)))
_pg = sorted({p for s in REG["page"] for p in _pages(s)})
A.check(f"every proposal page 1 to {N_PAGES} is covered", _pg == list(range(1, N_PAGES + 1)),
        f"missing {sorted(set(range(1, N_PAGES + 1)) - set(_pg))}")
_sec = sorted({x for s in REG["section"] for x in _sections(s)})
A.check(f"every proposal section 1 to {N_SECTIONS} is covered", _sec == list(range(1, N_SECTIONS + 1)),
        f"missing {sorted(set(range(1, N_SECTIONS + 1)) - set(_sec))}")
A.check("the research questions and hypotheses are in the register",
        REG["requirement"].str.contains("RQ1 to RQ7").any() and REG["requirement"].str.contains("H1 to H5").any())
_blocked = REG[REG["verdict"].isin(["needs data", "needs expertise", "not identifiable"])]
A.check("every blocked requirement is gated (status G) and states its reason", (_blocked["status"] == "G").all() and (_blocked["note"] != "").all())
_amend = REG[REG["verdict"].str.contains("amendment")]
A.check("every requirement that relies on the amendment carries status A", _amend["status"].str.contains("A").all())
A.check("no locked-only requirement carries status A or G",
        not REG.loc[REG["verdict"] == "existing data; locked", "status"].str.contains("A|G").any())

print("\n  Requirements by verdict:")
print(REG["verdict"].value_counts().rename_axis("verdict").reset_index(name="requirements").to_string(index=False))
print("\n  Requirements that cannot be completed with RICH and NAGALA alone:")
print(_blocked[["id", "page", "requirement", "note"]].to_string(index=False))
_used = sorted({c for s in REG["cells"] for c in _ints(s)})
print(f"\n  Cells that serve at least one proposal requirement: {len(_used)} of {N_CELLS}. "
      f"The others serve the protocol, the tests and the gates: {sorted(set(range(1, N_CELLS + 1)) - set(_used))}")
REG["verdict"].value_counts().rename_axis("verdict").reset_index(name="requirements").to_csv(
    DIR["audit"] / "coverage_summary.csv", index=False)
_ = A.close(outputs={"register_sha": REG_SHA, "rows": len(REG), "by_verdict": REG["verdict"].value_counts().to_dict(),
                     "blocked_ids": _blocked["id"].tolist()})

# %%
# CELL 6 - GATE A: governance and source authority                                              [status L]
# Why: a stop point that re-verifies stage A from the files themselves, not from memory. Stage B starts only if this
# cell ends with ALL CHECKS PASSED. A record on disk counts only when its hashes equal the current files.
A = Audit("gate_A", "Gate A: governance and source authority")

_recs = {c: load_record(c) for c in (2, 3, 4, 5)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current protocol and configuration", bool(r) and r["valid"],
            "missing" if not r else r["time"])
A.check("locked protocol still matches protocol/LOCK.sha256",
        sha256_file(PROTOCOL_FILE) == (DIR["protocol"] / "LOCK.sha256").read_text().split()[0])
A.check("deviations file unchanged since cell 3 and lists D-1 to D-5",
        re.findall(r"^## (D-\d+)", (DIR["protocol"] / "deviations.md").read_text(encoding="utf-8"), flags=re.M) == _recs[3]["outputs"]["deviations"]
        if _recs[3] else False)
A.check("amendment A1 unchanged since cell 3", bool(_recs[3]) and sha256_file(AMEND_FILE) == _recs[3]["outputs"]["amendment_sha"])
A.check("approval record exists and belongs to the current amendment text",
        APPROVAL_FILE.exists() and read_json(APPROVAL_FILE)["amendment_sha"] == sha256_file(AMEND_FILE))
for cohort in RAW_SPEC:
    ok_file = cohort in RAW_PATH and sha256_file(RAW_PATH[cohort]) == CHECKSUMS[RAW_SPEC[cohort]["file"]]
    A.check(f"{cohort}: raw workbook re-hashed and equal to data/raw/CHECKSUMS.sha256", ok_file)
    ok_mem = cohort in RAW and bool(_recs[4]) and content_sha(RAW[cohort]) == _recs[4]["outputs"]["content_sha"].get(cohort)
    A.check(f"{cohort}: table in memory equals the one audited in cell 4", ok_mem)
    ok_cache = (DIR["cache"] / f"{cohort.lower()}_raw.json").exists() and \
        read_json(DIR["cache"] / f"{cohort.lower()}_raw.json")["content_sha"] == (_recs[4]["outputs"]["content_sha"].get(cohort) if _recs[4] else None)
    A.check(f"{cohort}: cache on disk equals the one audited in cell 4", ok_cache)
A.check("coverage register unchanged since cell 5", bool(_recs[5]) and sha256_file(REG_FILE) == _recs[5]["outputs"]["register_sha"])
try:
    _tracked = [f for f in subprocess.run(["git", "-C", str(ROOT), "ls-files", "data/raw", "data/processed", "results/preds"],
                                          capture_output=True, text=True, timeout=20).stdout.split()
                if not f.endswith("CHECKSUMS.sha256")]
except Exception:
    _tracked = []                                   # git not available on this machine: nothing can be tracked from here
A.check("no workbook, processed data or prediction file is tracked by git", not _tracked, _tracked)

_approved = [k for k, v in read_json(APPROVAL_FILE)["items"].items() if v["approved"]]
print(f"\n  Amendment items approved so far: {_approved if _approved else 'none (only L and D cells can run)'}")
print("  Packages still to install before later stages: " + ("; ".join(_recs[2]["outputs"]["missing_later"]) or "none"))
_ = A.close(outputs={"stage": "A", "approved_items": _approved, "data_sha": DATA_SHA},
            extra_path=DIR["gates"] / "gate_A.json")
print("\n  GATE A PASSED. Stage B (cells 7 to 14, raw-data forensics) may start.")
