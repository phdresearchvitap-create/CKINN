# %% [markdown]
# # CKINN: clinical knowledge-informed neural networks for explainable, fair and cross-population prediction of type 2 diabetes risk
#
# **Cell 1. Scope of the study (status L).** This notebook is self-contained: it needs only the two Excel workbooks (RICH and NAGALA) in the same folder. It follows the 88-cell master plan and the locked analysis protocol v1.0; nothing in this notebook may change a locked setting.
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
# **Session rule.** Run the cells in order the first time. After that, if the kernel restarts (for example when a tablet app is closed or put in the background), simply run the next cell: its first three lines restore the earlier set-up cells automatically.

# %%
# CELL 2 - environment, seeds, folders, protocol constants, audit helpers        [status L; protocol section 12]
# Why: every later cell takes its settings and its bookkeeping from here. Nothing is read from the workbooks yet.
import os
os.environ.setdefault("CKINN_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ["CKINN_THREADS"]          # takes full effect in a fresh kernel

import sys, io, re, json, time, hashlib, platform, random, datetime, importlib
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)

# ---------------------------------------------------------------- folders: everything is written under ckinn_results/
WORK = Path.cwd()                              # the folder that holds this notebook and the two workbooks
OUT = WORK / "ckinn_results"
DIR = {k: OUT / k for k in ("cache", "data", "audit", "gates", "parts", "preds", "figs", "tables")}
for _p in DIR.values():
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


def cell_source_hash():
    """SHA-256 of the source of the cell being run (None outside Jupyter)."""
    try:
        return sha256_text(get_ipython().user_ns["In"][-1])
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, datetime.date, datetime.datetime)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_jsonable, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- session restore (for kernels that restart, as on a tablet)
# Cells that define things later cells need (2, 3, 4, 5, ...) save their own source in ckinn_results/session/. Every later
# cell starts with three "session guard" lines; if the kernel was restarted, the guard re-runs the saved cells silently.
SESSION = OUT / "session"
SESSION.mkdir(parents=True, exist_ok=True)
_CKINN_LOADED = globals().setdefault("_CKINN_LOADED", set())
(SESSION / "guard.py").write_text('''
def _ckinn_restore(g, cell):
    """Re-run, in order, every saved set-up cell numbered below `cell` that is not loaded in this kernel."""
    import io, contextlib, pathlib
    done = g.setdefault("_CKINN_LOADED", set())
    for f in sorted((pathlib.Path.cwd() / "ckinn_results" / "session").glob("cell_*.py")):
        n = int(f.stem.split("_")[1])
        if n >= cell or n in done:
            continue
        g["_CKINN_DEPTH"] = g.get("_CKINN_DEPTH", 0) + 1
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(f.read_text(encoding="utf-8"), f.name, "exec"), g)
        except Exception:
            print(buf.getvalue()[-3000:])
            print(f"[set-up cell {n} could not be restored: run cell {n} by hand, then run this cell again]")
            raise
        finally:
            g["_CKINN_DEPTH"] -= 1
        print(f"[the kernel had restarted: set-up cell {n} was restored automatically]")
''', encoding="utf-8")


def restoring():
    """True while a saved cell is being re-run by the session guard (nothing is recorded then)."""
    return globals().get("_CKINN_DEPTH", 0) > 0


def keep_for_session(cell):
    """Call at the end of a cell that later cells depend on: marks it loaded and saves its source for the guard."""
    _CKINN_LOADED.add(cell)
    if restoring():
        return
    try:
        (SESSION / f"cell_{cell:02d}.py").write_text(get_ipython().user_ns["In"][-1], encoding="utf-8")
    except Exception:
        pass                                   # not running in Jupyter: nothing to save


# ---------------------------------------------------------------- protocol constants (locked protocol v1.0; frozen in cell 3)
PROTOCOL_SHA = "93d337162d6ca9c07f5d3f552276314e1ca0d993fb85bbc1a2a2d53180d6d58f"    # SHA-256 of the locked protocol document
SEED = 0                                      # section 12
H = 3.0                                       # section 6: fixed 3-year horizon
BASE = ["age", "sex_male", "bmi", "sbp", "dbp"]
TIERS = {"T1": BASE,                          # section 4
         "T2": BASE + ["tc", "tg", "alt"],
         "T3": BASE + ["tc", "tg", "alt", "fpg"],
         "T4": BASE + ["tc", "tg", "alt", "fpg", "hdl", "ast"]}
HEADLINE_TIER = "T2"                          # section 11
PRIORS = {"age": +1, "bmi": +1, "sbp": +1, "tg": +1, "fpg": +1, "hdl": -1,      # section 7: +1 risk up, -1 risk down
          "famhx": +1, "hba1c": +1, "exercise": -1}
EXTRAS = {"RICH": ["famhx", "smoke"],         # section 4: cohort-specific, used within one cohort only
          "NAGALA": ["waist", "exercise", "fatty_liver", "ggt"], "NAGALA_E3_extra": ["hba1c"]}
SMALL_SITES = (0, 16)                         # section 8: pooled into training, never tested
NO_TEST_SITES = (14,)                         # deviation D-1: follow-up shorter than 3 years
AGE_BANDS = [(0, 40, "under 40"), (40, 55, "40 to 54"), (55, 200, "55 and over")]      # section 9
HP = {                                        # section 6; no other hyperparameter is tuned
    "cox_ridge": 0.01,
    "xgb_trees": 300, "xgb_depth": 3, "xgb_eta": 0.05,
    "hidden": (64, 64), "dropout": 0.1, "lr": 2e-3, "weight_decay": 1e-4, "batch": 8192,
    "epochs": 40, "patience": 4, "val_frac": 0.10,
    "lambda_grid": (0.0, 0.1, 1.0, 10.0), "lambda_tol": 0.002, "lambda_fallback": 0.1, "inner_frac": 0.20}
EVAL = {                                      # sections 8 and 9
    "cv_folds": 5, "cv_repeats": 3, "boot_n": 200, "boot_max_n": 40000, "recal_n": 1000,
    "viol_delta_sd": 0.25, "viol_n": 5000,
    "net_benefit_thresholds": [round(0.01 * k, 2) for k in range(1, 11)], "c1_margin": 0.01}


def settings_fingerprint():
    """One SHA-256 over every analysis setting. If any setting is edited, this value changes and cell 3 fails."""
    s = {"seed": SEED, "horizon": H, "tiers": TIERS, "headline": HEADLINE_TIER, "priors": PRIORS, "extras": EXTRAS,
         "small_sites": SMALL_SITES, "no_test_sites": NO_TEST_SITES, "age_bands": AGE_BANDS, "hp": HP, "eval": EVAL}
    return sha256_text(json.dumps(s, sort_keys=True))


SETTINGS_SHA = settings_fingerprint()

# ---------------------------------------------------------------- seeds
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- packages: (import name, first stage that needs it, required or amendment-only)
PACKAGES = [("numpy", "A", "required"), ("pandas", "A", "required"),
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
if PKG.set_index("package").loc["torch", "version"] != "NOT INSTALLED":
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
    """Collects the checks of one cell, prints PASS or FAIL for each, writes ckinn_results/audit/<cell>.json, appends
    one line to ckinn_results/manifest.jsonl, and raises if any check failed. A stored record is evidence only
    together with its hashes: later cells and gates compare them with the current settings and files."""

    def __init__(self, cell, title, status="L"):
        self.cell = f"cell_{int(cell):02d}" if isinstance(cell, int) else str(cell)
        self.title, self.status, self.rows, self.t0 = title, status, [], datetime.datetime.now()
        print(f"[{self.cell}] {title}   (status {status})\n")

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append({"check": name, "ok": ok, "detail": str(detail)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if str(detail) != "" else ""))
        return ok

    def close(self, outputs=None, extra_path=None):
        rec = {"unit": f"nb_{self.cell}", "title": self.title, "status": self.status, "time": now(),
               "seconds": round((datetime.datetime.now() - self.t0).total_seconds(), 1),
               "passed": all(r["ok"] for r in self.rows), "checks": self.rows, "outputs": outputs or {},
               "source_sha": cell_source_hash(), "protocol_sha": PROTOCOL_SHA, "settings_sha": settings_fingerprint(),
               "data_sha": globals().get("DATA_SHA", {}), "env": ENV}
        n_fail = sum(not r["ok"] for r in self.rows)
        if restoring():                        # re-run by the session guard: verify again, record nothing
            if n_fail:
                raise AssertionError(f"{self.cell}: {n_fail} check(s) failed while the session was being restored.")
            return rec
        write_json(DIR["audit"] / f"{self.cell}.json", rec)
        if extra_path is not None:
            write_json(extra_path, rec)
        with open(OUT / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: rec[k] for k in ("unit", "time", "passed", "protocol_sha", "settings_sha", "source_sha")}) + "\n")
        print(f"\n[{self.cell}] {'ALL CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED'}"
              f" ({len(self.rows)} checks, {rec['seconds']} s). Record: ckinn_results/audit/{self.cell}.json")
        if n_fail:
            raise AssertionError(f"{self.cell}: {n_fail} check(s) failed. Fix this cell before running the next one.")
        return rec


def load_record(cell):
    """Record of an earlier cell, or None. Valid only if it passed and was made under the current settings."""
    p = DIR["audit"] / (f"cell_{int(cell):02d}.json" if isinstance(cell, int) else f"{cell}.json")
    if not p.exists():
        return None
    r = read_json(p)
    r["valid"] = bool(r.get("passed")) and r.get("settings_sha") == settings_fingerprint()
    return r


def require_gate(letter):
    """Stop unless the named gate passed under the current settings."""
    p = DIR["gates"] / f"gate_{letter}.json"
    if not p.exists():
        raise RuntimeError(f"Gate {letter} has not been run. Run its gate cell first.")
    g = read_json(p)
    if not (g.get("passed") and g.get("settings_sha") == settings_fingerprint()):
        raise RuntimeError(f"Gate {letter} is not valid for the current settings. Re-run stage {letter}.")
    return g


# ---------------------------------------------------------------- report and checks
A = Audit(2, "Environment, seeds, folders, protocol constants")
print(f"  working folder  : {WORK}")
print(f"  results folder  : {OUT}")
print(f"  python          : {ENV['python']} on {ENV['platform']}")
print(f"  threads / device: {ENV['threads']} / {DEVICE} (accelerator seen: {_accel})")
print(f"  seed            : {SEED}     horizon: {H} years     headline tier: {HEADLINE_TIER}")
print(f"  settings sha256 : {SETTINGS_SHA}\n")
print(PKG.to_string(index=False), "\n")

_have = dict(zip(PKG["package"], PKG["version"]))
A.check("Python is 3.9 or newer", sys.version_info >= (3, 9), ENV["python"])
A.check("packages needed in stages A and B are installed (numpy, pandas)",
        all(_have[p] != "NOT INSTALLED" for p in ("numpy", "pandas")))
A.check("one computation thread is set", ENV["threads"] == 1, f"CKINN_THREADS={ENV['threads']}")
A.check("output folders exist and are writable", all(os.access(p, os.W_OK) for p in DIR.values()))
A.check("tiers are nested T1 < T2 < T3 < T4",
        all(set(TIERS[a]) < set(TIERS[b]) for a, b in (("T1", "T2"), ("T2", "T3"), ("T3", "T4"))), {k: len(v) for k, v in TIERS.items()})
A.check("headline tier T2 contains no glucose", "fpg" not in TIERS[HEADLINE_TIER] and "hba1c" not in TIERS[HEADLINE_TIER])
A.check("every prior has direction +1 or -1", all(v in (+1, -1) for v in PRIORS.values()), PRIORS)
_missing_later = [f"{r['package']} (stage {r['first needed in stage']}, {r['need']})" for _, r in PKG.iterrows() if r["version"] == "NOT INSTALLED"]
if _missing_later:
    print("\n  Not installed yet, needed later: " + "; ".join(_missing_later))
_ = A.close(outputs={"work": str(WORK), "missing_later": _missing_later, "tiers": TIERS, "priors": PRIORS, "hp": HP, "eval": EVAL})
keep_for_session(2)

# %%
# CELL 3 - settings freeze, deviations register, amendment approvals              [status L; protocol sections 4 to 12]
# Why: proves that the settings in cell 2 are exactly the locked ones, lists the logged deviations, and records which
# exploratory items of amendment A1 the investigator has approved. Cells marked A refuse to run without approval.
# session guard (same three lines in every cell): restores cells 2 to 5 by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 3)

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

A = Audit(3, "Settings freeze, deviations, amendment approvals")

# 1. the whole settings block must equal the frozen fingerprint
LOCKED_SETTINGS_SHA = "66d76d3adcc9a1e88c740f61153565d552dfdb2ebe14dd2bb1b03a2e837e672e"
A.check("settings of cell 2 equal the frozen fingerprint (nothing was edited)", settings_fingerprint() == LOCKED_SETTINGS_SHA,
        settings_fingerprint()[:16] + "...")

# 2. each setting against the value written in the locked protocol (section; value; phrase in the protocol text)
LOCKED = [
    ("6", "horizon 3 years", H == 3.0, "Risk at horizon h = 3 years"),
    ("4", "tier T1 = age, sex, BMI, SBP, DBP", TIERS["T1"] == ["age", "sex_male", "bmi", "sbp", "dbp"], "| T1 | age, sex, BMI, SBP, DBP |"),
    ("4", "tier T2 = T1 + TC, TG, ALT", TIERS["T2"] == TIERS["T1"] + ["tc", "tg", "alt"], "| T2 | T1 + total cholesterol, triglycerides, ALT |"),
    ("4", "tier T3 = T2 + FPG", TIERS["T3"] == TIERS["T2"] + ["fpg"], "| T3 | T2 + FPG |"),
    ("4", "tier T4 = T3 + HDL, AST", TIERS["T4"] == TIERS["T3"] + ["hdl", "ast"], "| T4 | T3 + HDL, AST (RICH: heavy missingness) |"),
    ("11", "headline tier T2", HEADLINE_TIER == "T2", "Pre-specified decision rules (headline tier T2)"),
    ("6", "Cox ridge penalty 0.01", HP["cox_ridge"] == 0.01, "Cox regression, ridge penalty 0.01"),
    ("6", "hidden layers (64, 64)", HP["hidden"] == (64, 64), "hidden layers (64, 64)"),
    ("6", "AdamW learning rate 2e-3", HP["lr"] == 2e-3, "AdamW, learning rate 2e-3"),
    ("6", "weight decay 1e-4", HP["weight_decay"] == 1e-4, "weight decay 1e-4"),
    ("6", "dropout 0.1", HP["dropout"] == 0.1, "dropout 0.1"),
    ("6", "batch 8192", HP["batch"] == 8192, "batch 8192"),
    ("6", "up to 40 epochs", HP["epochs"] == 40, "up to 40 epochs"),
    ("6", "early-stopping patience 4", HP["patience"] == 4, "early stopping (patience 4)"),
    ("6", "10% validation split", HP["val_frac"] == 0.10, "on a 10% validation split"),
    ("6", "lambda grid {0, 0.1, 1, 10}", HP["lambda_grid"] == (0.0, 0.1, 1.0, 10.0), "fit lambda in {0, 0.1, 1, 10}"),
    ("6", "lambda tolerance 0.002", HP["lambda_tol"] == 0.002, "within 0.002 of the best value"),
    ("6", "lambda fallback 0.1", HP["lambda_fallback"] == 0.1, "if none qualifies, lambda = 0.1"),
    ("6", "inner 80/20 split", HP["inner_frac"] == 0.20, "inner 80/20 split, fixed seed"),
    ("8", "5-fold CV, 3 repeats", (EVAL["cv_folds"], EVAL["cv_repeats"]) == (5, 3), "repeated stratified 5-fold CV in RICH, 3 repeats"),
    ("8", "sites 0 and 16 never tested", SMALL_SITES == (0, 16), "sites 0 and 16 are pooled into training and never tested"),
    ("8", "recalibration on 1,000 target participants", EVAL["recal_n"] == 1000, "recalibration on a random 1,000 target participants"),
    ("9", "bootstrap 200 resamples", EVAL["boot_n"] == 200, "paired bootstrap, 200 resamples"),
    ("9", "bootstrap subsample at most 40,000", EVAL["boot_max_n"] == 40000, "at most 40,000 target participants"),
    ("9", "violation check +0.25 SD on 5,000", (EVAL["viol_delta_sd"], EVAL["viol_n"]) == (0.25, 5000), "finite differences, +0.25 SD, 5,000 test patients"),
    ("9", "net benefit at 1% to 10%", EVAL["net_benefit_thresholds"] == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], "net benefit at 1% to 10% thresholds"),
    ("9", "age bands under 40, 40 to 54, 55 and over", [(b[0], b[1]) for b in AGE_BANDS] == [(0, 40), (40, 55), (55, 200)], "age band (under 40, 40 to 54, 55 and over)"),
    ("11", "C1 non-inferiority margin 0.01", EVAL["c1_margin"] == 0.01, "exceeds -0.01"),
    ("12", "seed 0", SEED == 0, "CV seed = repeat index; model seed = 1000 x repeat + fold (other analyses: seed 0)"),
    ("7", "priors up: BMI, FPG, TG, SBP, age, family history, HbA1c", all(PRIORS[v] == +1 for v in ("bmi", "fpg", "tg", "sbp", "age", "famhx", "hba1c")), "| BMI | up | both |"),
    ("7", "priors down: HDL, exercise", all(PRIORS[v] == -1 for v in ("hdl", "exercise")), "| HDL | down | both (T4) |"),
    ("7", "exactly nine priors; sex, ALT, AST, cholesterol, DBP unconstrained", len(PRIORS) == 9 and not {"sex_male", "alt", "ast", "tc", "dbp"} & set(PRIORS), "Sex, ALT, AST, cholesterol, DBP, waist, fatty liver, GGT, smoking are unconstrained"),
]
for sec, name, ok, phrase in LOCKED:
    A.check(f"section {sec:>2}: {name}", ok)

# 3. optional: if the protocol document itself is in this folder, check its hash and that every phrase above is in it
_pfile = next((p for p in (WORK / "protocol_v1.0_LOCKED.md", WORK / "protocol" / "protocol_v1.0_LOCKED.md",
                           WORK.parent / "protocol" / "protocol_v1.0_LOCKED.md") if p.exists()), None)
if _pfile is None:
    print("\n  Protocol document not in this folder: settings are verified by value and by fingerprint (enough to proceed).")
else:
    _ptext = _pfile.read_text(encoding="utf-8")
    A.check("protocol document has the locked SHA-256", sha256_file(_pfile) == PROTOCOL_SHA, _pfile.name)
    A.check("every locked value above is stated in the protocol document", all(ph in _ptext for *_x, ph in LOCKED),
            [n for _s, n, _o, ph in LOCKED if ph not in _ptext])

# 4. deviations logged so far (date; rule; effect on the claims)
DEVIATIONS = [
    ("D-1", "2026-11-11", "RICH site 14 has under 3 years of follow-up: it stays in training and is never a test site (12 test sites, not 13).", "none"),
    ("D-2", "2026-11-11", "Main analyses use every row (training-fold median plus missingness indicators); complete cases only in sensitivity analysis S2.", "none"),
    ("D-3", "2026-11-11", "Exact zeros in triglycerides, HDL, ALT and AST are treated as not measured and set to missing; no row is removed.", "none on the rules"),
    ("D-4", "2026-11-11", "Exploratory analyses requested by the proposal were added; replaced by amendment A1 (see D-5).", "none"),
    ("D-5", "2026-11-11", "Amendment A1: fifteen exploratory items, each proposed and effective only when approved below.", "none"),
    ("D-6", "2026-11-12", "Software on the tablet: the study runs in Carnets Plus on an iPad, where XGBoost and PyTorch cannot be installed. The boosted "
                          "Cox models and the networks are fitted with NumPy code of the same algorithms and locked settings (boosting: 300 trees, depth 3, "
                          "rate 0.05, L2 penalty 1, minimum child weight 20, no random subsampling). Logged before any model result existed.", "none on the rules"),
]
print("\n  Deviations on file:")
for d_id, d_date, d_text, d_effect in DEVIATIONS:
    print(f"   {d_id}  {d_date}  {d_text}  Effect on C1 to C4: {d_effect}.")
A.check("deviations D-1 to D-6 are listed", [d[0] for d in DEVIATIONS] == [f"D-{i}" for i in range(1, 7)])
A.check("deviation D-1 is implemented in the settings (site 14 never tested)", NO_TEST_SITES == (14,))

# 5. amendment A1: items and approvals
A1_ITEMS = {
    "A1.1": "Additional metrics", "A1.2": "Additional comparators", "A1.3": "H1 reported separately from C1",
    "A1.4": "Prior ablations", "A1.5": "Extended extras E4", "A1.6": "Recalibration audit",
    "A1.7": "Cohort-comparability sensitivity", "A1.8": "Explainability", "A1.9": "Clinical expert review (gated)",
    "A1.10": "Subgroup audit", "A1.11": "Fairness-aware CKINN", "A1.12": "Population alignment",
    "A1.13": "Knowledge-graph CKINN", "A1.14": "Glycaemic strata and progression", "A1.15": "Gated branches"}
A1_SHA = sha256_text(json.dumps(A1_ITEMS, sort_keys=True))
A.check("the approval block names exactly items A1.1 to A1.15", set(A1_APPROVED) == set(A1_ITEMS), len(A1_APPROVED))
A.check("approval values are True or False", all(isinstance(v, bool) for v in A1_APPROVED.values()))

APPROVAL_FILE = DIR["audit"] / "amendment_A1_approvals.json"
_old = read_json(APPROVAL_FILE) if APPROVAL_FILE.exists() else {"items": {}}
APPROVALS = {"amendment_sha": A1_SHA, "updated": now(), "items": {}}
for item, title in A1_ITEMS.items():
    prev = _old["items"].get(item, {})
    ok = bool(A1_APPROVED.get(item, False))
    keep_date = ok and prev.get("approved") and prev.get("amendment_sha") == A1_SHA
    APPROVALS["items"][item] = {"title": title, "approved": ok,
                                "approved_on": prev.get("approved_on") if keep_date else (now() if ok else None),
                                "amendment_sha": A1_SHA if ok else None}
write_json(APPROVAL_FILE, APPROVALS)


def require_approval(item):
    """Called by every cell marked A. Stops unless the item is approved in cell 3."""
    rec = read_json(APPROVAL_FILE)["items"].get(item)
    if not rec or not rec["approved"] or rec["amendment_sha"] != A1_SHA:
        raise PermissionError(f"{item} is not approved. It is exploratory (amendment A1). Set it to True in cell 3, "
                              f"re-run cell 3, then re-run this cell.")
    return True


print("\n  Amendment A1 (exploratory; none of these can enter a C1 to C4 verdict):")
print(pd.DataFrame([{"item": k, "approved": v["approved"], "approved on": v["approved_on"] or "", "title": v["title"]}
                    for k, v in APPROVALS["items"].items()]).to_string(index=False))
_ = A.close(outputs={"settings_sha": LOCKED_SETTINGS_SHA, "deviations": [d[0] for d in DEVIATIONS], "amendment_sha": A1_SHA,
                     "approved_items": [k for k, v in A1_APPROVED.items() if v]})
keep_for_session(3)

# %%
# CELL 4 - raw workbooks: identity by SHA-256, one read, structural assertions, validated cache     [status L; protocol section 2]
# Why: proves the analysis starts from the two published files, unchanged. The workbooks are opened read-only and are
# never written. The Excel files are read with the Python standard library only (no openpyxl needed). The first run
# reads them (RICH can take a few minutes); later runs load a cache that is accepted only if its recorded source hash
# and its content hash both still match.
# On a tablet, keep the app open and in front while this cell runs the first time.
# session guard (same three lines in every cell): restores cells 2 to 5 by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 4)
A = Audit(4, "Raw workbooks: hashes, shapes, headers, cache")

RAW_SPEC = {   # the published files; header_sha is the SHA-256 of the header names joined by newlines
    "NAGALA": {"usual_name": "DatasetIntJObesity.xlsx", "sheets": ["Sheet 1"], "shape": (15464, 31), "filled_cells": 479373,
               "sha": "671359aea2019c848f969395f88e57f888eaba8463ea0351bcda8f1c947f5523",
               "header_sha": "db9dbc8f949105a4578326f421d54262641c5ff6645241fc69c36c22a980962d",
               "first": "Patient Number", "last": "Incident DM 0/1"},
    "RICH": {"usual_name": "RCHealthCareData-20180820.xlsx", "sheets": ["RC"], "shape": (211833, 25), "filled_cells": 4426500,
             "sha": "ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6",
             "header_sha": "88782dd81f55586b5753d6f87b2a8dc00b106561f48c5dcb3c8ec28341849bcc",
             "first": "id", "last": "family histroy of diabetes(1,Yes;0,No)"},
}
SEARCH_FOLDERS = [WORK, WORK / "data", WORK / "data" / "raw", WORK.parent, WORK.parent / "data" / "raw"]


def excel_letter(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def read_xlsx_first_sheet(path):
    """Read the first worksheet of an .xlsx workbook with the standard library only (an .xlsx file is a zip of XML).
    Returns (sheet names, DataFrame). The first row gives the column names. Every number is taken exactly as stored
    (for a formula cell, the value Excel saved); a cell that is absent in the file is missing (NaN)."""
    import zipfile
    import xml.etree.ElementTree as ET
    M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    P = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    def col_index(ref, _memo={}):
        letters = ref.rstrip("0123456789")
        if letters not in _memo:
            k = 0
            for ch in letters:
                k = k * 26 + (ord(ch) - 64)
            _memo[letters] = k - 1
        return _memo[letters]

    with zipfile.ZipFile(path) as z:
        sheets = [(e.get("name"), e.get(R + "id")) for e in ET.fromstring(z.read("xl/workbook.xml")).iter(M + "sheet")]
        targets = {e.get("Id"): e.get("Target") for e in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).iter(P + "Relationship")}
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(M + "si"):
                shared.append("".join((t.text or "") for t in si.iter(M + "t")))
        target = targets[sheets[0][1]]
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        header, rows = None, []
        with z.open(member) as f:
            for _event, el in ET.iterparse(f):
                if el.tag != M + "row":
                    continue
                cells = {}
                for c in el:
                    v = c.find(M + "v")
                    if v is None or v.text is None:
                        continue
                    kind, txt = c.get("t", "n"), v.text
                    if kind == "s":
                        val = shared[int(txt)]
                    elif kind == "n":
                        val = int(txt) if txt.lstrip("-").isdigit() else float(txt)
                    elif kind == "b":
                        val = bool(int(txt))
                    else:
                        raise ValueError(f"unexpected cell type {kind!r} at {c.get('r')} in {Path(path).name}")
                    cells[col_index(c.get("r"))] = val
                el.clear()
                if header is None:
                    header = [cells.get(k) for k in range(max(cells) + 1)]
                elif cells:
                    row = [None] * len(header)
                    for k, val in cells.items():
                        row[k] = val
                    rows.append(row)
    return [n for n, _id in sheets], pd.DataFrame(rows, columns=header)


def content_sha(df):
    """Order-sensitive hash of every value, the index and the column names of a data frame."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update("\n".join(map(str, df.columns)).encode("utf-8"))
    return h.hexdigest()


_XLSX = {}                                     # path -> SHA-256 of every .xlsx file found (any file name is accepted)
for _folder in SEARCH_FOLDERS:
    if _folder.is_dir():
        for _p in sorted(_folder.glob("*.xlsx")):
            if not _p.name.startswith("~$") and _p.resolve() not in _XLSX:
                _XLSX[_p.resolve()] = sha256_file(_p)
print(f"  Excel files found: {[p.name for p in _XLSX] if _XLSX else 'none'}")

RAW, DATA_SHA, RAW_PATH, _inventory = {}, {}, {}, []
for cohort, spec in RAW_SPEC.items():
    path = next((p for p, s in _XLSX.items() if s == spec["sha"]), None)
    if not A.check(f"{cohort}: workbook with the published SHA-256 was found", path is not None,
                   path.name if path else f"put the original {spec['usual_name']} in the notebook folder"):
        continue
    RAW_PATH[cohort], DATA_SHA[cohort] = path, spec["sha"]
    cache, meta_p = DIR["cache"] / f"{cohort.lower()}_raw.pkl", DIR["cache"] / f"{cohort.lower()}_raw.json"
    df, source = None, "cache"
    if cache.exists() and meta_p.exists():
        meta = read_json(meta_p)
        if meta.get("source_sha") == spec["sha"]:
            cand = pd.read_pickle(cache)
            if content_sha(cand) == meta.get("content_sha") and list(cand.shape) == meta.get("shape"):
                df = cand
    if df is None:
        t0 = time.time()
        sheets, df = read_xlsx_first_sheet(path)
        source = f"Excel, read in {time.time() - t0:.0f} s"
        df.to_pickle(cache)
        write_json(meta_p, {"cohort": cohort, "source_file": path.name, "source_sha": spec["sha"], "sheets": sheets,
                            "shape": list(df.shape), "header_sha": sha256_text("\n".join(map(str, df.columns))),
                            "content_sha": content_sha(df), "pandas": pd.__version__, "written": now()})
    meta = read_json(meta_p)
    RAW[cohort] = df
    print(f"\n  {cohort}: {path.name}  ({path.stat().st_size / 1e6:.1f} MB; loaded from {source})")
    A.check(f"{cohort}: the raw file still has the published SHA-256 after reading", sha256_file(path) == spec["sha"])
    A.check(f"{cohort}: single worksheet named {spec['sheets']}", meta["sheets"] == spec["sheets"], meta["sheets"])
    A.check(f"{cohort}: {spec['shape'][0]:,} rows by {spec['shape'][1]} columns", df.shape == spec["shape"], df.shape)
    A.check(f"{cohort}: {spec['filled_cells']:,} filled cells were read (none lost, none invented)",
            int(df.notna().sum().sum()) == spec["filled_cells"], int(df.notna().sum().sum()))
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
print("\n" + INVENTORY.to_string(index=False))


def get_raw(cohort):
    """A copy of the unchanged raw table. Later cells never modify RAW itself."""
    return RAW[cohort].copy()


_ = A.close(outputs={"files": {k: v.name for k, v in RAW_PATH.items()}, "data_sha": DATA_SHA,
                     "shapes": {k: list(v.shape) for k, v in RAW.items()},
                     "content_sha": {k: content_sha(v) for k, v in RAW.items()},
                     "inventory": "ckinn_results/audit/raw_inventory.csv"})
keep_for_session(4)

# %%
# CELL 5 - proposal coverage register: every requirement of the 26-page proposal has a cell and a status   [status L]
# Why: makes it impossible for a proposal requirement to drop out unnoticed. The final audit (cell 86) re-reads this
# register and demands, for every row, either a result file or a recorded blocking reason.
# Columns: id, proposal page, proposal section, requirement, verdict, notebook cells, status (L, D, A, G), note.
# session guard (same three lines in every cell): restores cells 2 to 5 by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 5)
REGISTER_CSV = """
id,page,section,requirement,verdict,cells,status,note
P01,1,Title,Name the method CKINN and not PINN,existing data; locked,1;88,L,No differential-equation model is implemented
P02,1,Title,Explainable fair cross-population prediction of T2D risk,existing data; locked and amendment,1;85,L;A,Each word of the title maps to a stage
P03,1-2,1,Inputs: age,existing data; locked,15;17,L,T1
P04,1-2,1,Inputs: BMI,existing data; locked,15;17,L,T1
P05,1-2,1,Inputs: blood pressure,existing data; locked,15;17,L,T1 (SBP and DBP)
P06,2,1,Inputs: fasting glucose,existing data; locked,15;17,L,T3 and T4 only
P07,2,1,Inputs: HbA1c,existing data; locked,15;51,L,NAGALA only; never across cohorts
P08,2,1,Inputs: family history,existing data; locked,15;51,L,RICH only (S4)
P09,2,1,Inputs: physical activity,existing data; locked,15;51,L,NAGALA exercise habit only (S4)
P10,2,1,Inputs: lipids,existing data; locked and amendment,15;17;51,L;A,TC and TG in T2; HDL in T4; LDL in E4 (RICH only)
P11,2,1,Inputs: smoking status,existing data; locked,15;51,L,RICH 71.6% missing; extras only
P12,2,1,Inputs: waist circumference,existing data; locked,15;51,L,NAGALA only (S4)
P13,2,1,Inputs: medication information,needs data,15;81,G,Not recorded in either cohort
P14,2,1,Inputs: previous glucose measurements,needs data,15;81,G,Only one baseline value per person
P15,2,1,Reference to ADA screening risk factors,writing,88,L,Citation must be verified before use
P16,2-3,2,Clinical consistency constraint; plausible relationships subject to data and population,existing data; locked,19;38;39,L,Soft penalty
P17,3,3A,Current diabetes classification,not identifiable,13;81,G,Zero prevalent cases; the proposal itself prefers B or C
P18,3-4,3B,Future diabetes risk P(D at t+1 given X at t),existing data; locked,43;44;46,L,3-year absolute risk
P19,4,3C,Progression: prediabetes to diabetes,existing data; amendment,13;79,A,Baseline glucose strata in both cohorts
P20,4,3C,Progression: healthy to prediabetes,existing data; amendment,13;80,A,RICH only; state known at the final visit only
P21,4,3C,Full multistate progression with transition times,needs data,81,G,Needs dated intermediate visits
P22,4-5,4,Output is risk prediction and not automatic diagnosis,existing data; locked,1;67;88,L,Wording rule
P23,5,5,Demographic: sex,existing data; locked,12;15;17,L,Male = 1 verified
P24,5,5,Demographic: ethnicity or ancestry,needs data,15;81,G,Not recorded; both cohorts East Asian
P25,5,5,Anthropometric: weight and height,existing data; locked,15,L,Recorded but redundant with BMI; not used as predictors
P26,5,5,Clinical: previous prediabetes,existing data; amendment,13;15;79,A,Derived baseline glucose stratum; used for stratified evaluation
P27,5,5,Clinical: cardiovascular disease,needs data,15;81,G,Not recorded
P28,5,5,Clinical: hypertension,existing data; locked,15;38,L,Represented by SBP with an upward prior
P29,5,5,Clinical: relevant medications,needs data,15;81,G,Not recorded
P30,6,5,Laboratory: HDL,existing data; locked,15;17,L,T4; 44.6% missing in RICH
P31,6,5,Laboratory: LDL,existing data; amendment,15;51,A,RICH only; 44.1% missing
P32,6,5,Laboratory: triglycerides and total cholesterol,existing data; locked,15;17,L,T2
P33,6,5,Lifestyle: diet,needs data,15;81,G,Not recorded
P34,6,5,Lifestyle: alcohol,existing data; amendment,15;51,A,Different coding in each cohort; within cohort only
P35,6,5,Lifestyle: sleep,needs data,15;81,G,Not recorded
P36,6,5,Assess variables for leakage; fairness; missingness; availability at prediction time,existing data; locked,7;8;15;17,L,Blocklist and registry
P37,6-7,6,Clinical priors table (ten factors),existing data; locked,38,L,BMI FPG TG SBP age HDL in shared tiers; family history HbA1c exercise within cohort
P38,7,6,Priors are not immutable laws; the model learns from data,existing data; locked,19;39;49,L,Soft penalty and lambda rule
P39,7-8,7,Architecture: processing; network; risk head; constraints; total loss; XAI,existing data; locked and amendment,22;37;39;63;67,L;A,
P40,8,7,Patient explanation and clinical explanation,existing data; amendment,67,A,Report template
P41,8,8,Network outputs a risk between 0 and 1; data loss,existing data; locked,23;37,L,Cox partial likelihood replaces cross-entropy because follow-up is censored
P42,9,9,Clinical knowledge loss: squared negative part of the derivative; weights w; lambda,existing data; locked,39;40;41,L,On the log hazard; same sign as the derivative of risk; weights equal
P43,9,9,Constraint within a specified population and reasonable range,existing data; locked,39,L,Penalty evaluated at observed covariate values only
P44,10,10,Soft constraints and not hard rules,existing data; locked,39;49,L,
P45,10-11,11,Latent metabolic state with physiological constraints,needs data,81,G,No insulin or glucose-tolerance data
P46,11,11,Differential-equation glucose-insulin model (PhINN),needs data,81,G,Model class and data needs recorded; not implemented
P47,11-12,12,Population A to population B,existing data; locked,57;58;60,L,Both directions
P48,12,12,Train on several populations and test on a new one,needs data,82,G,Needs a third cohort
P49,12,12,Train on hospitals A to C and predict in D,existing data; locked,53;55,L,Leave-one-site-out in RICH
P50,12,12,External validation; calibration; generalizability,existing data; locked,57;58;59;60,L,
P51,12-13,13,Research question on generalization; calibration; fairness; explainability,existing data; locked and amendment,85,L;A,
P52,13,14,SHAP,existing data; amendment,63,A,
P53,13,14,Integrated Gradients,existing data; amendment,63,A,
P54,13,14,Counterfactual explanations,existing data; amendment,65,A,Model behaviour only
P55,13,14,Feature perturbation,existing data; amendment,65;66,A,
P56,13,14,Monotonicity analysis,existing data; locked and amendment,28;46;64,L;A,Violation rate is locked (C4)
P57,14,14,SHAP alone is not a novelty contribution,writing,88,L,
P58,14,15,Clinical reasoning explanation consistent with the constraints,existing data; amendment,67,A,
P59,14,15,No medical advice,existing data; amendment,65;67,A,Wording rule
P60,14-15,16,Fairness groups: sex and age,existing data; locked and amendment,20;53;60;70;71,L;A,Subgroup metrics at T2 are locked
P61,15,16,Fairness groups: geographic populations,existing data; amendment,70,A,RICH sites and the two cohorts
P62,15,16,Fairness groups: socio-economic,needs data,81,G,Not recorded
P63,15,16,Group sensitivity; specificity; calibration; FNR; FPR; AUROC; AUPRC,existing data; amendment,25;27;70,A,
P64,15-16,17,Calibration curve; intercept; slope; Brier,existing data; locked and amendment,26;50;60,L;A,Intercept is added
P65,16,18,Do not build the study on the Pima dataset,existing data; locked,1,L,Not used
P66,16-17,18,Public benchmark and independent population,existing data; locked,4;57;58,L,RICH and NAGALA
P67,16-17,18,Third external clinical dataset; train on A + B and validate on C,needs data,82,G,The proposal says if available
P68,17,19,Logistic regression,existing data; locked and amendment,30;34;45,L;A,Cox is the censored-data counterpart; a weighted classifier is added
P69,17,19,Random forest,existing data; amendment,32;34;45,A,Survival forest and weighted classifier
P70,17,19,XGBoost,existing data; locked and amendment,31;34;43;45,L;A,
P71,17,19,LightGBM,existing data; amendment,34;45,A,Weighted classifier; no built-in Cox objective
P72,17,19,SVM,existing data; amendment,35;45,A,Linear and approximated radial kernel
P73,17,19,MLP and CKINN,existing data; locked,37;39;44,L,
P74,18,20,Ablation Models 1 and 3: MLP and MLP + constraints,existing data; locked,44;46,L,
P75,18,20,Ablation Model 2: MLP + XAI,existing data; amendment,76,A,Same predictions as Model 1 by definition
P76,18,20,Ablation Model 4: + cross-population,existing data; amendment,75;76,A,Importance-weighted training
P77,18,20,Ablation Model 5: + fairness,existing data; amendment,72;73;76,A,Group-loss parity penalty
P78,18-19,20,Determine what actually improves performance,existing data; amendment,47;48;49,A,Single-prior and leave-one-prior-out
P79,19-20,21,Framework steps including population alignment and decision support,existing data; locked and amendment,16;22;67;75,L;A,No deployment
P80,20-21,22,RQ1 to RQ7,existing data; locked and amendment,85,L;A,See plan section 5
P81,21,23,H1 to H5,existing data; locked and amendment,46;84;85,L;A,H1 separate from C1
P82,22,24,Sensitivity; specificity; PPV; NPV; F1,existing data; amendment,27;50,A,Censoring-weighted
P83,22,24,Clinical expert agreement,needs expertise,68,G,At least two independent clinicians
P84,23-24,25,Novelty positioning,writing,88,L,Claims limited to what section 11 supports
P85,24-25,26,Clinical knowledge graph combined with the network,existing data; amendment,77;78,A,Every edge needs a verified reference
P86,25-26,27;28,Four layers and five evaluation axes; final title and central question,existing data; locked and amendment,85;86,L;A,
"""

A = Audit(5, "Proposal coverage register")
REG = pd.read_csv(io.StringIO(REGISTER_CSV.strip()), dtype=str, keep_default_na=False)
REG_SHA = sha256_text(REG.to_csv(index=False))
REG.to_csv(DIR["audit"] / "proposal_coverage.csv", index=False, encoding="utf-8")
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
_ = A.close(outputs={"register_sha": REG_SHA, "rows": len(REG), "by_verdict": REG["verdict"].value_counts().to_dict(),
                     "blocked_ids": _blocked["id"].tolist()})
keep_for_session(5)

# %%
# CELL 6 - GATE A: governance and source authority                                              [status L]
# Why: a stop point that re-verifies stage A from the files themselves, not from memory. Stage B starts only if this
# cell ends with ALL CHECKS PASSED. A record on disk counts only when its hashes equal the current settings and files.
# session guard (same three lines in every cell): restores cells 2 to 5 by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 6)
A = Audit("gate_A", "Gate A: governance and source authority")

_recs = {c: load_record(c) for c in (2, 3, 4, 5)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else r["time"])
A.check("settings still equal the frozen fingerprint", settings_fingerprint() == LOCKED_SETTINGS_SHA)
A.check("approval record exists and matches the amendment items",
        APPROVAL_FILE.exists() and read_json(APPROVAL_FILE)["amendment_sha"] == A1_SHA)
for cohort, spec in RAW_SPEC.items():
    A.check(f"{cohort}: raw workbook re-hashed and equal to the published SHA-256",
            cohort in RAW_PATH and sha256_file(RAW_PATH[cohort]) == spec["sha"])
    _audited = _recs[4]["outputs"]["content_sha"].get(cohort) if _recs[4] else None
    A.check(f"{cohort}: table in memory equals the one audited in cell 4", cohort in RAW and content_sha(RAW[cohort]) == _audited)
    _meta = DIR["cache"] / f"{cohort.lower()}_raw.json"
    A.check(f"{cohort}: cache on disk equals the one audited in cell 4", _meta.exists() and read_json(_meta)["content_sha"] == _audited)
A.check("coverage register unchanged since cell 5", bool(_recs[5]) and REG_SHA == _recs[5]["outputs"]["register_sha"])

_approved = [k for k, v in read_json(APPROVAL_FILE)["items"].items() if v["approved"]]
print(f"\n  Amendment items approved so far: {_approved if _approved else 'none (only L and D cells can run)'}")
print("  Packages still to install before later stages: " + ("; ".join(_recs[2]["outputs"]["missing_later"]) or "none"))
_ = A.close(outputs={"stage": "A", "approved_items": _approved, "data_sha": DATA_SHA}, extra_path=DIR["gates"] / "gate_A.json")
print("\n  GATE A PASSED. Stage B (cells 7 to 14, raw-data forensics) may start.")
