# %% [markdown]
# # CKINN versus M1 and M2 on real data, version 2
# **Clinical Knowledge-Informed Neural Networks (CKINN) for type 2 diabetes risk: NAGALA (Japan), Rich Healthcare Group (China) and NHANES (USA).**
#
# This notebook carries out `protocol_v2.md`. Version 1 (`CKINN_real_data_study.ipynb`) and its results are the record; every change is an amendment listed in protocol section 2, decided after the version 1 results were seen. `AUDIT.md` gives the reasons.
#
# **Research-integrity rules followed here**
# 1. Every number in a table, figure or report is computed by a cell from stored outputs; none is typed. Cell 41 checks that no markdown cell holds a result-like number and that every number in the report traces to a stored table.
# 2. Nothing is tuned to favour CKINN. M1, M2 and CKINN get the same tuning budget; lambda follows a rule declared before any v2 result.
# 3. No evaluation row is used for fitting, tuning, threshold choice or lambda choice. Development rows are used only for tuning; thresholds come from an inner validation split of each training fold.
# 4. NULL and negative results are reported as found. Each analysis is labelled P1 (pre-specified in v1), P2 (pre-specified in v2), E (exploratory) or PH (post hoc).
# 5. Outputs are research risk estimates, not diagnoses. No clinical claim is made.
# 6. Row-level data and predictions never leave the local cache; the outputs archive holds aggregate tables, figures and reports only.
#
# **How to run.** Set `CKINN_REAL_MODE` to `smoke` (pipeline check on subsamples, under 10 minutes on four cores) or `full` (the reported analysis), `CKINN_DATA_DIR` to the folder with the three data files and `CKINN_OUT_DIR` to an output folder. On Colab, mount Drive first. Work is checkpointed per model fit; re-running the notebook resumes.

# %%
# CELL 01 | setup: imports, versions, run mode, paths, worker processes
import os, sys, json, time, hashlib, platform, re, warnings, zipfile, dataclasses, datetime as _dt
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import sklearn, xgboost, lightgbm, scipy, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

IN_COLAB = "google.colab" in sys.modules
MODE = os.environ.get("CKINN_REAL_MODE", "full")
assert MODE in ("smoke", "full"), MODE
if IN_COLAB and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")
DATA_DIR = Path(os.environ.get("CKINN_DATA_DIR", "/content/drive/MyDrive/CKINN_Real/data" if IN_COLAB else "data"))
OUT = Path(os.environ.get("CKINN_OUT_DIR", f"outputs_v2_{MODE}")).resolve()
CACHE = OUT / "cache"                       # row-level: frames, task checkpoints, predictions (never archived)
for _d in ("tables", "figures", "report", "latex"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)
(CACHE / "preds").mkdir(parents=True, exist_ok=True)
LIB = Path("ckv2_lib").resolve()
(LIB / "ckv2").mkdir(parents=True, exist_ok=True)
(LIB / "tests").mkdir(parents=True, exist_ok=True)
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
N_WORKERS = int(os.environ.get("CKINN_WORKERS", str(os.cpu_count() or 1)))
VERSIONS = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform()}
print(" | ".join(f"{k} {v}" for k, v in VERSIONS.items()))
print(f"mode={MODE} | data={DATA_DIR} | out={OUT} | workers={N_WORKERS}")
T_START = time.perf_counter()

# %% [markdown]
# ## Protocol v2 (frozen before any v2 result)
# The full protocol is `protocol_v2.md`. In short: two prediction tasks (incident diabetes in NAGALA and RICH; undiagnosed diabetes in NHANES, label corrected by amendment A11), headline predictor tier T2 without glucose or HbA1c, repeated stratified 5-fold cross-validation with 5 repeats and paired network seeds, an equal tuning budget for M1, M2 and CKINN, and two hypothesis families (CKINN versus M2, primary; CKINN versus M1), each with Holm correction across its primary endpoints. H3 and H4 are redefined (amendments A8, A9) and H5 uses sensitivity at matched specificity (A10). Labels: P1 pre-specified in v1; P2 pre-specified in v2; E exploratory; PH post hoc.
#
# The next cells write the analysis library (`ckv2`) to disk so that the model fits can run in parallel worker processes, then test it.

# %% [writefile ckv2/__init__.py]
# %% [writefile ckv2/config.py]
# %% [writefile ckv2/data.py]
# %% [writefile ckv2/prep.py]
# %% [writefile ckv2/missing.py]
# %% [writefile ckv2/nets.py]
# %% [writefile ckv2/models.py]
# %% [writefile ckv2/metrics.py]
# %% [writefile ckv2/engine.py]
# %% [writefile ckv2/plan.py]
# %% [writefile ckv2/analysis.py]
# %% [writefile tests/test_ckv2.py]

# %%
# CELL 15 | unit tests of the library (toy arrays only): metrics against scikit-learn, lifelines and scikit-survival where installed,
# monotone baselines, repeated folds, penalty, CKINN(lambda = 0) = M2
import importlib
import pytest
for _m in [m for m in list(sys.modules) if m == "ckv2" or m.startswith("ckv2.")]:
    del sys.modules[_m]
_t0 = time.perf_counter()
_rc = pytest.main(["-q", "-p", "no:cacheprovider", "-W", "ignore", str(LIB / "tests" / "test_ckv2.py")])
assert _rc == 0, "unit tests failed"
from ckv2 import config as C, data as D, metrics as MT, engine as EN, plan as PL, analysis as AN
from ckv2.missing import scen_tag, scenarios
print(f"CELL 15 OK | unit tests passed | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 16 | frozen configuration and protocol hash (protocol section 9); full mode stops if the configuration differs
CFG = C.make_config(MODE)
PROTOCOL_HASH = C.config_hash(CFG)
EXPECTED_FULL_HASH = "7cdc5bdde12b17df19a15650f00f1fe693168b33f39b193219ded6d70ea0468b"
(OUT / "report" / "frozen_protocol_v2.json").write_text(json.dumps({"config": json.loads(CFG.to_json()), "sha256": PROTOCOL_HASH}, indent=1))
print("configuration sha256:", PROTOCOL_HASH)
if MODE == "full":
    assert PROTOCOL_HASH == EXPECTED_FULL_HASH, "the full-mode configuration differs from the frozen protocol: record a deviation first"
print("CELL 16 OK | configuration frozen")

# %%
# CELL 17 | data: locate and hash the three files, harmonise, audit, task frames, development / evaluation split, feature sets
_t0 = time.perf_counter()
_roots = [DATA_DIR, Path("."), Path("/content"), Path("/mnt/user-data/uploads")]
FILES = {"NAGALA": D.find_file(_roots, ["*DatasetIntJObesity*.xlsx"]), "RICH": D.find_file(_roots, ["*RCHealthCareData*.xlsx"]),
         "NHANES": D.find_file(_roots, ["*nhanes_t2d_extract*.csv"])}
FILE_HASH = {k: D.sha256_file(v) for k, v in FILES.items()}
EXPECTED_SHA = {"NAGALA": "671359aea2019c848f969395f88e57f888eaba8463ea0351bcda8f1c947f5523",
                "RICH": "ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6",
                "NHANES": "d39795db4a687cd5c37e7eb1e85ac8850f78192f031d0dcf3887e413f976899f"}
for k in FILES:
    assert FILE_HASH[k] == EXPECTED_SHA[k], f"{k}: file differs from the one audited (sha256 {FILE_HASH[k][:16]})"
    print(f"{k:7s} {FILES[k].name} sha256 {FILE_HASH[k][:16]}... verified")
RAW = {"NAGALA": D.load_nagala(D.read_cached(FILES["NAGALA"], CACHE)), "RICH": D.load_rich(D.read_cached(FILES["RICH"], CACHE)),
       "NHANES": D.load_nhanes(D.read_cached(FILES["NHANES"], CACHE))}
_audit = []
for k, d in RAW.items():
    oor = {c: float(((d[c] < lo) | (d[c] > hi)).mean()) for c, (lo, hi) in C.RANGES.items() if d[c].notna().any()}
    _audit.append({"cohort": k, "rows": len(d), "labelled": int(d.label.notna().sum()), "positives": int(np.nansum(d.label)),
                   "prevalence_pct": 100 * float(np.nanmean(d.label)), "predictors_present": int(sum(d[c].notna().any() for c in C.ALL_FEATURES)),
                   "max_pct_out_of_range": 100 * max(oor.values()), "worst_variable": max(oor, key=oor.get),
                   "follow_up_min_years": float(d.follow_years.min()), "follow_up_median_years": float(d.follow_years.median()),
                   "follow_up_max_years": float(d.follow_years.max())})
DATA_CARD = pd.DataFrame(_audit)
MISS = pd.DataFrame({k: d[C.ALL_FEATURES].isna().mean() for k, d in RAW.items()}).reset_index().rename(columns={"index": "variable"})
_nh = RAW["NHANES"]
NH_LABEL = pd.DataFrame([{"definition": "v1 (no reported diagnosis)", "labelled": int(_nh.label_v1.notna().sum()), "positives": int(np.nansum(_nh.label_v1))},
                         {"definition": "v2 (A11: also no medication, answered, fasting >= 8 h for FPG)", "labelled": int(_nh.label.notna().sum()),
                          "positives": int(np.nansum(_nh.label))}])
TASK = {k: D.task_frame(d) for k, d in RAW.items()}
TASK["NHANESv1"] = D.task_frame(RAW["NHANES"], "label_v1")
DEV, EVAL = {}, {}
for k, t in TASK.items():
    key = "NHANES" if k == "NHANESv1" else k                      # v1 split seed
    DEV[k], EVAL[k] = D.dev_eval_split(t, key, CFG.dev_frac)
    DEV[k] = D.subsample(DEV[k], CFG.max_rows.get(key, 0) // 4, "dev" + k)
    EVAL[k] = D.subsample(EVAL[k], CFG.max_rows.get(key, 0), "eval" + k)
    DEV[k] = D.cap_rows(DEV[k], CFG.dev_max_rows, key)
SPLIT = pd.DataFrame([{"cohort": k, "dev_rows": len(DEV[k]), "dev_events": int(DEV[k].y.sum()), "eval_rows": len(EVAL[k]),
                       "eval_events": int(EVAL[k].y.sum())} for k in TASK])
FEATS = {(k, "T2"): D.features_for(TASK[k], "T2") for k in ("NAGALA", "RICH", "NHANES")}
FEATS.update({(k, "T3"): D.features_for(TASK[k], "T3") for k in ("NAGALA", "RICH")})
FEATS[("NHANESv1", "T2")] = FEATS[("NHANES", "T2")]
for (k, tier), f in FEATS.items():
    D.assert_no_leakage(f, TASK[k].task.iloc[0], tier)
for _bad, _task, _tier in ((["fpg_mmol"], "A", "T2"), (["hba1c_pct"], "B", "T2"), (["age_years", "age_years"], "B", "T2")):
    try:
        D.assert_no_leakage(_bad, _task, _tier)
        raise RuntimeError("leakage guard did not fire")
    except AssertionError:
        pass
SHARED = PL.shared_features(TASK["NAGALA"], TASK["RICH"], C.LOW_COST + C.PANEL)
POOLED_FEATS = list(dict.fromkeys(FEATS[("NAGALA", "T2")] + FEATS[("RICH", "T2")]))
FEAT_TBL = pd.DataFrame([{"cohort": k, "tier": t, "n_features": len(f), "features": ", ".join(f)} for (k, t), f in FEATS.items()]
                        + [{"cohort": "NAGALA+RICH", "tier": "shared (transport)", "n_features": len(SHARED), "features": ", ".join(SHARED)}])
FR = {k: EVAL[k] for k in EVAL}
FR.update({f"{k}_dev": DEV[k] for k in DEV})
FR["POOL"] = pd.concat([EVAL["NAGALA"], EVAL["RICH"]], ignore_index=True)
EN.save_frames(CACHE, FR)
FOLDS = {k: D.repeated_folds(EVAL[k].y.to_numpy(), "NHANES" if k == "NHANESv1" else k, CFG.n_splits, CFG.n_repeats) for k in EVAL}
for name, df in (("table01_data_card", DATA_CARD), ("table02_missingness", MISS), ("table03_nhanes_label", NH_LABEL),
                 ("table04_split", SPLIT), ("table05_feature_sets", FEAT_TBL)):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(DATA_CARD.round(3).to_string(index=False)); print(NH_LABEL.to_string(index=False)); print(SPLIT.to_string(index=False))
    print(FEAT_TBL.to_string(index=False))
print(f"CELL 17 OK | data loaded, audited and split | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 18 | tuning on development rows only (protocol section 5): 32 configurations for M1 and M2; CKINN = M2's best of its first 24
# plus 8 lambda values scored with lambda 0 on repeated 3-fold CV; competitors on their own grids
_t0 = time.perf_counter()
HP, LAM, TUNE_ROWS, LAM_ROWS = {}, {}, [], []
_specs = {c: PL.tune_specs(c, DEV[c], FEATS[(c, "T2")], CFG) for c in ("NAGALA", "RICH", "NHANES")}
EN.run_tasks([s for v in _specs.values() for s in v], CFG, CACHE, N_WORKERS, "tuning")
for c, sp in _specs.items():
    sc = PL.score_specs(sp, DEV[c], CACHE)
    g = sc.groupby(["model", "ci"]).score.mean().reset_index()
    for m, gm in g.groupby("model"):
        best = gm.loc[gm.score.idxmin()]
        HP[(c, m)] = PL.config_lists(c, CFG)[m][int(best.ci)]
        TUNE_ROWS += [{"cohort": c, "model": m, "config": int(r.ci), "dev_score": r.score, "chosen": int(r.ci) == int(best.ci),
                       "hp": json.dumps(PL.config_lists(c, CFG)[m][int(r.ci)])} for r in gm.itertuples()]
    gm2 = g[(g.model == "M2") & (g.ci < CFG.tune_ck_base)]
    HP[(c, "CKINN")] = PL.config_lists(c, CFG)["M2"][int(gm2.loc[gm2.score.idxmin()].ci)]
_lspecs = {(c, t): PL.lam_specs(c, t, DEV[c], FEATS[(c, t)], HP[(c, "CKINN")], CFG)
           for c, t in (("NAGALA", "T2"), ("RICH", "T2"), ("NHANES", "T2"), ("NAGALA", "T3"), ("RICH", "T3"))}
EN.run_tasks([s for v in _lspecs.values() for s in v], CFG, CACHE, N_WORKERS, "lambda")
for (c, t), sp in _lspecs.items():
    LAM[(c, t)], tab = PL.select_lambda(PL.score_specs(sp, DEV[c], CACHE), CFG.lam_grid)
    LAM_ROWS += [dict(cohort=c, tier=t, **r) for r in tab.to_dict("records")]
for m in list(C.MAIN_MODELS):
    HP[("NHANESv1", m)] = HP[("NHANES", m)]
LAM[("NHANESv1", "T2")] = LAM[("NHANES", "T2")]
TUNING, LAMBDAS = pd.DataFrame(TUNE_ROWS), pd.DataFrame(LAM_ROWS)
TUNING.to_csv(OUT / "tables" / "table06_tuning.csv", index=False)
LAMBDAS.to_csv(OUT / "tables" / "table07_lambda.csv", index=False)
print(TUNING[TUNING.chosen].drop(columns=["chosen"]).round(4).to_string(index=False))
print(LAMBDAS.round(5).to_string(index=False))
assert set(LAMBDAS.groupby(["cohort", "tier"]).chosen.sum()) == {1}
print(f"CELL 18 OK | tuning done on development rows only | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 19 | main runs: 5 repeats x 5 folds, every model, network seed r in repeat r; test-time missingness scenarios;
# repeat 0 also stores attributions, violation statistics, sparse-region probes and partial dependence
_t0 = time.perf_counter()
CV_SPECS = {c: PL.cv_specs(c, "T2", FOLDS[c], FEATS[(c, "T2")], HP, LAM, CFG) for c in ("NAGALA", "RICH", "NHANES")}
SEED_SPECS = {c: PL.seed_specs(c, "T2", FOLDS[c], FEATS[(c, "T2")], HP, LAM, CFG) for c in ("NAGALA", "RICH", "NHANES")}
EN.run_tasks([s for c in CV_SPECS for s in CV_SPECS[c] + SEED_SPECS[c]], CFG, CACHE, N_WORKERS, "main runs and seed study")
print(f"CELL 19 OK | main runs and seed study fitted | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 20 | repeat 0 analyses: constraint variants (leave one out, wrong signs, alternative tables, lambda path), tier T3,
# RICH without family history, NHANES with the v1 label, 3-year IPCW models
_t0 = time.perf_counter()
VAR_SPECS = {c: PL.variant_specs(c, FOLDS[c], FEATS[(c, "T2")], HP, LAM, CFG) for c in ("NAGALA", "RICH", "NHANES")}
_t3m = ("LR", "XGB", "XGB_mono", "M1", "M2", "CKINN")
T3_SPECS = {c: PL.cv_specs(c, "T3", FOLDS[c], FEATS[(c, "T3")], HP, LAM, CFG, models=_t3m, repeats=[0], scen=False, extras=False)
            for c in ("NAGALA", "RICH")}
_nofh = [f for f in FEATS[("RICH", "T2")] if f != "family_history_dm"]
NOFH_SPECS = PL.cv_specs("RICH", "T2", FOLDS["RICH"], _nofh, HP, LAM, CFG, models=("LR", "M2", "CKINN"), repeats=[0], variant="noFH",
                         scen=False, extras=False)
NHV1_SPECS = PL.cv_specs("NHANESv1", "T2", FOLDS["NHANESv1"], FEATS[("NHANESv1", "T2")], HP, LAM, CFG, models=("LR", "M2", "CKINN"),
                         repeats=[0], scen=False, extras=False)
IPCW_SPECS = {c: PL.ipcw_specs(c, FOLDS[c], FEATS[(c, "T2")], HP, LAM, CFG, EVAL[c]) for c in ("NAGALA", "RICH")}
_all = [s for v in VAR_SPECS.values() for s in v] + [s for v in T3_SPECS.values() for s in v] + NOFH_SPECS + NHV1_SPECS + \
       [s for v in IPCW_SPECS.values() for s in v]
EN.run_tasks(_all, CFG, CACHE, N_WORKERS, "repeat-0 analyses")
print(f"CELL 20 OK | repeat-0 analyses fitted | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 21 | transport (H3), reweighting, pooled training and RICH leave-one-site-out
_t0 = time.perf_counter()
TR_SPECS = PL.transport_specs(EVAL, FOLDS, SHARED, HP, LAM, CFG)
POOL_SPECS = PL.pooled_specs(EVAL, FOLDS, POOLED_FEATS, HP, LAM, CFG)
SITE_SPECS, SITES_HELD = PL.site_specs(EVAL["RICH"], FEATS[("RICH", "T2")], HP, LAM, CFG)
EN.run_tasks(TR_SPECS + POOL_SPECS + SITE_SPECS, CFG, CACHE, N_WORKERS, "transport, pooled, sites")
print(f"held-out RICH sites: {SITES_HELD}")
print(f"CELL 21 OK | transport, pooled and site analyses fitted | {time.perf_counter() - _t0:.0f}s")

# %% [markdown]
# ## Analysis of the stored predictions
# Everything below reads the checkpointed predictions; no model is fitted. Each table states its label (P1, P2, E).

# %%
# CELL 23 | assemble out-of-fold predictions of every stage and save them locally (row-level: never archived)
_t0 = time.perf_counter()
COHORTS = ("NAGALA", "RICH", "NHANES")
NE = {c: len(EVAL[c]) for c in EVAL}
Y = {c: EVAL[c].y.to_numpy().astype(float) for c in EVAL}
MAINRUNS = [(r, r) for r in range(CFG.n_repeats)]
SEEDRUNS = [(0, s) for s in range(CFG.n_seeds)]
NN = list(C.NN_MODELS)
P, THR, FOLDID = {}, {}, {}
for c in COHORTS:
    P[c], THR[c], FOLDID[c] = AN.collect(CV_SPECS[c] + SEED_SPECS[c], NE[c], CACHE)
PV = {c: AN.collect(VAR_SPECS[c], NE[c], CACHE, key=lambda s: s["_variant"])[0] for c in VAR_SPECS}
PT3 = {c: AN.collect(T3_SPECS[c], NE[c], CACHE)[0] for c in T3_SPECS}
PNOFH = AN.collect(NOFH_SPECS, NE["RICH"], CACHE)[0]
PNHV1 = AN.collect(NHV1_SPECS, NE["NHANESv1"], CACHE)[0]
PIPCW = {c: AN.collect(IPCW_SPECS[c], NE[c], CACHE, key=lambda s: s["_model"])[0] for c in IPCW_SPECS}
_ext = [s for s in TR_SPECS if s["id"].startswith("trext")]
_ref = [s for s in TR_SPECS if s["id"].startswith("trref")]
_rw = [s for s in TR_SPECS if s["id"].startswith("trrw")]
TRX = {(s["_src"], s["_tgt"], s["_model"], s["_s"]): EN.load(CACHE, s["id"])["preds"]["tgt"].astype(float) for s in _ext}
TRW = {(s["_src"], s["_tgt"], s["_model"]): EN.load(CACHE, s["id"])["preds"]["tgt"].astype(float) for s in _rw}
TRREF = {}
for _tgt in ("RICH", "NAGALA"):
    _p = AN.collect([s for s in _ref if s["_tgt"] == _tgt], NE[_tgt], CACHE, key=lambda s: (s["_model"], s["_s"]))[0]
    TRREF.update({(_tgt, m, s): v["none"].astype(float) for (m, s), v in _p.items()})
POOLP = {}
for s in POOL_SPECS:
    r_ = EN.load(CACHE, s["id"])
    for t in s["tests"]:
        off = 0 if t["tag"] == "NAGALA" else NE["NAGALA"]
        POOLP.setdefault((s["_model"], t["tag"]), np.full(NE[t["tag"]], np.nan))[t["idx"] - off] = r_["preds"][t["tag"]]
assert all(np.isfinite(v).all() for v in POOLP.values())
SITEP = {}
for s in SITE_SPECS:
    SITEP.setdefault(s["_model"], np.full(NE["RICH"], np.nan))[s["tests"][0]["idx"]] = EN.load(CACHE, s["id"])["preds"]["none"]
SITE_ROWS = np.isfinite(SITEP["CKINN"])
for c in COHORTS:
    np.savez_compressed(CACHE / "preds" / f"{c}_runs.npz", y=Y[c],
                        **{f"{m}|r{r}|s{s}|{tag}": v for (m, r, s), d in P[c].items() for tag, v in d.items()},
                        **{f"THRSPEC|{m}|r{r}|s{s}": THR[c][(m, r, s)][0] for (m, r, s) in THR[c]},
                        **{f"FOLD|{m}|r{r}|s{s}": FOLDID[c][(m, r, s)] for (m, r, s) in FOLDID[c]})
np.savez_compressed(CACHE / "preds" / "transport.npz", **{f"ext|{a}|{b}|{m}|s{s}": v for (a, b, m, s), v in TRX.items()},
                    **{f"ref|{a}|{m}|s{s}": v for (a, m, s), v in TRREF.items()}, y_NAGALA=Y["NAGALA"], y_RICH=Y["RICH"])
print(f"CELL 23 OK | predictions assembled for {sum(len(P[c]) for c in COHORTS)} cohort-model-run keys | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 24 | performance of every model (P2): run-averaged AUROC, AUPRC, Brier, scaled Brier, calibration slope,
# calibration-in-the-large and ECE, with 95% intervals from the paired bootstrap (descriptive draws)
_t0 = time.perf_counter()
_pl = {}
for c in COHORTS:
    vecs = {m: [P[c][(m, r, s)]["none"] for r, s in MAINRUNS] for m in C.MAIN_MODELS}
    vecs.update({m: [P[c][(m, 0, 0)]["none"]] for m in C.RUN0_EXTRA})
    vecs.update({f"{m}_r0": [P[c][(m, 0, 0)]["none"]] for m in NN})
    _pl[c] = {"y": Y[c], "vecs": vecs, "stats": ["auroc", "brier", "slope", "citl", "ece"]}
BOOT_DESC = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
rows = []
for c in COHORTS:
    b, y = BOOT_DESC[c], Y[c]
    prev = y.mean()
    for m in list(C.MAIN_MODELS) + list(C.RUN0_EXTRA):
        runs = MAINRUNS if m in C.MAIN_MODELS else [(0, 0)]
        row = {"cohort": c, "model": m, "runs": len(runs), "n": len(y), "events": int(y.sum()),
               "auprc": float(np.mean([AN.auprc(y, P[c][(m, r, s)]["none"]) for r, s in runs]))}
        for st in ("auroc", "brier", "slope", "citl", "ece"):
            row[st] = b["point"][(m, st)]
            row[st + "_lo"], row[st + "_hi"] = AN.ci(b["draws"][(m, st)])
        row["scaled_brier"] = 1 - row["brier"] / (prev * (1 - prev))
        rows.append(row)
PERF = pd.DataFrame(rows)
PERF.to_csv(OUT / "tables" / "table08_performance.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(PERF[["cohort", "model", "runs", "events", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier", "scaled_brier", "slope", "citl", "ece"]].round(4).to_string(index=False))
print(f"CELL 24 OK | performance table | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 25 | hypothesis families A (CKINN minus M2, primary) and B (CKINN minus M1): H1 to H5 with Holm correction (P2)
_t0 = time.perf_counter()
GROUPS, N_BANDS = {}, {}
_pl = {}
for c in COHORTS:
    GROUPS[c], N_BANDS[c] = AN.groups_for(EVAL[c], Y[c], CFG.min_group_events)
    _pl[c] = {"y": Y[c], "vecs": {m: [P[c][(m, r, s)]["none"] for r, s in MAINRUNS] for m in NN},
              "stats": ["auroc", "brier", "slope_err", "tpr_sex_gap", "tpr_age_range"], "groups": GROUPS[c], "fpr": CFG.fpr_matched}
for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
    v = {}
    for m in NN:
        v[f"ext_{m}"] = [TRX[(src, tgt, m, s)] for s in range(CFG.n_seeds)]
        v[f"ref_{m}"] = [TRREF[(tgt, m, s)] for s in range(CFG.n_seeds)]
    for m in ("LR", "XGB", "XGB_mono"):
        v[f"ext_{m}"], v[f"ref_{m}"] = [TRX[(src, tgt, m, 0)]], [TRREF[(tgt, m, 0)]]
    _pl[f"{src}->{tgt}"] = {"y": Y[tgt], "vecs": v, "stats": ["auroc", "citl_abs", "brier", "slope"]}
BOOT_FAM = AN.bootstrap(_pl, CFG.n_boot, N_WORKERS)
# H4: across-seed attribution stability on the explained rows of repeat 0
STAB, STAB_DRAWS = {}, {}
for c in COHORTS:
    rows_m = {m: [] for m in NN}
    for k in range(CFG.n_splits):
        for m in NN:
            A = [EN.load(CACHE, PL.tid("cv", c, "T2", "main" if s == 0 else "seed", "r0", f"k{k}", m, f"s{s}"))["attr"] for s in range(CFG.n_seeds)]
            rows_m[m].append(AN.stability_rows(A))
    STAB[c] = {m: np.concatenate(v) for m, v in rows_m.items()}
    STAB_DRAWS[c] = AN.mean_boot(STAB[c], CFG.n_boot, C.seed_for("h4", c))
EP = []
for fam, F in (("A: CKINN-M2", "M2"), ("B: CKINN-M1", "M1")):
    for c in COHORTS:
        u, pt, dr = f"{c} (T2)", BOOT_FAM[c]["point"], BOOT_FAM[c]["draws"]
        d = lambda st: (pt[("CKINN", st)] - pt[(F, st)], dr[("CKINN", st)] - dr[(F, st)])
        EP.append(AN.endpoint_row(fam, "H1", u, "AUROC (non-inferiority, margin 0.01)", "higher", *d("auroc"), margin=CFG.ni_margin))
        EP.append(AN.endpoint_row(fam, "H1", u, "AUROC (superiority, descriptive)", "higher", *d("auroc"), primary=False))
        EP.append(AN.endpoint_row(fam, "H2", u, "Brier score", "lower", *d("brier")))
        EP.append(AN.endpoint_row(fam, "H2", u, "|calibration slope - 1|", "lower", *d("slope_err")))
        s_ck, s_f = STAB[c]["CKINN"], STAB[c][F]
        EP.append(AN.endpoint_row(fam, "H4", u, "across-seed attribution agreement (Spearman)", "higher",
                                  float(np.nanmean(s_ck) - np.nanmean(s_f)), STAB_DRAWS[c]["CKINN"] - STAB_DRAWS[c][F],
                                  note=f"{len(s_ck)} explained rows, {CFG.n_seeds} seeds"))
        EP.append(AN.endpoint_row(fam, "H5", u, "sex gap in sensitivity at FPR 0.10", "lower", *d("tpr_sex_gap")))
        if N_BANDS[c] >= 2:
            EP.append(AN.endpoint_row(fam, "H5", u, "age-band range in sensitivity at FPR 0.10", "lower", *d("tpr_age_range"),
                                      note=f"{N_BANDS[c]} age bands with >= {CFG.min_group_events} events"))
    for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
        u = f"{src} -> {tgt}"
        pt, dr = BOOT_FAM[u.replace(" ", "")]["point"], BOOT_FAM[u.replace(" ", "")]["draws"]
        gap = lambda m, D_: D_[(f"ref_{m}", "auroc")] - D_[(f"ext_{m}", "auroc")]
        EP.append(AN.endpoint_row(fam, "H3", u, "transport gap (target CV AUROC - external AUROC)", "lower",
                                  gap("CKINN", pt) - gap(F, pt), gap("CKINN", dr) - gap(F, dr)))
        EP.append(AN.endpoint_row(fam, "H3", u, "|calibration-in-the-large| in the target", "lower",
                                  pt[("ext_CKINN", "citl_abs")] - pt[(f"ext_{F}", "citl_abs")], dr[("ext_CKINN", "citl_abs")] - dr[(f"ext_{F}", "citl_abs")]))
EP, VERDICTS = AN.verdicts(pd.DataFrame(EP), CFG.alpha)
EP.to_csv(OUT / "tables" / "table09_hypothesis_endpoints.csv", index=False)
VERDICTS.to_csv(OUT / "tables" / "table10_verdicts.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
    print(EP[["family", "hypothesis", "unit", "endpoint", "difference", "ci_lo", "ci_hi", "p_pred_holm", "p_opp_holm", "status"]].round(5).to_string(index=False))
    print(VERDICTS.to_string(index=False))
assert EP[EP.primary].groupby("family").size().nunique() == 1, "families must have the same endpoints"
print(f"CELL 25 OK | verdicts: " + "; ".join(f"{r.family[:1]} {r.hypothesis} {r.verdict}" for r in VERDICTS.itertuples()) + f" | {time.perf_counter() - _t0:.0f}s")

# %%
from scipy.stats import norm
# CELL 26 | cross-checks of H1 (P2): DeLong on run 0, corrected repeated-CV t-test over 25 folds, TOST equivalence, RICH cluster
# bootstrap by site, minimum detectable effect for NAGALA; variance components (sampling, seed, split)
_t0 = time.perf_counter()
rows = []
_clu = {"RICH": {**{k: v for k, v in _pl["RICH"].items() if k != "groups"}, "stats": ["auroc", "brier", "slope_err"], "mode": "cluster",
                 "cluster": EVAL["RICH"].site.to_numpy()}}
BOOT_CLU = AN.bootstrap(_clu, CFG.n_boot, N_WORKERS)
for fam, F in (("A: CKINN-M2", "M2"), ("B: CKINN-M1", "M1")):
    for c in COHORTS:
        y = Y[c]
        dl = MT.delong_paired(y, P[c][("CKINN", 0, 0)]["none"], P[c][(F, 0, 0)]["none"])
        diffs = []
        for r, s in MAINRUNS:
            fd = FOLDID[c][("CKINN", r, s)]
            diffs.append(AN.fold_aurocs(y, P[c][("CKINN", r, s)]["none"], fd) - AN.fold_aurocs(y, P[c][(F, r, s)]["none"], fd))
        nb = MT.corrected_t(np.concatenate(diffs), 1.0, CFG.n_splits - 1.0)
        dr = BOOT_FAM[c]["draws"][("CKINN", "auroc")] - BOOT_FAM[c]["draws"][(F, "auroc")]
        p_tost = max(MT.p_one_sided(dr, -CFG.ni_margin, +1), MT.p_one_sided(dr, CFG.ni_margin, -1))
        row = {"family": fam, "cohort": c, "auroc_diff_runavg": BOOT_FAM[c]["point"][("CKINN", "auroc")] - BOOT_FAM[c]["point"][(F, "auroc")],
               "boot_lo": AN.ci(dr)[0], "boot_hi": AN.ci(dr)[1], "boot_sd": float(np.std(dr, ddof=1)),
               "p_superiority_two_sided": min(1.0, 2 * min(MT.p_one_sided(dr, 0, +1), MT.p_one_sided(dr, 0, -1))),
               "p_tost_margin_0.01": p_tost, "equivalent_within_0.01": p_tost < CFG.alpha,
               "delong_diff_run0": dl["diff"], "delong_se": dl["se"], "delong_p": dl["p_two_sided"],
               "nb_mean_fold_diff": nb["mean"], "nb_lo": nb["lo"], "nb_hi": nb["hi"], "nb_p": nb["p_two_sided"],
               "mde_80pct_power": (norm.ppf(0.975) + norm.ppf(0.80)) * float(np.std(dr, ddof=1)),
               "power_noninferiority_if_true_diff_0": float(norm.cdf(CFG.ni_margin / float(np.std(dr, ddof=1)) - norm.ppf(0.975)))}
        if c == "RICH":
            cd = BOOT_CLU["RICH"]["draws"]
            dcl = cd[("CKINN", "auroc")] - cd[(F, "auroc")]
            row.update({"cluster_lo": AN.ci(dcl)[0], "cluster_hi": AN.ci(dcl)[1],
                        "cluster_brier_lo": AN.ci(cd[("CKINN", "brier")] - cd[(F, "brier")])[0],
                        "cluster_brier_hi": AN.ci(cd[("CKINN", "brier")] - cd[(F, "brier")])[1]})
        rows.append(row)
XCHECK = pd.DataFrame(rows)
XCHECK.to_csv(OUT / "tables" / "table11_h1_crosschecks.csv", index=False)
# variance components: sampling SE (bootstrap of run 0 alone), seed SD (repeat 0, 5 seeds), between-run SD (5 main runs)
rows = []
for c in COHORTS:
    y = Y[c]
    for m in NN + ["LR", "XGB_mono"]:
        for st, fn in (("auroc", lambda p: MT.auroc_w(MT.Ranked(p, y))), ("brier", lambda p: MT.brier_w(y, p))):
            row = {"cohort": c, "model": m, "metric": st,
                   "sampling_se_run0": float(np.std(BOOT_DESC[c]["draws"][(f"{m}_r0" if m in NN else m, st)], ddof=1)) if m in NN else np.nan,
                   "sd_between_main_runs": float(np.std([fn(P[c][(m, r, s)]["none"]) for r, s in MAINRUNS], ddof=1))}
            row["sd_across_seeds_repeat0"] = float(np.std([fn(P[c][(m, 0, s)]["none"]) for _, s in SEEDRUNS], ddof=1)) if m in NN else np.nan
            rows.append(row)
    for F in ("M2", "M1"):
        for st, fn in (("auroc", lambda p: MT.auroc_w(MT.Ranked(p, y))), ("brier", lambda p: MT.brier_w(y, p))):
            rows.append({"cohort": c, "model": f"CKINN-{F}", "metric": st,
                         "sd_across_seeds_repeat0": float(np.std([fn(P[c][("CKINN", 0, s)]["none"]) - fn(P[c][(F, 0, s)]["none"]) for _, s in SEEDRUNS], ddof=1)),
                         "sd_between_main_runs": float(np.std([fn(P[c][("CKINN", r, s)]["none"]) - fn(P[c][(F, r, s)]["none"]) for r, s in MAINRUNS], ddof=1)),
                         "sampling_se_run0": float(np.std(BOOT_DESC[c]["draws"][("CKINN_r0", st)] - BOOT_DESC[c]["draws"][(f"{F}_r0", st)], ddof=1))})
VARCOMP = pd.DataFrame(rows)
VARCOMP.to_csv(OUT / "tables" / "table12_variance_components.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(XCHECK.round(5).to_string(index=False)); print(VARCOMP.round(5).to_string(index=False))
print(f"CELL 26 OK | cross-checks and variance components | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 27 | CKINN against every competitor (P2, descriptive, no multiplicity correction): AUROC and Brier differences
_t0 = time.perf_counter()
rows = []
for c in COHORTS:
    pt, dr = BOOT_DESC[c]["point"], BOOT_DESC[c]["draws"]
    for m in list(C.MAIN_MODELS) + list(C.RUN0_EXTRA):
        if m == "CKINN":
            continue
        ck = "CKINN" if m in C.MAIN_MODELS else "CKINN_r0"
        for st in ("auroc", "brier"):
            d = dr[(ck, st)] - dr[(m, st)]
            rows.append({"cohort": c, "comparator": m, "metric": st, "ckinn_minus_comparator": pt[(ck, st)] - pt[(m, st)],
                         "lo": AN.ci(d)[0], "hi": AN.ci(d)[1], "runs": CFG.n_repeats if m in C.MAIN_MODELS else 1})
SEC = pd.DataFrame(rows)
SEC.to_csv(OUT / "tables" / "table13_ckinn_vs_competitors.csv", index=False)
print(SEC[SEC.metric == "auroc"].round(4).to_string(index=False))
print(f"CELL 27 OK | competitor comparisons | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 28 | calibration (P2): reliability curves with 95% bootstrap bands (run 0, seed 0; equal-frequency tenths) for M1, M2, CKINN,
# LR and monotone XGBoost; slope, calibration-in-the-large and ECE intervals are in table 8
_t0 = time.perf_counter()
CAL_MODELS = ("M1", "M2", "CKINN", "LR", "XGB_mono")
rows = []
for c in COHORTS:
    y = Y[c]
    for m in CAL_MODELS:
        p = P[c][(m, 0, 0)]["none"].astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        nb_ = len(edges) - 1
        obs = np.bincount(b, y, nb_) / np.bincount(b, minlength=nb_)
        pred = np.bincount(b, p, nb_) / np.bincount(b, minlength=nb_)
        bands = []
        for w in MT.boot_weight_iter(len(y), CFG.n_boot_band, C.seed_for("band", c, m)):
            bands.append(np.bincount(b, w * y, nb_) / np.maximum(np.bincount(b, w, nb_), 1e-12))
        lo, hi = np.quantile(np.vstack(bands), [0.025, 0.975], axis=0)
        rows += [{"cohort": c, "model": m, "bin": i, "mean_predicted": pred[i], "observed": obs[i], "obs_lo": lo[i], "obs_hi": hi[i],
                  "n": int((b == i).sum())} for i in range(nb_)]
CALCURVE = pd.DataFrame(rows)
CALCURVE.to_csv(OUT / "tables" / "table14_reliability_curves.csv", index=False)
print(f"CELL 28 OK | reliability curves | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 29 | clinical utility (P2): decision curves (net benefit, run-averaged) with intervals at the stated thresholds, and threshold
# metrics at fixed sensitivity 0.80 and fixed specificity 0.90 (thresholds from the inner validation split of each training fold)
_t0 = time.perf_counter()
UTIL_MODELS = ("M1", "M2", "CKINN", "LR", "cLR", "GAM", "XGB", "XGB_mono", "LGBM_mono")
rows, rows_pts, rows_thr = [], [], []
_pl = {}
for c in COHORTS:
    a, b_, st = CFG.dca_a if c == "NHANES" else CFG.dca_b
    pts = CFG.nb_points_a if c == "NHANES" else CFG.nb_points_b
    grid = np.round(np.arange(a, b_ + 1e-9, st), 4)
    y = Y[c]
    for t in grid:
        rows.append({"cohort": c, "model": "treat all", "threshold": t, "net_benefit": MT.net_benefit_all(y, t)})
        rows.append({"cohort": c, "model": "treat none", "threshold": t, "net_benefit": 0.0})
        for m in UTIL_MODELS:
            rows.append({"cohort": c, "model": m, "threshold": t,
                         "net_benefit": float(np.mean([MT.net_benefit(y, P[c][(m, r, s)]["none"], t) for r, s in MAINRUNS]))})
    _pl[c] = {"y": y, "vecs": {m: [P[c][(m, r, s)]["none"] for r, s in MAINRUNS] for m in UTIL_MODELS}, "stats": [f"nb@{t}" for t in pts]}
    for m in UTIL_MODELS:
        for which, tgt in (("fixed specificity 0.90", 0), ("fixed sensitivity 0.80", 1)):
            vals = [MT.threshold_metrics(y, P[c][(m, r, s)]["none"], THR[c][(m, r, s)][tgt]) for r, s in MAINRUNS]
            rows_thr.append({"cohort": c, "model": m, "operating_point": which, **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}})
BOOT_NB = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
for c in COHORTS:
    pts = CFG.nb_points_a if c == "NHANES" else CFG.nb_points_b
    for m in UTIL_MODELS:
        for t in pts:
            k = (m, f"nb@{t}")
            d = BOOT_NB[c]["draws"][k] - BOOT_NB[c]["draws"][("M2", f"nb@{t}")]
            rows_pts.append({"cohort": c, "model": m, "threshold": t, "net_benefit": BOOT_NB[c]["point"][k], "lo": AN.ci(BOOT_NB[c]["draws"][k])[0],
                             "hi": AN.ci(BOOT_NB[c]["draws"][k])[1], "treat_all": MT.net_benefit_all(Y[c], t),
                             "minus_M2": BOOT_NB[c]["point"][k] - BOOT_NB[c]["point"][("M2", f"nb@{t}")], "minus_M2_lo": AN.ci(d)[0], "minus_M2_hi": AN.ci(d)[1]})
DCA, NBPTS, THRMET = pd.DataFrame(rows), pd.DataFrame(rows_pts), pd.DataFrame(rows_thr)
DCA.to_csv(OUT / "tables" / "table15_decision_curves.csv", index=False)
NBPTS.to_csv(OUT / "tables" / "table16_net_benefit_points.csv", index=False)
THRMET.to_csv(OUT / "tables" / "table17_threshold_metrics.csv", index=False)
print(NBPTS.round(5).to_string(index=False)); print(THRMET.round(3).to_string(index=False))
print(f"CELL 29 OK | decision curves and threshold metrics | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 30 | censoring-aware evaluation in NAGALA and RICH (P2): Harrell's C, Uno's C at 3 years, IPCW cumulative/dynamic AUROC at
# 3 years (NAGALA also 5 years); and the 3-year IPCW-trained models (E): IPCW Brier, scaled Brier, observed and predicted 3-year risk
_t0 = time.perf_counter()
TTE_MODELS = list(C.MAIN_MODELS)
rows, rows_ip, _jobs, _pl = [], [], [], {}
for c in ("NAGALA", "RICH"):
    t_, e_ = EVAL[c].follow_years.to_numpy(), Y[c].astype(int)
    ut, G = MT.km_censoring(t_, e_)
    uw = np.where(e_ == 1, 1 / np.maximum(MT.G_at(ut, G, t_), 1e-12) ** 2, 0.0)
    for m in TTE_MODELS:
        for r, s in MAINRUNS:
            _jobs.append((c, m, r, "harrell", P[c][(m, r, s)]["none"].astype(float), t_, e_, None, None))
            _jobs.append((c, m, r, "uno3", P[c][(m, r, s)]["none"].astype(float), t_, e_, uw, CFG.horizon))
    for tau in ((CFG.horizon, CFG.nagala_horizon2) if c == "NAGALA" else (CFG.horizon,)):
        lab, iw = MT.td_auc_weights(t_, e_, tau, ut, G)
        keep = iw > 0
        _pl[f"{c}|{tau:g}"] = {"y": lab[keep], "base_w": iw[keep], "stats": ["auroc"],
                               "vecs": {m: [P[c][(m, r, s)]["none"][keep] for r, s in MAINRUNS] for m in TTE_MODELS}}
    for m, d in PIPCW[c].items():
        p3 = d["none"].astype(float)
        bs = MT.ipcw_brier(p3, t_, e_, CFG.horizon, ut, G)
        km3 = MT.km_risk(t_, e_, CFG.horizon)
        bs0 = MT.ipcw_brier(np.full(len(p3), km3), t_, e_, CFG.horizon, ut, G)
        rows_ip.append({"cohort": c, "model": m, "ipcw_brier_3y": bs, "scaled_brier_3y": 1 - bs / bs0, "td_auroc_3y": MT.td_auc(p3, t_, e_, CFG.horizon, ut, G),
                        "km_observed_risk_3y": km3, "mean_predicted_risk_3y": float(p3.mean())})

_conc_out = AN.parallel_map(AN.conc_job, _jobs, N_WORKERS)
BOOT_TTE = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
_cd = pd.DataFrame(_conc_out, columns=["cohort", "model", "run", "kind", "value"]).groupby(["cohort", "model", "kind"]).value.mean().unstack()
for c in ("NAGALA", "RICH"):
    for m in TTE_MODELS:
        row = {"cohort": c, "model": m, "harrell_c": _cd.loc[(c, m), "harrell"], "uno_c_3y": _cd.loc[(c, m), "uno3"]}
        for tau in ((CFG.horizon, CFG.nagala_horizon2) if c == "NAGALA" else (CFG.horizon,)):
            k = f"{c}|{tau:g}"
            row[f"td_auroc_{tau:g}y"] = BOOT_TTE[k]["point"][(m, "auroc")]
            row[f"td_auroc_{tau:g}y_lo"], row[f"td_auroc_{tau:g}y_hi"] = AN.ci(BOOT_TTE[k]["draws"][(m, "auroc")])
        rows.append(row)
TTE, IPCWM = pd.DataFrame(rows), pd.DataFrame(rows_ip)
TTE.to_csv(OUT / "tables" / "table18_censoring_aware.csv", index=False)
IPCWM.to_csv(OUT / "tables" / "table19_ipcw_3y_models.csv", index=False)
print(TTE.round(4).to_string(index=False)); print(IPCWM.round(4).to_string(index=False))
print(f"CELL 30 OK | censoring-aware metrics | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 31 | fairness (E, with H5 in cell 25): by sex and age band (and RICH site, NHANES ethnic group, descriptive): AUROC,
# sensitivity at FPR 0.10, true- and false-positive rates at the common threshold, calibration-in-the-large; gap intervals
_t0 = time.perf_counter()
FAIR_MODELS = ("M1", "M2", "CKINN", "LR", "XGB_mono")
rows, gap_rows, _pl = [], [], {}
for c in COHORTS:
    fr, y = EVAL[c], Y[c]
    gtypes = {"sex": fr.sex.to_numpy(), "age_band": fr.age_band.to_numpy()}
    if c == "RICH":
        gtypes["site (descriptive)"] = fr.site.to_numpy()
    if c == "NHANES":
        gtypes["ethnic group (descriptive)"] = fr.group_extra.to_numpy()
    for gt, gv in gtypes.items():
        est = [g for g in pd.unique(gv) if y[gv == g].sum() >= CFG.min_group_events and (y[gv == g] == 0).sum() >= CFG.min_group_events]
        for g in sorted(pd.unique(gv), key=str):
            ix = np.flatnonzero(gv == g)
            for m in FAIR_MODELS:
                ok = g in est
                vals = []
                for r, s in MAINRUNS:
                    p, thr = P[c][(m, r, s)]["none"][ix].astype(float), THR[c][(m, r, s)][0][ix]
                    h = p >= thr
                    vals.append({"auroc": MT.auroc_w(MT.Ranked(p, y[ix])) if ok else np.nan,
                                 "tpr_at_fpr_0.10": MT.tpr_at_fpr_w(MT.Ranked(p, y[ix]), CFG.fpr_matched) if ok else np.nan,
                                 "tpr_common_thr": (h & (y[ix] == 1)).sum() / max((y[ix] == 1).sum(), 1) if ok else np.nan,
                                 "fpr_common_thr": (h & (y[ix] == 0)).sum() / max((y[ix] == 0).sum(), 1),
                                 "citl": MT.citl_w(y[ix], p) if ok else np.nan, "slope": MT.calib_w(y[ix], p)[1] if ok else np.nan,
                                 "mean_pred": p.mean()})
                rows.append({"cohort": c, "group_type": gt, "group": str(g), "model": m, "n": len(ix), "events": int(y[ix].sum()),
                             "estimable": ok, **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}, "observed_rate": float(y[ix].mean())})
        if "descriptive" not in gt:
            key = f"{c}|{gt}"
            _pl[key] = {"y": y, "vecs": {m: [P[c][(m, r, s)]["none"] for r, s in MAINRUNS] for m in FAIR_MODELS},
                        "thrs": {m: [THR[c][(m, r, s)][0] for r, s in MAINRUNS] for m in FAIR_MODELS},
                        "groups": {"fair": [np.flatnonzero(gv == g) for g in sorted(est, key=str)]}, "fpr": CFG.fpr_matched,
                        "stats": ["gap_auroc", "gap_tpr", "gap_tprthr", "gap_fprthr", "gap_citl"]}
BOOT_FAIR = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
for key, b in BOOT_FAIR.items():
    c, gt = key.split("|")
    for m in FAIR_MODELS:
        for st in ("gap_auroc", "gap_tpr", "gap_tprthr", "gap_fprthr", "gap_citl"):
            d = b["draws"][(m, st)] - b["draws"][("M2", st)]
            gap_rows.append({"cohort": c, "group_type": gt, "model": m, "gap": st[4:], "value": b["point"][(m, st)], "lo": AN.ci(b["draws"][(m, st)])[0],
                             "hi": AN.ci(b["draws"][(m, st)])[1], "minus_M2": b["point"][(m, st)] - b["point"][("M2", st)],
                             "minus_M2_lo": AN.ci(d)[0], "minus_M2_hi": AN.ci(d)[1]})
FAIR, FAIRGAP = pd.DataFrame(rows), pd.DataFrame(gap_rows)
FAIR.to_csv(OUT / "tables" / "table20_subgroups.csv", index=False)
FAIRGAP.to_csv(OUT / "tables" / "table21_subgroup_gaps.csv", index=False)
print(FAIRGAP[FAIRGAP.model.isin(["M2", "CKINN"])].round(4).to_string(index=False))
print(f"CELL 31 OK | subgroup tables | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 32 | missingness (P2): test-time removal of the laboratory panel (MCAR, MAR, MNAR at 10, 30, 50%), AUROC loss at MCAR 30% for
# CKINN against each comparator; RICH structural missingness patterns; RICH leave-one-site-out
_t0 = time.perf_counter()
SCEN = [scen_tag(m, q) for m, q in scenarios(CFG)]
PRIM = scen_tag(*CFG.primary_miss)
rows, _pl = [], {}
for c in COHORTS:
    y = Y[c]
    for m in list(C.MAIN_MODELS) + list(C.RUN0_EXTRA):
        runs = MAINRUNS if m in C.MAIN_MODELS else [(0, 0)]
        row = {"cohort": c, "model": m, "runs": len(runs)}
        for tag in ["none"] + SCEN:
            row[tag] = float(np.mean([MT.auroc_w(MT.Ranked(P[c][(m, r, s)][tag], y)) for r, s in runs]))
        rows.append(row)
    v = {m: [P[c][(m, r, s)]["none"] for r, s in MAINRUNS] for m in C.MAIN_MODELS}
    v.update({f"{m}@{PRIM}": [P[c][(m, r, s)][PRIM] for r, s in MAINRUNS] for m in C.MAIN_MODELS})
    for m in list(C.RUN0_EXTRA) + ["CKINN"]:
        v[f"{m}_r0"], v[f"{m}_r0@{PRIM}"] = [P[c][(m, 0, 0)]["none"]], [P[c][(m, 0, 0)][PRIM]]
    _pl[c] = {"y": y, "vecs": v, "stats": ["auroc"]}
MISS_AUC = pd.DataFrame(rows)
BOOT_MISS = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
rows = []
for c in COHORTS:
    pt, dr = BOOT_MISS[c]["point"], BOOT_MISS[c]["draws"]
    for m in [x for x in C.MAIN_MODELS if x != "CKINN"] + [f"{x}_r0" for x in C.RUN0_EXTRA]:
        ck = "CKINN_r0" if m.endswith("_r0") else "CKINN"
        lk = lambda name, D_: D_[(name, "auroc")] - D_[(f"{name}@{PRIM}", "auroc")]
        d = lk(ck, dr) - lk(m, dr)
        rows.append({"cohort": c, "scenario": PRIM, "comparator": m.replace("_r0", ""), "runs": 1 if m.endswith("_r0") else CFG.n_repeats,
                     "loss_ckinn": lk(ck, pt), "loss_comparator": lk(m, pt), "ckinn_minus_comparator": lk(ck, pt) - lk(m, pt),
                     "lo": AN.ci(d)[0], "hi": AN.ci(d)[1], "comparator_trained_with_matching_augmentation": m.replace("_r0", "") in ("M2", "LR_aug", "XGB_aug")})
MISS_TEST = pd.DataFrame(rows)
# structural missingness in RICH: performance by observed pattern of (HDL, LDL, AST, smoking) missingness
fr = EVAL["RICH"]
pat = (fr.hdl_mmol.isna().astype(int).astype(str) + fr.ldl_mmol.isna().astype(int).astype(str) + fr.ast_u_l.isna().astype(int).astype(str)
       + fr.smoker.isna().astype(int).astype(str)).to_numpy()
rows = []
for pv in sorted(pd.unique(pat)):
    ix = np.flatnonzero(pat == pv)
    if Y["RICH"][ix].sum() < CFG.min_group_events:
        continue
    for m in ("LR", "XGB", "XGB_mono", "MI_LR", "M1", "M2", "CKINN"):
        runs = MAINRUNS if m in C.MAIN_MODELS else [(0, 0)]
        rows.append({"pattern_missing_hdl_ldl_ast_smoking": pv, "n": len(ix), "events": int(Y["RICH"][ix].sum()), "model": m,
                     "auroc": float(np.mean([MT.auroc_w(MT.Ranked(P["RICH"][(m, r, s)]["none"][ix], Y["RICH"][ix])) for r, s in runs]))})
STRUCT = pd.DataFrame(rows)
# leave-one-site-out
rows = []
for s_ in SITES_HELD:
    ix = np.flatnonzero(EVAL["RICH"].site.to_numpy() == s_)
    for m in SITEP:
        p = SITEP[m][ix]
        rows.append({"held_out_site": s_, "model": m, "n": len(ix), "events": int(Y["RICH"][ix].sum()), "auroc": MT.auroc_w(MT.Ranked(p, Y["RICH"][ix])),
                     "brier": MT.brier_w(Y["RICH"][ix], p), "citl": MT.citl_w(Y["RICH"][ix], p)})
SITE_TBL = pd.DataFrame(rows)
_ys = Y["RICH"][SITE_ROWS]
_bs = AN.bootstrap({"loso": {"y": _ys, "vecs": {m: [SITEP[m][SITE_ROWS]] for m in SITEP}, "stats": ["auroc", "brier"]}}, CFG.n_boot_desc, N_WORKERS)["loso"]
SITE_POOL = pd.DataFrame([{"model": m, "auroc_pooled_held_out": _bs["point"][(m, "auroc")],
                           "ckinn_minus_model": _bs["point"][("CKINN", "auroc")] - _bs["point"][(m, "auroc")],
                           "lo": AN.ci(_bs["draws"][("CKINN", "auroc")] - _bs["draws"][(m, "auroc")])[0],
                           "hi": AN.ci(_bs["draws"][("CKINN", "auroc")] - _bs["draws"][(m, "auroc")])[1]} for m in SITEP])
for name, df in (("table22_missingness_auroc", MISS_AUC), ("table23_missingness_loss_tests", MISS_TEST), ("table24_rich_structural_missingness", STRUCT),
                 ("table25_rich_site_held_out", SITE_TBL), ("table26_rich_site_pooled", SITE_POOL)):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
print(MISS_TEST.round(4).to_string(index=False)); print(SITE_POOL.round(4).to_string(index=False))
print(f"CELL 32 OK | missingness and site tables | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 33 | constraint evidence (E; manipulation check P1): in-distribution violation rates of M1, M2, CKINN by seed; leave-one-out,
# wrong-sign, alternative tables and lambda path; empirical signs against held-out partial dependence; sparse regions; other cohort
_t0 = time.perf_counter()
rows = []
for c in COHORTS:
    for m in NN:
        for s in range(CFG.n_seeds):
            vv, vo, ic, inn = 0.0, 0.0, 0.0, 0.0
            for k in range(CFG.n_splits):
                r_ = EN.load(CACHE, PL.tid("cv", c, "T2", "main" if s == 0 else "seed", "r0", f"k{k}", m, f"s{s}"))
                vv, vo = vv + r_["viol"]["viol"].sum(), vo + r_["viol"]["vobs"].sum()
                ic, inn = ic + r_["viol"]["ig_cons"].sum(), inn + r_["viol"]["ig_n"].sum()
            rows.append({"cohort": c, "model": m, "seed": s, "violation_rate": vv / max(vo, 1), "ig_direction_consistency": ic / max(inn, 1)})
MANIP = pd.DataFrame(rows).groupby(["cohort", "model"]).agg(violation_rate=("violation_rate", "mean"), violation_min=("violation_rate", "min"),
                                                             violation_max=("violation_rate", "max"),
                                                             ig_direction_consistency=("ig_direction_consistency", "mean")).reset_index()
# variants
rows, _pl = [], {}
for c in COHORTS:
    base = P[c][("CKINN", 0, 0)]["none"]
    _pl[c] = {"y": Y[c], "vecs": {"CKINN": [base], **{v: [PV[c][v]["none"]] for v in PV[c]}}, "stats": ["auroc", "brier"]}
BOOT_VAR = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
for c in COHORTS:
    pt, dr = BOOT_VAR[c]["point"], BOOT_VAR[c]["draws"]
    for v in PV[c]:
        vb = vo = vt = vto = vd = vdo = 0.0
        for k in range(CFG.n_splits):
            r_ = EN.load(CACHE, PL.tid("var", c, "T2", v, "r0", f"k{k}", "CKINN", "s0"))
            vb, vo = vb + r_["viol"]["viol"].sum(), vo + r_["viol"]["vobs"].sum()
            vt, vto = vt + r_["extra_viol"]["own_table"]["viol"].sum(), vto + r_["extra_viol"]["own_table"]["vobs"].sum()
            if "dropped" in r_["extra_viol"]:
                vd, vdo = vd + r_["extra_viol"]["dropped"]["viol"].sum(), vdo + r_["extra_viol"]["dropped"]["vobs"].sum()
        da, db = dr[(v, "auroc")] - dr[("CKINN", "auroc")], dr[(v, "brier")] - dr[("CKINN", "brier")]
        rows.append({"cohort": c, "variant": v, "auroc": pt[(v, "auroc")], "auroc_minus_ckinn": pt[(v, "auroc")] - pt[("CKINN", "auroc")],
                     "lo": AN.ci(da)[0], "hi": AN.ci(da)[1], "brier_minus_ckinn": pt[(v, "brier")] - pt[("CKINN", "brier")],
                     "brier_lo": AN.ci(db)[0], "brier_hi": AN.ci(db)[1], "violation_rate_v1_table": vb / max(vo, 1),
                     "violation_rate_own_table": vt / max(vto, 1), "violation_rate_dropped_feature": (vd / vdo) if vdo else np.nan})
VARIANTS = pd.DataFrame(rows)
# empirical signs (descriptive) against partial dependence of each model on held-out rows of repeat 0
rows = []
for c in COHORTS:
    feats = FEATS[(c, "T2")]
    sg = PL.signs_for("T2", feats)
    emp = AN.empirical_signs(EVAL[c], feats, sg)
    for m in ("LR", "GAM", "XGB", "XGB_mono", "M1", "M2", "CKINN"):
        slopes = {}
        for k in range(CFG.n_splits):
            r_ = EN.load(CACHE, PL.tid("cv", c, "T2", "main", "r0", f"k{k}", m, "s0"))
            for f, d in r_.get("pd", {}).items():
                slopes.setdefault(f, []).append(float(d["pd"][-1] - d["pd"][0]))
        for f, sl in slopes.items():
            e = emp[emp.feature == f].iloc[0]
            rows.append({"cohort": c, "model": m, "feature": f, "prior_sign": sg[f], "pd_change_p10_to_p90": float(np.mean(sl)),
                         "pd_sign_agrees_with_prior": bool(np.sign(np.mean(sl)) == sg[f]), "folds_agreeing": int(sum(np.sign(x) == sg[f] for x in sl)),
                         "adjusted_evidence": e.adjusted_evidence, "adjusted_coef": e.adjusted_coef, "coef_lo": e.coef_lo, "coef_hi": e.coef_hi,
                         "quintile_agrees": e.quintile_agrees})
EMPSIGN = pd.DataFrame(rows)
# sparse regions (held-out rows beyond the training 95th percentile of Mahalanobis distance) and the other cohort
rows = []
for c in COHORTS:
    for m in NN:
        sv = so = dv = do = npool = nsp = 0.0
        for k in range(CFG.n_splits):
            r_ = EN.load(CACHE, PL.tid("cv", c, "T2", "main", "r0", f"k{k}", m, "s0"))
            sv, so = sv + r_["sparse"]["viol"].sum(), so + r_["sparse"]["vobs"].sum()
            dv, do = dv + r_["viol"]["viol"].sum(), do + r_["viol"]["vobs"].sum()
            npool, nsp = npool + r_["sparse"]["n_pool"], nsp + r_["sparse"]["n_sparse"]
        rows.append({"cohort": c, "model": m, "region": "held-out sparse rows", "violation_rate": sv / max(so, 1), "share_sparse": nsp / max(npool, 1)})
        rows.append({"cohort": c, "model": m, "region": "held-out random rows", "violation_rate": dv / max(do, 1), "share_sparse": np.nan})
for s_ in [s for s in TR_SPECS if s["id"].startswith("trext") and s["_model"] in NN]:
    r_ = EN.load(CACHE, s_["id"])
    for tag in ("src", "tgt"):
        ev_ = r_["extra_viol"][tag]
        rows.append({"cohort": f"{s_['_src']}-trained", "model": s_["_model"], "region": f"{'source' if tag == 'src' else 'other cohort'} ({s_['_src'] if tag == 'src' else s_['_tgt']})",
                     "violation_rate": ev_["viol"].sum() / max(ev_["vobs"].sum(), 1), "share_sparse": np.nan, "seed": s_["_s"]})
SPARSE = pd.DataFrame(rows).groupby(["cohort", "model", "region"], dropna=False).agg(violation_rate=("violation_rate", "mean"),
                                                                                       share_sparse=("share_sparse", "mean")).reset_index()
for name, df in (("table27_manipulation_check", MANIP), ("table28_constraint_variants", VARIANTS), ("table29_empirical_signs_vs_pd", EMPSIGN),
                 ("table30_violation_sparse_and_other_cohort", SPARSE)):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
print(MANIP.round(4).to_string(index=False)); print(VARIANTS.round(4).to_string(index=False)); print(SPARSE.round(4).to_string(index=False))
print(f"CELL 33 OK | constraint evidence tables | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 34 | transport detail (P2) and secondary transport analyses (E): seed-averaged internal reference and external metrics;
# recalibration in the small with 1,000 target participants (target labels used); label-free reweighting; pooled training
_t0 = time.perf_counter()
rows = []
for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
    b = BOOT_FAM[f"{src}->{tgt}"]
    y = Y[tgt]
    for m in NN + ["LR", "XGB", "XGB_mono"]:
        seeds = range(CFG.n_seeds) if m in NN else [0]
        ext = [TRX[(src, tgt, m, s)] for s in seeds]
        rows.append({"source": src, "target": tgt, "model": m, "seeds": len(ext), "auroc_target_cv": b["point"][(f"ref_{m}", "auroc")],
                     "auroc_external": b["point"][(f"ext_{m}", "auroc")], "transport_gap": b["point"][(f"ref_{m}", "auroc")] - b["point"][(f"ext_{m}", "auroc")],
                     "gap_lo": AN.ci(b["draws"][(f"ref_{m}", "auroc")] - b["draws"][(f"ext_{m}", "auroc")])[0],
                     "gap_hi": AN.ci(b["draws"][(f"ref_{m}", "auroc")] - b["draws"][(f"ext_{m}", "auroc")])[1],
                     "abs_citl_external": b["point"][(f"ext_{m}", "citl_abs")], "slope_external": b["point"][(f"ext_{m}", "slope")],
                     "brier_external": b["point"][(f"ext_{m}", "brier")]})
TRANS = pd.DataFrame(rows)
rows = []
for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
    y = Y[tgt]
    rng = np.random.default_rng(C.seed_for("recal", tgt))
    cal_ix = np.sort(rng.choice(len(y), CFG.recal_n, replace=False))
    rest = np.setdiff1d(np.arange(len(y)), cal_ix)
    for m in NN + ["LR", "XGB_mono"]:
        p = TRX[(src, tgt, m, 0)]
        a, b_ = MT.calib_w(y[cal_ix], p[cal_ix])
        prc = 1 / (1 + np.exp(-(a + b_ * MT.logit(MT.clip(p)))))
        row = {"source": src, "target": tgt, "model": m, "events_in_recalibration_sample": int(y[cal_ix].sum()),
               "brier_before": MT.brier_w(y[rest], p[rest]), "brier_after": MT.brier_w(y[rest], prc[rest]),
               "citl_before": MT.citl_w(y[rest], p[rest]), "citl_after": MT.citl_w(y[rest], prc[rest]),
               "slope_before": MT.calib_w(y[rest], p[rest])[1], "slope_after": MT.calib_w(y[rest], prc[rest])[1]}
        if (src, tgt, m) in TRW:
            pw = TRW[(src, tgt, m)]
            row.update({"auroc_unweighted_s0": MT.auroc_w(MT.Ranked(p, y)), "auroc_reweighted": MT.auroc_w(MT.Ranked(pw, y)),
                        "abs_citl_unweighted_s0": abs(MT.citl_w(y, p)), "abs_citl_reweighted": abs(MT.citl_w(y, pw))})
        rows.append(row)
TRANS2 = pd.DataFrame(rows)
_pl = {c: {"y": Y[c], "vecs": {"M2": [P[c][("M2", 0, 0)]["none"]], "CKINN": [P[c][("CKINN", 0, 0)]["none"]],
                               "M2_pooled": [POOLP[("M2", c)]], "CKINN_pooled": [POOLP[("CKINN", c)]]}, "stats": ["auroc", "brier"]}
       for c in ("NAGALA", "RICH")}
BOOT_POOL = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
rows = []
for c in ("NAGALA", "RICH"):
    pt, dr = BOOT_POOL[c]["point"], BOOT_POOL[c]["draws"]
    for m in ("M2", "CKINN", "M2_pooled", "CKINN_pooled"):
        d = dr[(m, "auroc")] - dr[("M2", "auroc")]
        rows.append({"cohort": c, "variant": m, "auroc": pt[(m, "auroc")], "auroc_minus_M2": pt[(m, "auroc")] - pt[("M2", "auroc")],
                     "lo": AN.ci(d)[0], "hi": AN.ci(d)[1], "brier": pt[(m, "brier")]})
POOLTBL = pd.DataFrame(rows)
for name, df in (("table31_transport", TRANS), ("table32_transport_recalibration_reweighting", TRANS2), ("table33_pooled_training", POOLTBL)):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
print(TRANS.round(4).to_string(index=False)); print(TRANS2.round(4).to_string(index=False)); print(POOLTBL.round(4).to_string(index=False))
print(f"CELL 34 OK | transport tables | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 35 | sensitivity analyses (E): tier T3, RICH without family history, NHANES with the v1 label, NHANES survey-weighted metrics
_t0 = time.perf_counter()
rows, _pl = [], {}
for c in ("NAGALA", "RICH"):
    _pl[f"T3|{c}"] = {"y": Y[c], "vecs": {m: [PT3[c][(m, 0, 0)]["none"]] for m in _t3m}, "stats": ["auroc", "brier"]}
_pl["noFH|RICH"] = {"y": Y["RICH"], "vecs": {m: [PNOFH[(m, 0, 0)]["none"]] for m in ("LR", "M2", "CKINN")}, "stats": ["auroc", "brier"]}
_pl["v1label|NHANES"] = {"y": Y["NHANESv1"], "vecs": {m: [PNHV1[(m, 0, 0)]["none"]] for m in ("LR", "M2", "CKINN")}, "stats": ["auroc", "brier"]}
_sv = EVAL["NHANES"]
_wk = _sv.svy_w.to_numpy() > 0
_pl["survey|NHANES"] = {"y": Y["NHANES"][_wk], "vecs": {m: [P["NHANES"][(m, r, s)]["none"][_wk] for r, s in MAINRUNS] for m in FAIR_MODELS},
                        "stats": ["auroc", "brier", "citl"], "mode": "survey", "base_w": _sv.svy_w.to_numpy()[_wk],
                        "strata": _sv.svy_strata.to_numpy()[_wk], "psu": _sv.svy_psu.to_numpy()[_wk]}
BOOT_SENS = AN.bootstrap(_pl, CFG.n_boot_desc, N_WORKERS)
for key, b in BOOT_SENS.items():
    an, c = key.split("|")
    ms = sorted({m for m, _ in b["point"]})
    for m in ms:
        row = {"analysis": an, "cohort": c, "model": m, "auroc": b["point"][(m, "auroc")], "brier": b["point"][(m, "brier")]}
        for F in ("M2", "M1"):
            if F in ms and m == "CKINN":
                d = b["draws"][("CKINN", "auroc")] - b["draws"][(F, "auroc")]
                row[f"ckinn_minus_{F}"] = b["point"][("CKINN", "auroc")] - b["point"][(F, "auroc")]
                row[f"ckinn_minus_{F}_lo"], row[f"ckinn_minus_{F}_hi"] = AN.ci(d)
        if (m, "citl") in b["point"]:
            row["citl"] = b["point"][(m, "citl")]
        rows.append(row)
SENS = pd.DataFrame(rows)
SENS.to_csv(OUT / "tables" / "table34_sensitivity.csv", index=False)
print(SENS.round(4).to_string(index=False))
print(f"CELL 35 OK | sensitivity analyses | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 36 | figures: vector PDF and 300 dpi PNG. Colours follow the model, never its rank: M2 blue, CKINN orange, M1 aqua, LR violet;
# every series also has its own marker and line style, so identity never rests on colour alone
_t0 = time.perf_counter()
FIG = OUT / "figures"
COL = {"M2": "#2a78d6", "CKINN": "#eb6834", "M1": "#1baf7a", "LR": "#6250d6", "XGB_mono": "#52514e"}
MRK = {"M2": "o", "CKINN": "s", "M1": "^", "LR": "D", "XGB_mono": "v"}
LST = {"M2": "-", "CKINN": "-", "M1": "--", "LR": ":", "XGB_mono": "-."}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "lines.linewidth": 1.5, "axes.axisbelow": True})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


SHORT = {"AUROC (non-inferiority, margin 0.01)": "AUROC, non-inferiority", "Brier score": "Brier", "|calibration slope - 1|": "|slope - 1|",
         "across-seed attribution agreement (Spearman)": "attribution agreement across seeds", "sex gap in sensitivity at FPR 0.10": "sex gap, TPR at FPR 0.10",
         "age-band range in sensitivity at FPR 0.10": "age-band range, TPR at FPR 0.10", "transport gap (target CV AUROC - external AUROC)": "transport gap",
         "|calibration-in-the-large| in the target": "|CITL| in target"}
# Figure 1: hypothesis endpoints, standardised difference (difference / bootstrap SD, oriented so that right = the predicted direction)
_pr = EP[EP.primary].copy()
_pr["z"] = np.where(_pr.predicted == "higher", 1, -1) * (_pr.difference + np.where(_pr.hypothesis == "H1", _pr.margin, 0)) / _pr.boot_sd
fig, axs = plt.subplots(1, 2, figsize=(9, 0.26 * _pr.groupby("family").size().max() + 1.2), sharey=True)
for ax, (fam, g) in zip(axs, _pr.groupby("family")):
    g = g.reset_index(drop=True)
    ax.barh(range(len(g)), g.z, color=[COL["CKINN"] if st == "predicted" else "#9b9a95" if st == "null" else "#e34948" for st in g.status], height=0.6)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    for xv in (-1.96, 1.96):
        ax.axvline(xv, color="#9b9a95", ls=":", lw=0.8)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{h} {u.replace(' (T2)', '')}: {SHORT.get(e, e)}" for h, u, e in zip(g.hypothesis, g.unit, g.endpoint)], fontsize=6)
    ax.set_title(fam, fontsize=8)
    ax.set_xlabel("standardised difference (right = predicted direction)")
axs[0].invert_yaxis()
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=COL["CKINN"], label="predicted (Holm)"), Patch(color="#9b9a95", label="null"),
                       Patch(color="#e34948", label="opposite (Holm)")], fontsize=6, loc="lower right")
savefig(fig, "fig1_hypothesis_endpoints")
# Figure 2: reliability curves with bands
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, c in zip(axs, COHORTS):
    sub = CALCURVE[CALCURVE.cohort == c]
    for m in ("M1", "M2", "CKINN"):
        g = sub[sub.model == m]
        ax.fill_between(g.mean_predicted, g.obs_lo, g.obs_hi, color=COL[m], alpha=0.15, lw=0)
        ax.plot(g.mean_predicted, g.observed, LST[m], marker=MRK[m], ms=4, color=COL[m], label=m)
    lim = float(max(sub.mean_predicted.max(), sub.observed.max()))
    ax.plot([0, lim], [0, lim], color="#9b9a95", lw=0.8)
    ax.set_title(f"{c}: calibration (run 0)")
    ax.set_xlabel("mean predicted risk")
    ax.set_ylabel("observed proportion")
    ax.legend(fontsize=7)
savefig(fig, "fig2_calibration")
# Figure 3: decision curves
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, c in zip(axs, COHORTS):
    sub = DCA[DCA.cohort == c]
    for m in ("M1", "M2", "CKINN"):
        g = sub[sub.model == m]
        ax.plot(g.threshold, g.net_benefit, LST[m], color=COL[m], label=m)
    g = sub[sub.model == "treat all"]
    ax.plot(g.threshold, g.net_benefit, color="#9b9a95", lw=1, label="treat all")
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    top = float(sub[sub.model != "treat all"].net_benefit.max())
    ax.set_ylim(-0.25 * top, 1.1 * top)
    ax.set_title(f"{c}: decision curve")
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.legend(fontsize=7)
savefig(fig, "fig3_decision_curves")
# Figure 4: AUROC as the laboratory panel is removed completely at random
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
_mc = [t for t in SCEN if t.startswith("mcar")]
_x = [0.0] + [float(t[4:]) / 100 for t in _mc]
for ax, c in zip(axs, COHORTS):
    sub = MISS_AUC[MISS_AUC.cohort == c].set_index("model")
    for m in ("M1", "M2", "CKINN", "LR"):
        ax.plot(_x, [sub.loc[m, "none"]] + [sub.loc[m, t] for t in _mc], LST[m], marker=MRK[m], ms=4, color=COL[m], label=m)
    ax.set_title(f"{c}: AUROC, panel removed (MCAR)")
    ax.set_xlabel("share of people without the laboratory panel")
    ax.set_ylabel("AUROC (run-averaged)")
    ax.legend(fontsize=7)
savefig(fig, "fig4_missingness")
# Figure 5: CKINN minus M2 AUROC across seeds of repeat 0 and across the main runs (variance components)
fig, ax = plt.subplots(figsize=(6, 3))
for i, c in enumerate(COHORTS):
    y = Y[c]
    a = lambda p: MT.auroc_w(MT.Ranked(p, y))
    sd = [a(P[c][("CKINN", 0, s)]["none"]) - a(P[c][("M2", 0, s)]["none"]) for _, s in SEEDRUNS]
    rd = [a(P[c][("CKINN", r, s)]["none"]) - a(P[c][("M2", r, s)]["none"]) for r, s in MAINRUNS]
    ax.scatter(np.full(len(sd), i - 0.12), sd, marker="o", s=18, color=COL["M2"], label="seeds of repeat 0" if i == 0 else None, zorder=3)
    ax.scatter(np.full(len(rd), i + 0.12), rd, marker="s", s=18, color=COL["CKINN"], label="main runs (repeat r, seed r)" if i == 0 else None, zorder=3)
    lo, hi = AN.ci(BOOT_FAM[c]["draws"][("CKINN", "auroc")] - BOOT_FAM[c]["draws"][("M2", "auroc")])
    ax.plot([i + 0.3, i + 0.3], [lo, hi], color="#0b0b0b", lw=1.5)
    ax.plot([i + 0.3], [BOOT_FAM[c]["point"][("CKINN", "auroc")] - BOOT_FAM[c]["point"][("M2", "auroc")]], marker="D", ms=4, color="#0b0b0b",
            label="run average, 95% bootstrap interval" if i == 0 else None)
ax.axhline(0, color="#9b9a95", lw=0.8)
ax.axhline(-CFG.ni_margin, color="#9b9a95", lw=0.8, ls=":")
ax.set_xticks(range(len(COHORTS)))
ax.set_xticklabels(COHORTS)
ax.set_ylabel("AUROC, CKINN minus M2")
ax.legend(fontsize=6, loc="best")
savefig(fig, "fig5_seed_and_split_variation")
# Figure 6: constraint variants, AUROC difference from CKINN (repeat 0)
fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharex=True)
for ax, c in zip(axs, COHORTS):
    g = VARIANTS[VARIANTS.cohort == c].reset_index(drop=True)
    ax.errorbar(g.auroc_minus_ckinn, range(len(g)), xerr=[g.auroc_minus_ckinn - g.lo, g.hi - g.auroc_minus_ckinn], fmt="o", ms=3,
                color=COL["CKINN"], ecolor="#52514e", elinewidth=0.8)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.variant, fontsize=6)
    ax.invert_yaxis()
    ax.set_title(f"{c}: variant minus CKINN (AUROC)")
savefig(fig, "fig6_constraint_variants")
assert all((FIG / f"{n}.{e}").exists() for n in ("fig1_hypothesis_endpoints", "fig2_calibration", "fig3_decision_curves", "fig4_missingness",
                                                  "fig5_seed_and_split_variation", "fig6_constraint_variants") for e in ("pdf", "png"))
print(f"CELL 36 OK | six figures saved as PDF and 300 dpi PNG | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 37 | LaTeX (booktabs) exports of the main tables; every value formatted from the stored CSV tables
_t0 = time.perf_counter()


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, digits: int = 3) -> None:
    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace(">", r"$>$").replace("<", r"$<$")
    cols = [str(c).replace("_", r"\_") for c in df.columns]
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    lines += [" & ".join(f(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


_rd = lambda n: pd.read_csv(OUT / "tables" / f"{n}.csv", keep_default_na=False, na_values=[""])
_perf = _rd("table08_performance")
latex_table(_perf[_perf.model.isin(["LR", "cLR", "GAM", "XGB_mono", "LGBM_mono", "M1", "M2", "CKINN"])][
    ["cohort", "model", "events", "auroc", "auroc_lo", "auroc_hi", "brier", "scaled_brier", "slope", "citl", "ece"]],
    OUT / "latex" / "table_performance.tex", "Run-averaged out-of-fold performance at tier T2 (5 repeats of 5-fold cross-validation).", "tab:perf")
_ep = _rd("table09_hypothesis_endpoints")
latex_table(_ep[_ep.primary.astype(str) == "True"][["family", "hypothesis", "unit", "endpoint", "difference", "ci_lo", "ci_hi", "p_pred_holm", "status"]],
            OUT / "latex" / "table_endpoints.tex", "Primary endpoints of hypothesis families A (CKINN minus M2) and B (CKINN minus M1).", "tab:endpoints", 4)
latex_table(_rd("table10_verdicts"), OUT / "latex" / "table_verdicts.tex", "Pre-specified verdicts (protocol v2, section 7).", "tab:verdicts")
latex_table(_rd("table11_h1_crosschecks")[["family", "cohort", "auroc_diff_runavg", "boot_lo", "boot_hi", "delong_p", "nb_lo", "nb_hi", "p_tost_margin_0.01"]],
            OUT / "latex" / "table_h1_crosschecks.tex", "Cross-checks of the AUROC comparison.", "tab:h1x", 4)
print(f"CELL 37 OK | LaTeX tables written | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 38 | generated results report (report.md): every number is formatted by code from a stored table
_t0 = time.perf_counter()
REP = OUT / "report"
T = {p.stem: pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in sorted((OUT / "tables").glob("*.csv"))}


def _fmt(v, d=3):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return "--" if not np.isfinite(v) else f"{v:.{d}f}"
    return str(v)


def md_table(df: pd.DataFrame, d: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(_fmt(v, d) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def ci_str(est, lo, hi, d=4):
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


_ep, _vd, _perf = T["table09_hypothesis_endpoints"], T["table10_verdicts"], T["table08_performance"]
_epp = _ep[_ep.primary.astype(str) == "True"]
verdict_lines = [f"- **{r.hypothesis} {r.verdict}** (family {r.family}): {r.units_predicted} of {r.units} units in the predicted direction after Holm "
                 f"correction, {r.units_opposite} in the opposite direction." for r in _vd.itertuples()]
h1_lines = []
for r in _ep[_ep.endpoint.str.startswith("AUROC (non")].itertuples():
    h1_lines.append(f"- {r.family}, {r.unit}: AUROC difference {ci_str(r.difference, r.ci_lo, r.ci_hi)}; Holm-adjusted one-sided p for "
                    f"non-inferiority {r.p_pred_holm:.4f}; status {r.status}.")
_main = _perf[_perf.runs == CFG.n_repeats]
best_lines = [f"- {c}: highest run-averaged AUROC {g.loc[g.auroc.idxmax(), 'model']} ({g.auroc.max():.4f}); lowest Brier "
              f"{g.loc[g.brier.idxmin(), 'model']} ({g.brier.min():.5f})." for c, g in _main.groupby("cohort")]
_x = T["table11_h1_crosschecks"]
_mde = _x[(_x.cohort == "NAGALA") & (_x.family.str.startswith("A"))].iloc[0]
mde_line = (f"For NAGALA, the bootstrap standard deviation of the CKINN minus M2 AUROC difference was {_mde.boot_sd:.4f}, so the difference detectable with "
            f"80% power at two-sided 5% is {_mde.mde_80pct_power:.4f}, and the power of the non-inferiority test (margin {CFG.ni_margin}) if the true "
            f"difference were zero is {_mde.power_noninferiority_if_true_diff_0:.3f}.")
_sens_cols = ["cohort", "model", "runs", "events", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier", "scaled_brier", "slope", "citl", "ece"]
_dc0 = T["table01_data_card"].set_index("cohort")
LIMITS = [
    "Version 1 results were seen before every v2 decision; v2 is a re-analysis of the same development and evaluation rows, not a new test set. "
    "All amendments are listed in protocol_v2.md, section 2.",
    f"Task B labels are binary incident diabetes over follow-up windows that differ between cohorts (NAGALA {_dc0.loc['NAGALA', 'follow_up_min_years']:.1f} "
    f"to {_dc0.loc['NAGALA', 'follow_up_max_years']:.1f} years, RICH {_dc0.loc['RICH', 'follow_up_min_years']:.1f} to {_dc0.loc['RICH', 'follow_up_max_years']:.1f} "
    "years; AUDIT F15); transport differences mix population shift with design. The censoring-aware analyses address the horizon only in part.",
    "Cross-population evidence is from Japan and China; NHANES is a detection task, its extract is a non-random subset of the survey (AUDIT F14), and "
    "survey-weighted estimates do not represent the US population.",
    "RICH family history is probably under-recorded at several sites (AUDIT F16); RICH drinking status is unused (AUDIT F17).",
    "Missingness experiments remove values at test time; the training augmentation of M2, CKINN, LR_aug and XGB_aug matches the primary MCAR "
    "scenario, which favours those models against the other comparators (AUDIT F10).",
    "H4 is now measured by across-seed attribution stability, which CKINN is not trained for; the violation rates remain a manipulation check and "
    "are not evidence of clinical validity. Agreement with clinical priors is not proof that a model is causally correct.",
    "Subgroup gaps at one common threshold reflect differences in risk between groups; sensitivity at matched specificity isolates ranking within "
    "groups but subgroups with few events have wide intervals.",
    "Outputs are risk estimates from retrospective research data; they are not diagnoses and have not been validated for clinical use.",
]
REPORT = f"""# CKINN versus M1 and M2 on real data, version 2: results report (generated)

Generated {_dt.date.today().isoformat()} in `{CFG.mode}` mode; configuration hash `{PROTOCOL_HASH[:16]}`. Every number below is formatted by code from the
stored tables in `tables/`; none was typed. Outputs are research risk estimates, not diagnoses. Labels: P1 pre-specified in v1, P2 pre-specified
in v2 (decided after seeing v1), E exploratory.

## 1. Data
{md_table(T["table01_data_card"][["cohort", "rows", "labelled", "positives", "prevalence_pct", "predictors_present"]], 2)}

NHANES label (amendment A11):

{md_table(T["table03_nhanes_label"])}

Development and evaluation rows:

{md_table(T["table04_split"])}

## 2. Pre-specified verdicts (P2)
{chr(10).join(verdict_lines)}

{md_table(_vd)}

H1 in detail (run-averaged AUROC difference, 95% paired-bootstrap interval, {CFG.n_boot:,} draws):

{chr(10).join(h1_lines)}

All primary endpoints (difference = CKINN minus comparator):

{md_table(_epp[["family", "hypothesis", "unit", "endpoint", "difference", "ci_lo", "ci_hi", "p_pred_holm", "p_opp_holm", "status"]], 4)}

## 3. Cross-checks of H1 and sources of variance (P2)
{md_table(_x[["family", "cohort", "auroc_diff_runavg", "boot_lo", "boot_hi", "delong_diff_run0", "delong_p", "nb_lo", "nb_hi", "nb_p", "p_tost_margin_0.01"]], 4)}

{mde_line}

{md_table(T["table12_variance_components"][T["table12_variance_components"].metric == "auroc"], 4)}

## 4. Discrimination and calibration of every model (P2)
{chr(10).join(best_lines)}

{md_table(_perf[_sens_cols], 4)}

CKINN minus each competitor (AUROC; descriptive, no multiplicity correction):

{md_table(T["table13_ckinn_vs_competitors"][T["table13_ckinn_vs_competitors"].metric == "auroc"], 4)}

## 5. Clinical utility (P2)
{md_table(T["table16_net_benefit_points"], 4)}

{md_table(T["table17_threshold_metrics"][T["table17_threshold_metrics"].model.isin(["M1", "M2", "CKINN", "LR", "XGB_mono"])], 3)}

## 6. Censoring-aware evaluation (P2) and 3-year IPCW models (E)
{md_table(T["table18_censoring_aware"], 4)}

{md_table(T["table19_ipcw_3y_models"], 4)}

## 7. Subgroups (E; H5 is in section 2)
{md_table(T["table21_subgroup_gaps"][T["table21_subgroup_gaps"].model.isin(["M2", "CKINN", "M1"])], 4)}

## 8. Missingness (P2)
{md_table(T["table22_missingness_auroc"], 4)}

{md_table(T["table23_missingness_loss_tests"], 4)}

RICH leave-one-site-out, pooled held-out rows:

{md_table(T["table26_rich_site_pooled"], 4)}

## 9. Constraint evidence (E; manipulation check P1)
{md_table(T["table27_manipulation_check"], 4)}

{md_table(T["table28_constraint_variants"][["cohort", "variant", "auroc_minus_ckinn", "lo", "hi", "brier_minus_ckinn", "violation_rate_v1_table", "violation_rate_own_table", "violation_rate_dropped_feature"]], 4)}

Empirical sign of each constrained predictor (adjusted logistic regression on the evaluation rows) and agreement of each model's held-out partial
dependence with the prior:

{md_table(T["table29_empirical_signs_vs_pd"][T["table29_empirical_signs_vs_pd"].model.isin(["M2", "CKINN"])][["cohort", "model", "feature", "prior_sign", "adjusted_coef", "coef_lo", "coef_hi", "adjusted_evidence", "pd_change_p10_to_p90", "pd_sign_agrees_with_prior"]], 4)}

{md_table(T["table30_violation_sparse_and_other_cohort"], 4)}

## 10. Transport (P2) and secondary transport analyses (E)
{md_table(T["table31_transport"], 4)}

{md_table(T["table32_transport_recalibration_reweighting"], 4)}

{md_table(T["table33_pooled_training"], 4)}

## 11. Sensitivity analyses (E)
{md_table(T["table34_sensitivity"], 4)}

## 12. Limitations
{chr(10).join('- ' + x for x in LIMITS)}
"""
(REP / "report.md").write_text(REPORT, encoding="utf-8")
print(REPORT[:3000])
print(f"CELL 38 OK | report.md written ({len(REPORT):,} characters) | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 39 | manuscript-ready Methods and Results draft (British spelling); numbers formatted from stored tables only
_t0 = time.perf_counter()
_dc, _sp = T["table01_data_card"].set_index("cohort"), T["table04_split"].set_index("cohort")
_lam = T["table07_lambda"]
_lam_ch = _lam[_lam.chosen.astype(str) == "True"]


def _ep_sentence(fam_prefix: str, c: str) -> str:
    g = _ep[(_ep.family.str.startswith(fam_prefix)) & (_ep.unit == f"{c} (T2)")].set_index("endpoint")
    a = g.loc["AUROC (non-inferiority, margin 0.01)"]
    b = g.loc["Brier score"]
    return (f"{c}: AUROC difference {a.difference:.4f} (95% CI {a.ci_lo:.4f} to {a.ci_hi:.4f}; non-inferiority {a.status}), "
            f"Brier difference {b.difference:.5f} ({b.ci_lo:.5f} to {b.ci_hi:.5f})")


def _v(fam_prefix, h):
    r = _vd[(_vd.family.str.startswith(fam_prefix)) & (_vd.hypothesis == h)].iloc[0]
    return r.verdict


_tr = T["table31_transport"]
_tr_lines = [f"{r.source} to {r.target}, {r.model}: target cross-validated AUROC {r.auroc_target_cv:.3f}, external {r.auroc_external:.3f}, gap "
             f"{r.transport_gap:.3f}" for r in _tr[_tr.model.isin(["M1", "M2", "CKINN"])].itertuples()]
METHODS = f"""# Methods and Results (draft generated from stored tables)

_Draft for the manuscript. British (Oxford) spelling. Every number in this file is formatted by code from the tables of the v2 run (mode `{CFG.mode}`, configuration hash `{PROTOCOL_HASH[:16]}`). References marked TODO(cite) must be checked before submission._

## Methods

**Design and amendments.** This is a re-analysis (version 2) of a study whose version 1 results had been seen. All changes were fixed in a protocol before any version 2 result existed; each is listed with its reason in the amendment log (protocol_v2.md, section 2), and analyses are labelled as pre-specified in version 1, pre-specified in version 2, or exploratory.

**Data.** Three public, de-identified datasets were used: the NAGALA cohort (Japan; {int(_dc.loc['NAGALA', 'rows']):,} participants, {int(_dc.loc['NAGALA', 'positives']):,} incident diabetes events), the Rich Healthcare Group cohort (China, 15 sites; {int(_dc.loc['RICH', 'rows']):,} participants, {int(_dc.loc['RICH', 'positives']):,} events) and an NHANES extract (USA, cycles 1999 to 2018), in which undiagnosed diabetes was defined as fasting plasma glucose of at least 7.0 mmol/L after at least 8 hours of fasting or HbA1c of at least 6.5% among adults reporting no diagnosis and no glucose-lowering medication ({int(_dc.loc['NHANES', 'labelled']):,} labelled adults, {int(_dc.loc['NHANES', 'positives']):,} positive). Predictors were routine clinical and laboratory measurements without glucose or HbA1c (tier T2); a tier adding them (T3) was a sensitivity analysis in the two cohorts. Exact zeros in laboratory values were treated as not measured. In each cohort {CFG.dev_frac:.0%} of participants (stratified) formed a development subset used only for tuning; the remainder formed the evaluation subset.

**Models.** M1 is a multilayer network with per-group encoders in which missing values take the training mean; M2 is the same network with the missingness mask as input and training-time missingness augmentation; CKINN is M2 with a penalty on input derivatives whose sign contradicts a fixed clinical sign table (risk increasing with age, body-mass index, waist, systolic pressure, family history and triglycerides; decreasing with HDL cholesterol and regular exercise). Within a fold, M1, M2 and CKINN shared the seed, the inner validation split and the training rows, so that CKINN with lambda 0 is identical to M2. Competitors were L2 logistic regression, sign-constrained logistic regression, a spline additive logistic model, random forest, XGBoost and LightGBM with and without monotone constraints from the same sign table, and multiple imputation by chained equations ({CFG.mi_m} imputations) followed by logistic regression.

**Tuning.** On the development subset, M1 and M2 each evaluated {CFG.tune_mlp_configs} configurations; CKINN took M2's best configuration among the first {CFG.tune_ck_base} and evaluated {len(CFG.lam_grid)} values of lambda, the same budget of {CFG.tune_mlp_configs} evaluations. Lambda was the largest value whose development score was within one standard error of the best score among lambda 0 and the grid (declared before any version 2 result).

**Evaluation.** Each model was evaluated by {CFG.n_repeats} repeats of stratified {CFG.n_splits}-fold cross-validation on the evaluation subset, with network seed r in repeat r; in repeat 0 the networks were also trained with {CFG.n_seeds - 1} further seeds. Metrics were computed on the pooled out-of-fold predictions of each run and averaged over runs. Uncertainty came from a paired participant bootstrap ({CFG.n_boot:,} draws) applied identically to all runs and models; one-sided p-values were read from the draws. DeLong's test on one run, the corrected resampled t-test over the {CFG.n_repeats * CFG.n_splits} folds and, in the Rich cohort, a cluster bootstrap over sites were cross-checks. Two hypothesis families (CKINN minus M2, primary; CKINN minus M1) were each corrected by Holm's method across their primary endpoints: H1 AUROC non-inferiority (margin {CFG.ni_margin}); H2 Brier score and distance of the calibration slope from 1; H3 the transport gap (AUROC of the same model class cross-validated inside the target cohort minus the AUROC of the source-trained model applied unchanged) and absolute calibration-in-the-large in the target; H4 the agreement of integrated-gradient attributions across seeds; H5 the sex gap and age-band range of sensitivity at a false-positive rate of {CFG.fpr_matched}. Secondary analyses covered calibration (slope, calibration-in-the-large, expected calibration error and reliability curves), decision curves, censoring-aware metrics (Harrell's and Uno's C and the inverse-probability-of-censoring weighted AUROC at {CFG.horizon:g} years), subgroup metrics, missingness injected at test time, leave-one-site-out validation, constraint ablations with wrong-sign controls, empirical sign checks against held-out partial dependence, and transport with recalibration, reweighting and pooled training. TODO(cite) for each method.

## Results

**Verdicts.** In family A (CKINN minus M2), H1 was {_v('A', 'H1')}, H2 {_v('A', 'H2')}, H3 {_v('A', 'H3')}, H4 {_v('A', 'H4')} and H5 {_v('A', 'H5')}. In family B (CKINN minus M1), H1 was {_v('B', 'H1')}, H2 {_v('B', 'H2')}, H3 {_v('B', 'H3')}, H4 {_v('B', 'H4')} and H5 {_v('B', 'H5')}.

**CKINN versus M2.** {'; '.join(_ep_sentence('A', c) for c in COHORTS)}.

**CKINN versus M1.** {'; '.join(_ep_sentence('B', c) for c in COHORTS)}.

**Best-performing models.** {' '.join(x[2:] for x in best_lines)}

**Power.** {mde_line}

**Transport.** {'; '.join(_tr_lines)}.

## Limitations
{chr(10).join('- ' + x for x in LIMITS)}
"""
(REP / "methods_results_draft.md").write_text(METHODS, encoding="utf-8")
print(METHODS[:2500])
print(f"CELL 39 OK | Methods and Results draft written | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 40 | TRIPOD+AI checklist filled from what the code did (item wording abbreviated; check the numbering against the official
# checklist before submission: TODO(verify))
_t0 = time.perf_counter()
TRIPOD = [
    ("Title and abstract", "Identify the study as developing and evaluating prediction models; summarise", "To be written; report.md and methods_results_draft.md give the numbers"),
    ("Background and objectives", "Rationale, intended use, objectives", "Proposal and protocol_v2.md section 1; outputs are research risk estimates, not diagnoses"),
    ("Data sources", "Source and dates of data", f"NAGALA, Rich Healthcare Group, NHANES 1999-2018 extract; file hashes in manifest.json"),
    ("Participants", "Eligibility, setting, treatments", "Cell 17; NHANES exclusions (A11) in table03; no rows removed in NAGALA or RICH"),
    ("Data preparation", "Pre-processing and quality checks", "AUDIT.md; cell 17; zeros treated as missing (A12); winsorising, log and z-score fitted in training folds"),
    ("Outcome", "Definition, timing, blinding", "Incident diabetes as recorded by each cohort; NHANES laboratory definition; follow-up windows differ (AUDIT F15)"),
    ("Predictors", "Definition, timing, measurement", "table05 feature sets; T2 excludes glucose and HbA1c; T3 sensitivity"),
    ("Sample size", "Justification", "Fixed by the data; table04 events; minimum detectable effect in table11"),
    ("Missing data", "Handling", "Mask and augmentation (M2, CKINN); mean fill (M1); native handling (trees); multiple imputation baseline; tables 22-24"),
    ("Analytical methods", "Model type, building, tuning, internal validation", "protocol_v2.md sections 4-6; tables 06-07; repeated CV with paired seeds"),
    ("Class imbalance", "Methods used", "None (no resampling); output bias initialised at the training prevalence"),
    ("Fairness", "Approaches to fairness", "H5 and tables 20-21 (sex, age band, site, ethnic group); minimum-events rule"),
    ("Model output", "Form of predictions", "Probabilities; thresholds from inner validation splits for operating points"),
    ("Training versus evaluation", "Differences between data used", "Development rows tune only; evaluation rows by cross-validation; transport and site analyses external"),
    ("Performance measures", "Metrics and their uncertainty", "AUROC, AUPRC, Brier, scaled Brier, slope, calibration-in-the-large, ECE, net benefit, censoring-aware metrics; bootstrap intervals"),
    ("Model updating / recalibration", "If done", "Recalibration in the small in the target (table32, target labels used, exploratory)"),
    ("Ethics", "Approval and consent", "Public de-identified data; check each source's terms before submission: TODO(verify)"),
    ("Funding and conflicts", "Sources and roles", "To be completed by the authors"),
    ("Protocol and registration", "Where available", "protocol_v2.md with amendment log and configuration hash; not registered"),
    ("Data and code availability", "Access", "Data from public sources; this notebook and library (ckv2) reproduce every table"),
    ("Patient and public involvement", "Details", "None"),
    ("Participant flow and characteristics", "Numbers, events, missingness", "tables 01-04"),
    ("Model specification", "Full model or how to obtain it", "Configuration and chosen hyperparameters in manifest.json; code in ckv2"),
    ("Model performance", "With intervals, by subgroup", "tables 08-34; figures 1-6"),
    ("Interpretation and limitations", "Overall interpretation, limitations, usability", "report.md section 12; methods_results_draft.md"),
]
TRIPOD_MD = "# TRIPOD+AI checklist (filled from what the code did)\n\nItem wording is abbreviated and the numbering must be checked against the official TRIPOD+AI checklist before submission: TODO(verify).\n\n" + \
    md_table(pd.DataFrame(TRIPOD, columns=["Section", "Item (abbreviated)", "Where addressed in this study"])) + "\n"
(REP / "TRIPOD_AI_checklist.md").write_text(TRIPOD_MD, encoding="utf-8")
print(f"CELL 40 OK | TRIPOD+AI checklist written | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 41 | integrity scan: configuration hash; every decimal number in the report and the draft traces to a stored table or to the
# configuration; no result-like number in this notebook's markdown cells; no row-level file among the archived outputs
_t0 = time.perf_counter()
_fz = json.loads((OUT / "report" / "frozen_protocol_v2.json").read_text())
assert _fz["sha256"] == C.config_hash(CFG) == PROTOCOL_HASH, "configuration changed after it was frozen"
_vals = np.concatenate([t.select_dtypes(include=[float, int]).to_numpy(float).ravel() for t in T.values()])
_cfgv = [float(v) for v in json.loads(CFG.to_json()).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
_cfgv += [float(x) for v in json.loads(CFG.to_json()).values() if isinstance(v, list) for x in v if isinstance(x, (int, float))]
_vals = np.concatenate([_vals[np.isfinite(_vals)], np.array(_cfgv)])
_untraced = []
for name, text in (("report.md", REPORT), ("methods_results_draft.md", METHODS)):
    body = re.sub(r"`[^`]*`", "", text)
    for mm in re.finditer(r"(?<![\w.])-?\d+\.(\d{2,})(?![\w.])", body):
        tok, d = float(mm.group(0)), len(mm.group(1))
        if not np.any(np.abs(np.round(_vals, d) - tok) < 10 ** -(d + 2)):
            _untraced.append((name, mm.group(0)))
assert not _untraced, f"numbers that no stored table supports: {_untraced[:10]}"
_nb_path = next((p for p in [Path(os.environ.get("CKINN_NOTEBOOK", "")), *Path(".").glob("CKINN_vs_MLP_real_data_final*.ipynb")] if p.is_file()), None)
if _nb_path is not None:
    _md = [c for c in json.loads(_nb_path.read_text())["cells"] if c["cell_type"] == "markdown"]
    _bad = [x for c in _md for x in re.findall(r"\d\.\d{3,}", "".join(c["source"]))]
    assert not _bad, f"result-like numbers in markdown cells: {_bad[:5]}"
    print(f"markdown scan of {_nb_path.name}: {len(_md)} markdown cells, no result-like numbers")
else:
    print("notebook file not found in the working folder: markdown scan skipped (set CKINN_NOTEBOOK to its path)")
_archived = [p for p in OUT.rglob("*") if p.is_file() and CACHE not in p.parents]
assert not [p for p in _archived if p.suffix in (".npz", ".pkl", ".xlsx") or p.name.startswith("_raw_")], "row-level file outside the cache"
print(f"CELL 41 OK | {len(_untraced)} untraced numbers; configuration hash intact | {time.perf_counter() - _t0:.0f}s")

# %%
# CELL 42 | runtime, manifest with package versions, input and output hashes, and the outputs archive (aggregate files only)
_t0 = time.perf_counter()
_rt = []
for stage, specs in (("tuning", [s for v in _specs.values() for s in v] + [s for v in _lspecs.values() for s in v]),
                     ("main runs and seed study", [s for c in CV_SPECS for s in CV_SPECS[c] + SEED_SPECS[c]]), ("repeat-0 analyses", _all),
                     ("transport, pooled, sites", TR_SPECS + POOL_SPECS + SITE_SPECS)):
    secs = [EN.load(CACHE, s["id"])["seconds"] for s in specs]
    _rt.append({"stage": stage, "tasks": len(specs), "cpu_seconds": float(np.sum(secs)), "median_task_seconds": float(np.median(secs))})
RUNTIME = pd.DataFrame(_rt)
RUNTIME.to_csv(OUT / "tables" / "table35_runtime.csv", index=False)
print(RUNTIME.round(1).to_string(index=False))
MAN = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "mode": CFG.mode, "configuration_sha256": PROTOCOL_HASH,
       "configuration": json.loads(CFG.to_json()), "inputs": {k: {"file": FILES[k].name, "sha256": FILE_HASH[k]} for k in FILES},
       "environment": VERSIONS, "workers": N_WORKERS, "wall_seconds": time.perf_counter() - T_START,
       "chosen_hyperparameters": {f"{k[0]}|{k[1]}": v for k, v in HP.items()}, "chosen_lambda": {f"{k[0]}|{k[1]}": v for k, v in LAM.items()},
       "verdicts": T["table10_verdicts"].to_dict("records"), "held_out_sites": SITES_HELD, "shared_predictors": SHARED,
       "library_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((LIB / "ckv2").glob("*.py"))}}
(REP / "manifest.json").write_text(json.dumps(MAN, indent=2, default=str))
_files = sorted(p for p in OUT.rglob("*") if p.is_file() and CACHE not in p.parents and p.suffix != ".zip" and p.name != "output_hashes.json")
(REP / "output_hashes.json").write_text(json.dumps({str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _files}, indent=1))
with zipfile.ZipFile(OUT / f"ckinn_v2_outputs_{CFG.mode}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in _files + [REP / "output_hashes.json"]:
        z.write(p, p.relative_to(OUT))
print(f"{len(_files)} files archived in {OUT / f'ckinn_v2_outputs_{CFG.mode}.zip'} (cache with row-level data excluded)")
print(f"CELL 42 OK | manifest and archive written | total wall time {time.perf_counter() - T_START:.0f}s")

# %% [markdown]
# ## References and citation policy
# Only sources checked in PubMed or Scite are listed without a flag; anything else carries TODO(cite) and must be checked before submission. The verified list is kept in `references.md` next to this notebook.
