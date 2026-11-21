"""Post hoc additions for the 28-step protocol (protocol_v2.md, section 10; label PH).

Decided after the v2 results had been seen, to cover protocol steps the v2 analysis left partly or not covered. None of these
analyses enters a hypothesis family or changes a v2 verdict; the frozen configuration and its hash are untouched.
Model fits that repeat a main-run fit use the same seed and rows, so their unmodified predictions must equal the stored ones
(checked in the notebook as a reproducibility test).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import BINARY, C_GRID, MAIN_MODELS, PANEL, RUN0_EXTRA, seed_for
from .plan import fit_seed, signs_for, tid

SHAP_MODELS = ("LR", "XGB", "XGB_mono", "M1", "M2", "CKINN")
REFIT_MODELS = tuple(MAIN_MODELS) + tuple(RUN0_EXTRA) + ("SVM",)
TRANSPORT_EXTRA = ("RF", "LGBM", "LGBM_mono", "cLR", "GAM", "SVM", "MI_LR")
PROGRESSION_MODELS = ("LR", "XGB_mono", "M1", "M2", "CKINN")
TASK_LABEL = {"A": "undiagnosed diabetes", "B": "incident type 2 diabetes during follow-up"}

# One PubMed-verified reference per sign-table row (references.md); found after the table was frozen.
SIGN_CITATIONS = {
    "age_years": (17533210, "Wilson et al. 2007, Arch Intern Med (Framingham Offspring)"),
    "bmi_kgm2": (20493574, "Abdullah et al. 2010, Diabetes Res Clin Pract (meta-analysis)"),
    "waist_cm": (17494056, "Vazquez et al. 2007, Epidemiol Rev (meta-analysis)"),
    "sbp_mmHg": (26429079, "Emdin et al. 2015, J Am Coll Cardiol (4.1 million adults and meta-analysis)"),
    "dbp_mmHg": (26429079, "Emdin et al. 2015, J Am Coll Cardiol (4.1 million adults and meta-analysis)"),
    "family_history_dm": (23052052, "InterAct Consortium 2013, Diabetologia"),
    "tg_mmol": (17533210, "Wilson et al. 2007, Arch Intern Med (Framingham Offspring)"),
    "hdl_mmol": (17533210, "Wilson et al. 2007, Arch Intern Med (Framingham Offspring)"),
    "exercise_regular": (26092138, "Aune et al. 2015, Eur J Epidemiol (meta-analysis)"),
    "fpg_mmol": (17533210, "Wilson et al. 2007, Arch Intern Med (Framingham Offspring)"),
    "hba1c_pct": (20587727, "Zhang et al. 2010, Diabetes Care (systematic review)"),
}

LABELS = {"age_years": "age", "sex_female": "female sex", "bmi_kgm2": "BMI", "waist_cm": "waist circumference", "sbp_mmHg": "systolic BP",
          "dbp_mmHg": "diastolic BP", "smoker": "current smoking", "alcohol_level": "alcohol intake", "exercise_regular": "regular exercise",
          "family_history_dm": "family history of diabetes", "tg_mmol": "triglycerides", "hdl_mmol": "HDL cholesterol",
          "ldl_mmol": "LDL cholesterol", "tc_mmol": "total cholesterol", "alt_u_l": "ALT", "ast_u_l": "AST", "ggt_u_l": "GGT",
          "fpg_mmol": "fasting glucose", "hba1c_pct": "HbA1c", "tg_hdl_ratio": "TG/HDL ratio", "prediabetes_told": "told of prediabetes",
          "hypertension_told": "told of hypertension"}


def settings(cfg) -> dict:
    if cfg.mode == "full":
        return {"pair_rows": 1000, "shap_rows": 200, "n_bg": 50, "n_perm": 8, "reason_n": 50, "delta_sd": 0.25}
    return {"pair_rows": 150, "shap_rows": 30, "n_bg": 15, "n_perm": 4, "reason_n": 10, "delta_sd": 0.25}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ probes run inside the worker
def perturbation_violations(predict, df: pd.DataFrame, tr: pd.DataFrame, signs: Dict[str, int], feats: Sequence[str], delta_sd: float,
                            tol: float) -> Dict[str, dict]:
    """Model-agnostic violation rate (step 2): continuous predictors raised by delta_sd training SDs, binary predictors moved from 0 to
    1, observed values only; a violation is a sign-adjusted change of the log-odds below -tol. Works for every model."""
    z0 = _logit(predict(df))
    res = {}
    for f, s in signs.items():
        if f not in feats:
            continue
        v = df[f].to_numpy(float)
        obs = ~np.isnan(v)
        d2 = df.copy()
        if f in BINARY:
            rows = obs & (v == 0)
            d2.loc[rows, f] = 1.0
        else:
            rows = obs
            d2.loc[rows, f] = v[rows] + delta_sd * float(np.nanstd(tr[f].to_numpy(float)))
        if not rows.any():
            res[f] = {"n": 0, "viol": 0, "mean_change": float("nan")}
            continue
        ch = s * (_logit(predict(d2)) - z0)[rows]
        res[f] = {"n": int(rows.sum()), "viol": int((ch < -tol).sum()), "mean_change": float(ch.mean())}
    return res


def permutation_shap(predict, rows: pd.DataFrame, bg: pd.DataFrame, feats: Sequence[str], n_perm: int, rng):
    """Interventional SHAP values on the log-odds scale by permutation sampling with antithetic permutations and a background sample
    of training rows. Additive by construction: phi.sum(1) == fx - base for every row."""
    feats = list(feats)
    Xr, B = rows[feats].to_numpy(float), bg[feats].to_numpy(float)
    nb, p = B.shape
    half = max(1, n_perm // 2)
    base = float(_logit(predict(pd.DataFrame(B, columns=feats))).mean())
    phi, fx = np.zeros((len(Xr), p)), np.zeros(len(Xr))
    for i in range(len(Xr)):
        perms = [rng.permutation(p) for _ in range(half)]
        perms += [q[::-1] for q in perms]
        blocks = []
        for q in perms:
            cur = B.copy()
            blocks.append(cur.copy())
            for j in q:
                cur[:, j] = Xr[i, j]
                blocks.append(cur.copy())
        v = _logit(predict(pd.DataFrame(np.vstack(blocks), columns=feats))).reshape(len(perms), p + 1, nb).mean(axis=2)
        for k, q in enumerate(perms):
            phi[i, q] += np.diff(v[k])
        phi[i] /= len(perms)
        fx[i] = v[0, -1]
    return phi, base, fx


def reasoning_inputs(predict, rows: pd.DataFrame, tr: pd.DataFrame, signs: Dict[str, int], feats: Sequence[str], delta_sd: float, tol: float,
                     panel: Sequence[str]) -> dict:
    """Inputs for the patient-level reasoning text (steps 15 and 27): predicted probability, point-wise check of every constrained
    predictor, and the model's response when each blood-panel value is set to the training median (a model response, not advice)."""
    p0 = predict(rows)
    z0 = _logit(p0)
    cfeats = [f for f in signs if f in feats]
    viol = np.zeros((len(rows), len(cfeats)), bool)
    tested = np.zeros((len(rows), len(cfeats)), bool)
    for j, f in enumerate(cfeats):
        v = rows[f].to_numpy(float)
        obs = ~np.isnan(v)
        d2 = rows.copy()
        if f in BINARY:
            d2.loc[obs, f] = 1.0 - v[obs]
            direction = np.where(v == 0, 1.0, -1.0)
        else:
            d2.loc[obs, f] = v[obs] + delta_sd * float(np.nanstd(tr[f].to_numpy(float)))
            direction = np.ones(len(v))
        ch = signs[f] * direction * (_logit(predict(d2)) - z0)
        viol[:, j], tested[:, j] = obs & (ch < -tol), obs
    cf = [f for f in panel if f in feats]
    med = {f: float(np.nanmedian(tr[f].to_numpy(float))) for f in cf}
    p_cf = np.full((len(rows), len(cf)), np.nan)
    for j, f in enumerate(cf):
        obs = rows[f].notna().to_numpy()
        d2 = rows.copy()
        d2.loc[obs, f] = med[f]
        p_cf[obs, j] = predict(d2)[obs]
    return {"p": np.asarray(p0, float), "constrained": cfeats, "viol_point": viol, "tested": tested, "cf_feats": cf, "medians": med,
            "p_cf": p_cf, "x": rows[list(feats)].to_numpy(float), "feats": list(feats)}


def reasoning_text(rec: dict, i: int, task: str) -> str:
    """Fixed template (steps 15 and 27): top three contributions with direction, constraint consistency, one blood-panel counterfactual.
    Worded as the model's behaviour; never as a diagnosis or advice."""
    feats, phi, x, p = rec["feats"], rec["phi"][i], rec["x"][i], rec["p"][i]
    order = [j for j in np.argsort(-np.abs(phi)) if abs(phi[j]) > 0][:3]
    parts = []
    for j in order:
        val = "not recorded" if np.isnan(x[j]) else f"{x[j]:.3g}"
        parts.append(f"{LABELS.get(feats[j], feats[j])} ({val}) {'raises' if phi[j] > 0 else 'lowers'} the model's estimate")
    text = [f"Predicted probability of {TASK_LABEL[task]}: {p:.1%}.",
            "Largest contributions on the log-odds scale: " + ("; ".join(parts) if parts else "none") + "."]
    bad = [LABELS.get(f, f) for f, b in zip(rec["constrained"], rec["viol_point"][i]) if b]
    text.append("At these values the model's response agrees with every clinical prior it was given." if not bad
                else "At these values the model's response runs against the clinical prior for: " + ", ".join(bad) + ".")
    best, best_drop = None, 0.0
    for j, f in enumerate(rec["cf_feats"]):
        if np.isfinite(rec["p_cf"][i, j]) and p - rec["p_cf"][i, j] > best_drop:
            best, best_drop = j, p - rec["p_cf"][i, j]
    if best is None:
        text.append("Setting any one blood-panel value to the training median would not lower the model's estimate.")
    else:
        f = rec["cf_feats"][best]
        text.append(f"With {LABELS.get(f, f)} at the training median ({rec['medians'][f]:.3g}) the model's predicted probability would be "
                    f"{rec['p_cf'][i, best]:.1%} instead of {p:.1%}.")
    text.append("This describes the model's behaviour; it is not a diagnosis or a recommendation.")
    return " ".join(text)


# ------------------------------------------------------------------ weights
def balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n, n1 = len(y), max(int(y.sum()), 1)
    w = np.where(y == 1, n / (2 * n1), n / (2 * max(n - n1, 1)))
    return w / w.mean()


def group_weights(fr: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency weights over sex x age band, mean 1 (ablation M5, the v1 fairness weighting)."""
    g = fr["sex"].astype(str) + "|" + fr["age_band"].astype(str)
    w = 1.0 / g.map(g.value_counts()).to_numpy(float)
    return w / w.mean()


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


# ------------------------------------------------------------------ task specifications
def svm_tune_specs(cohort: str, dev: pd.DataFrame, feats: Sequence[str], cfg) -> List[dict]:
    folds = list(StratifiedKFold(cfg.tune_folds, shuffle=True, random_state=seed_for("devcv", cohort)).split(dev, dev.y))
    grid = C_GRID if cfg.mode == "full" else C_GRID[1:3]
    return [{"id": tid("phtune", cohort, "SVM", f"c{ci}", f"k{k}"), "model": "SVM", "feats": list(feats), "hp": {"C": c},
             "seed": seed_for("devfit", cohort, k), "train_frame": f"{cohort}_dev", "train_idx": i,
             "tests": [{"tag": "val", "frame": f"{cohort}_dev", "idx": j}], "_val_idx": j, "_model": "SVM", "_ci": ci}
            for ci, c in enumerate(grid) for k, (i, j) in enumerate(folds)]


def refit_specs(cohort: str, folds, feats: Sequence[str], HP, LAM, cfg) -> List[dict]:
    """Repeat 0, seed index 0, every model: identical to the main fit, plus BMI 24/35 pairs, perturbation violations, SHAP and the
    reasoning inputs."""
    st = settings(cfg)
    sg = signs_for("T2", feats)
    specs = []
    for k, (i, j) in enumerate(folds[0]):
        pr = np.sort(np.random.default_rng(seed_for("phpairs", cohort, k)).choice(j, min(st["pair_rows"], len(j)), replace=False))
        tests = [{"tag": "none", "frame": cohort, "idx": j}]
        if "bmi_kgm2" in feats:
            tests += [{"tag": "bmi24", "frame": cohort, "idx": pr, "set": {"bmi_kgm2": 24.0}},
                      {"tag": "bmi35", "frame": cohort, "idx": pr, "set": {"bmi_kgm2": 35.0}}]
        sr = np.sort(np.random.default_rng(seed_for("phshap", cohort)).choice(j, min(st["shap_rows"], len(j)), replace=False))
        for m in REFIT_MODELS:
            hp_key = {"MI_LR": "LR", "LR_aug": "LR", "XGB_aug": "XGB"}.get(m, m)
            spec = {"id": tid("ph", cohort, "T2", "refit", "r0", f"k{k}", m, "s0"), "model": m, "feats": list(feats), "hp": HP[(cohort, hp_key)],
                    "seed": fit_seed(cohort, "T2", 0, k, 0), "signs": sg, "lam": LAM[(cohort, "T2")] if m == "CKINN" else 0.0,
                    "train_frame": cohort, "train_idx": i, "tests": tests,
                    "perturb": {"frame": cohort, "idx": pr, "signs": sg, "delta_sd": st["delta_sd"]}, "_k": k, "_model": m, "_pairs": pr}
            if k == 0 and m in SHAP_MODELS:
                spec["shap"] = {"frame": cohort, "idx": sr, "n_bg": st["n_bg"], "n_perm": st["n_perm"], "seed": seed_for("phshapbg", cohort)}
                if m == "CKINN":
                    spec["reason"] = {"n": st["reason_n"], "signs": sg, "delta_sd": st["delta_sd"], "panel": [f for f in PANEL if f in feats]}
            specs.append(spec)
    return specs


def variant_list(cohort: str, feats: Sequence[str]) -> List[tuple]:
    """(variant, model, features, sign table, weights) for the repeat-0 post hoc variants."""
    sg = signs_for("T2", feats)
    out = []
    if "dbp_mmHg" in feats:
        out.append(("dbp_constrained", "CKINN", list(feats), dict(sg, dbp_mmHg=1), None))
    if "hdl_mmol" in sg:
        out.append(("wrong_hdl", "CKINN", list(feats), dict(sg, hdl_mmol=-sg["hdl_mmol"]), None))
    out.append(("m5_group_weights", "CKINN", list(feats), sg, "group"))
    for m in ("LR", "M2", "CKINN"):
        out.append(("class_weights", m, list(feats), sg, "balanced"))
    if "tg_mmol" in feats and "hdl_mmol" in feats:
        for m in ("LR", "M2", "CKINN"):
            out.append(("tg_hdl_ratio", m, list(feats) + ["tg_hdl_ratio"], dict(sg, tg_hdl_ratio=1), None))
    if cohort == "NHANES":
        for m in ("LR", "M2", "CKINN"):
            out.append(("history_questions", m, list(feats) + ["prediabetes_told", "hypertension_told"],
                        dict(sg, prediabetes_told=1, hypertension_told=1), None))
    return out


def variant_specs(cohort: str, folds, feats: Sequence[str], HP, LAM, cfg, fr: pd.DataFrame) -> List[dict]:
    specs = []
    for k, (i, j) in enumerate(folds[0]):
        for v, m, fts, sg, wkind in variant_list(cohort, feats):
            w = None
            if wkind == "group":
                w = group_weights(fr.iloc[i])
            elif wkind == "balanced":
                w = balanced_weights(fr.y.to_numpy()[i])
            specs.append({"id": tid("ph", cohort, "T2", v, "r0", f"k{k}", m, "s0"), "model": m, "feats": fts, "hp": HP[(cohort, m)],
                          "seed": fit_seed(cohort, "T2", 0, k, 0), "signs": sg, "lam": LAM[(cohort, "T2")] if m == "CKINN" else 0.0,
                          "train_frame": cohort, "train_idx": i, "train_w": w, "tests": [{"tag": "none", "frame": cohort, "idx": j}],
                          "_variant": v, "_model": m, "_k": k})
    return specs


def pool_indicator_specs(EVAL, FOLDS, pooled_feats: Sequence[str], HP, LAM, cfg, cap: int = 20000) -> List[dict]:
    """M4 of the 28-step ablation: pooled NAGALA + RICH training with a cohort indicator. Same rows, cap and seeds as the v2 pooled
    ablation, so the only difference is the indicator. Evaluated within the two training cohorts only (it has no value elsewhere)."""
    off = {"NAGALA": 0, "RICH": len(EVAL["NAGALA"])}
    feats = list(pooled_feats) + ["cohort_rich"]
    specs = []
    for k in range(cfg.n_splits):
        tr, tests = [], []
        for c in ("NAGALA", "RICH"):
            i, j = FOLDS[c][0][k]
            if len(i) > cap:
                i = np.sort(train_test_split(i, train_size=cap, stratify=EVAL[c].y.to_numpy()[i], random_state=seed_for("poolcap", c, k))[0])
            tr.append(i + off[c])
            tests.append({"tag": c, "frame": "POOL_IND", "idx": j + off[c]})
        for m in ("M2", "CKINN"):
            specs.append({"id": tid("phpool", m, f"k{k}"), "model": m, "feats": feats, "hp": HP[("RICH", m)], "seed": seed_for("pool", k),
                          "signs": signs_for("T2", feats), "lam": LAM[("RICH", "T2")] if m == "CKINN" else 0.0, "train_frame": "POOL_IND",
                          "train_idx": np.concatenate(tr), "tests": tests, "_k": k, "_model": m})
    return specs


def transport_specs(EVAL, FOLDS, shared: Sequence[str], HP, cfg, models=TRANSPORT_EXTRA) -> List[dict]:
    """Transport for the models the v2 transport analysis left out (seed 0), same seeds and rows as v2 (step 12 matrix)."""
    specs = []
    for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
        sg = signs_for("T2", shared)
        for m in models:
            hp_key = {"MI_LR": "LR"}.get(m, m)
            specs.append({"id": tid("phtrext", src, tgt, m, "s0"), "model": m, "feats": list(shared), "hp": HP[(src, hp_key)], "signs": sg,
                          "seed": seed_for("tr_ext", src, 0), "train_frame": src, "train_idx": np.arange(len(EVAL[src])),
                          "tests": [{"tag": "tgt", "frame": tgt, "idx": np.arange(len(EVAL[tgt]))}], "_src": src, "_tgt": tgt, "_model": m})
            for k, (i, j) in enumerate(FOLDS[tgt][0]):
                specs.append({"id": tid("phtrref", tgt, m, f"k{k}", "s0"), "model": m, "feats": list(shared), "hp": HP[(tgt, hp_key)],
                              "signs": sg, "seed": seed_for("tr_ref", tgt, k, 0), "train_frame": tgt, "train_idx": i,
                              "tests": [{"tag": "none", "frame": tgt, "idx": j}], "_tgt": tgt, "_model": m, "_k": k, "_r": 0, "_s": 0})
    return specs


def progression_frame(ev_rich: pd.DataFrame) -> pd.DataFrame:
    """Proposal section 3C, one step: among RICH evaluation participants with baseline FPG below 5.6 mmol/L, the outcome is FPG of at
    least 5.6 mmol/L at the final visit or diabetes during follow-up. The transition time is not observed."""
    rp = ev_rich[ev_rich.fpg_mmol < 5.6].copy()
    rp["y"] = ((rp.fpg_final >= 5.6) | (rp.label == 1)).astype(int)
    return rp.reset_index(drop=True)


def progression_specs(folds, feats: Sequence[str], HP, LAM, cfg) -> List[dict]:
    sg = signs_for("T2", feats)
    return [{"id": tid("phprog", m, f"k{k}"), "model": m, "feats": list(feats), "hp": HP[("RICH", m)], "seed": seed_for("phprog", k),
             "signs": sg, "lam": LAM[("RICH", "T2")] if m == "CKINN" else 0.0, "train_frame": "RICH_prog", "train_idx": i,
             "tests": [{"tag": "none", "frame": "RICH_prog", "idx": j}], "_k": k, "_model": m, "_r": 0, "_s": 0}
            for k, (i, j) in enumerate(folds[0]) for m in PROGRESSION_MODELS]


# ------------------------------------------------------------------ audit table (step 5)
UNAVAILABLE = [("ethnicity / ancestry", "NHANES only (descriptive fairness groups); absent in NAGALA and RICH"),
               ("previous prediabetes", "NHANES only (DIQ160, cycles 2005-06 onwards); post hoc exploratory input"),
               ("cardiovascular disease", "not in any of the three files used"),
               ("hypertension diagnosis", "NHANES only (BPQ020); post hoc exploratory input"),
               ("glucose-lowering medication", "NHANES only; used to define the exclusions of the detection label, not as an input"),
               ("diet", "not in any of the three files"), ("sleep", "not in any of the three files"),
               ("earlier glucose readings", "not in any of the three files"),
               ("weight and height", "NAGALA weight and RICH height and weight exist; BMI is used instead (derived from them)"),
               ("socio-economic status", "not in any of the three files")]


def audit_table(TASK: Dict[str, pd.DataFrame], FEATS: Dict[tuple, list], features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        r = {"predictor": f}
        for c in ("NAGALA", "RICH", "NHANES"):
            r[f"missing_pct_{c}"] = 100 * float(TASK[c][f].isna().mean()) if f in TASK[c] else 100.0
            r[f"used_{c}"] = ("T2" if f in FEATS.get((c, "T2"), []) else "") + (" T3" if f in FEATS.get((c, "T3"), []) and f not in FEATS.get((c, "T2"), []) else "")
        r["available_at_prediction_time"] = "yes (baseline measurement or questionnaire)"
        r["leakage_risk"] = ("high: defines the detection label, banned for NHANES; incident task only in T3" if f in ("fpg_mmol", "hba1c_pct")
                             else "moderate: implies an earlier glucose test; exploratory only" if f == "prediabetes_told" else "low")
        r["fairness_relevance"] = "group variable for subgroup analysis (also a predictor)" if f in ("sex_female", "age_years") else "none"
        r["prior_sign"] = {1: "+", -1: "-"}.get(signs_for("T3", [f]).get(f), "unconstrained")
        rows.append(r)
    return pd.DataFrame(rows)
