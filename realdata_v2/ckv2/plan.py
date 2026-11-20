"""Builds the task specifications of every stage (protocol_v2.md, sections 5, 6 and 8) and assembles out-of-fold predictions."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import metrics as mt
from .config import (C_GRID, MAIN_MODELS, MLP_DEFAULT, MLP_SPACE, NN_MODELS, RF_GRID, RUN0_EXTRA, SIGNS_CORE, SIGNS_EXT, SIGNS_T2,
                     SIGNS_T3, TREE_DEFAULT, TREE_SPACE, seed_for)
from .engine import load
from .missing import scen_tag, scenarios
from .nets import weighted_val_score
from .prep import Prep


def tid(*parts) -> str:
    return "__".join(str(p).replace("/", "-").replace(" ", "") for p in parts)


def signs_for(tier: str, feats: Sequence[str], table: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    t = table if table is not None else (SIGNS_T3 if tier == "T3" else SIGNS_T2)
    return {f: s for f, s in t.items() if f in feats}


# ------------------------------------------------------------------ tuning
def _grid(space: dict) -> List[dict]:
    keys = sorted(space)
    out = [{}]
    for k in keys:
        out = [dict(o, **{k: v}) for o in out for v in space[k]]
    return out


def sample_configs(space: dict, default: dict, n: int, seed: int) -> List[dict]:
    """The default first, then n-1 distinct configurations drawn without replacement from the full grid."""
    grid = [g for g in _grid(space) if g != default]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(grid), size=min(n - 1, len(grid)), replace=False)
    return [dict(default)] + [grid[i] for i in pick]


def config_lists(cohort: str, cfg) -> Dict[str, List[dict]]:
    mlp = sample_configs(MLP_SPACE, MLP_DEFAULT, cfg.tune_mlp_configs, seed_for("hp", cohort, "MLP"))
    xgb = sample_configs(TREE_SPACE["XGB"], TREE_DEFAULT["XGB"], cfg.tune_tree_configs, seed_for("hp", cohort, "XGB"))
    lgb = sample_configs(TREE_SPACE["LGBM"], TREE_DEFAULT["LGBM"], cfg.tune_tree_configs, seed_for("hp", cohort, "LGBM"))
    cg = [{"C": c} for c in C_GRID]
    rf = RF_GRID if cfg.mode == "full" else RF_GRID[:2]
    if cfg.mode != "full":
        cg = cg[1:3]
    return {"M1": mlp, "M2": mlp, "XGB": xgb, "XGB_mono": xgb, "LGBM": lgb, "LGBM_mono": lgb, "LR": cg, "cLR": cg, "GAM": cg, "RF": rf}


def tune_specs(cohort: str, dev: pd.DataFrame, feats: Sequence[str], cfg) -> List[dict]:
    folds = list(StratifiedKFold(cfg.tune_folds, shuffle=True, random_state=seed_for("devcv", cohort)).split(dev, dev.y))
    sg = signs_for("T2", feats)
    specs = []
    for model, lst in config_lists(cohort, cfg).items():
        for ci, hp in enumerate(lst):
            for k, (i, j) in enumerate(folds):
                specs.append({"id": tid("tune", cohort, model, f"c{ci}", f"k{k}"), "model": model, "feats": list(feats), "hp": hp,
                              "seed": seed_for("devfit", cohort, k), "signs": sg, "train_frame": f"{cohort}_dev", "train_idx": i,
                              "tests": [{"tag": "val", "frame": f"{cohort}_dev", "idx": j}], "_val_idx": j, "_model": model, "_ci": ci})
    return specs


def lam_folds(cohort: str, dev: pd.DataFrame, cfg):
    out = []
    for rep in range(cfg.lam_reps):
        rs = seed_for("devcv", cohort) if rep == 0 else seed_for("devcv", cohort, rep)
        for k, (i, j) in enumerate(StratifiedKFold(cfg.tune_folds, shuffle=True, random_state=rs).split(dev, dev.y)):
            out.append((rep, k, i, j))
    return out


def lam_specs(cohort: str, tier: str, dev: pd.DataFrame, feats: Sequence[str], base_hp: dict, cfg) -> List[dict]:
    sg = signs_for(tier, feats)
    specs = []
    for lam in (0.0,) + tuple(cfg.lam_grid):
        for rep, k, i, j in lam_folds(cohort, dev, cfg):
            specs.append({"id": tid("lam", cohort, tier, f"l{lam:g}", f"r{rep}", f"k{k}"), "model": "CKINN", "feats": list(feats),
                          "hp": base_hp, "seed": seed_for("devlam", cohort, rep, k), "signs": sg, "lam": lam,
                          "train_frame": f"{cohort}_dev", "train_idx": i, "tests": [{"tag": "val", "frame": f"{cohort}_dev", "idx": j}],
                          "_val_idx": j, "_lam": lam})
    return specs


def score_specs(specs: Sequence[dict], dev: pd.DataFrame, cache_dir) -> pd.DataFrame:
    rows = []
    y = dev.y.to_numpy()
    for s in specs:
        r = load(cache_dir, s["id"])
        rows.append({"id": s["id"], "model": s.get("_model", "CKINN"), "ci": s.get("_ci"), "lam": s.get("_lam"),
                     "score": weighted_val_score(y[s["_val_idx"]], r["preds"]["val"].astype(float)), "epochs": r["info"]["epochs"]})
    return pd.DataFrame(rows)


def select_lambda(scores: pd.DataFrame, grid: Sequence[float]) -> Tuple[float, pd.DataFrame]:
    """Protocol 5: largest positive lambda whose mean score is within one SE (over the fits) of the minimum over {0} and the grid."""
    g = scores.groupby("lam").score.agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    best = g.loc[g["mean"].idxmin()]
    ok = g[(g["lam"] > 0) & (g["mean"] <= best["mean"] + best["se"])]
    lam = float(ok["lam"].max()) if len(ok) else float(min(grid))
    g["chosen"] = g["lam"] == lam
    g["within_1se"] = g["mean"] <= best["mean"] + best["se"]
    return lam, g


# ------------------------------------------------------------------ evaluation stages
def explain_rows(cohort: str, test_idx: np.ndarray, k: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for("expl", cohort, k))
    return np.sort(rng.choice(test_idx, min(n, len(test_idx)), replace=False))


def pd_rows(cohort: str, test_idx: np.ndarray, k: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for("pd", cohort, k))
    return np.sort(rng.choice(test_idx, min(n, len(test_idx)), replace=False))


def fit_seed(cohort: str, tier: str, r: int, k: int, s: int) -> int:
    """The same seed for every model of a fold and seed index (pairing, A1)."""
    return seed_for("fit", cohort, tier, r, k, s)


def cv_specs(cohort: str, tier: str, folds, feats, HP, LAM, cfg, frame_key: Optional[str] = None, models=None,
             repeats=None, variant: str = "", signs: Optional[Dict[str, int]] = None, scen: bool = True, extras: bool = True) -> List[dict]:
    fk = frame_key or cohort
    sg = signs if signs is not None else signs_for(tier, feats)
    models = models or MAIN_MODELS
    specs = []
    for r in (range(cfg.n_repeats) if repeats is None else repeats):
        for k, (i, j) in enumerate(folds[r]):
            tests = [{"tag": "none", "frame": fk, "idx": j}]
            if scen and tier == "T2":
                tests += [{"tag": scen_tag(m, q), "frame": fk, "idx": j, "corrupt": (m, q, seed_for("corrupt", cohort, r, k, m, q))}
                          for m, q in scenarios(cfg)]
            mlist = list(models) + (list(RUN0_EXTRA) if (r == 0 and extras and models is MAIN_MODELS) else [])
            for m in mlist:
                hp_key = {"MI_LR": "LR", "LR_aug": "LR", "XGB_aug": "XGB"}.get(m, m)
                spec = {"id": tid("cv", cohort, tier, variant or "main", f"r{r}", f"k{k}", m, f"s{r}"), "model": m, "feats": list(feats),
                        "hp": HP[(cohort, hp_key)], "seed": fit_seed(cohort, tier, r, k, r), "signs": sg,
                        "lam": LAM[(cohort, tier)] if m == "CKINN" else 0.0, "train_frame": fk, "train_idx": i, "tests": tests,
                        "_r": r, "_k": k, "_s": r, "_model": m}
                if r == 0 and extras:
                    spec["pd"] = {"frame": fk, "idx": pd_rows(cohort, j, k, cfg.n_pd_rows), "features": list(sg)}
                    if m in NN_MODELS:
                        spec["explain"] = {"frame": fk, "idx": explain_rows(cohort, j, k, cfg.n_explain), "signs": signs_for(tier, feats),
                                           "sparse": True, "sparse_pool": j}
                specs.append(spec)
    return specs


def seed_specs(cohort: str, tier: str, folds, feats, HP, LAM, cfg) -> List[dict]:
    sg = signs_for(tier, feats)
    specs = []
    for s in range(1, cfg.n_seeds):
        for k, (i, j) in enumerate(folds[0]):
            for m in NN_MODELS:
                specs.append({"id": tid("cv", cohort, tier, "seed", "r0", f"k{k}", m, f"s{s}"), "model": m, "feats": list(feats),
                              "hp": HP[(cohort, m)], "seed": fit_seed(cohort, tier, 0, k, s), "signs": sg,
                              "lam": LAM[(cohort, tier)] if m == "CKINN" else 0.0, "train_frame": cohort, "train_idx": i,
                              "tests": [{"tag": "none", "frame": cohort, "idx": j}],
                              "explain": {"frame": cohort, "idx": explain_rows(cohort, j, k, cfg.n_explain), "signs": sg},
                              "_r": 0, "_k": k, "_s": s, "_model": m})
    return specs


def variant_tables(feats: Sequence[str]) -> Dict[str, Dict[str, int]]:
    base = signs_for("T2", feats)
    out = {f"loco_{f}": {g: s for g, s in base.items() if g != f} for f in base}
    flip = lambda keys: {g: (-s if g in keys else s) for g, s in base.items()}
    out["wrong_age"], out["wrong_bmi"], out["wrong_all"] = flip({"age_years"}), flip({"bmi_kgm2"}), flip(set(base))
    out["sign_core"] = signs_for("T2", feats, SIGNS_CORE)
    out["sign_ext"] = signs_for("T2", feats, SIGNS_EXT)
    return out


def variant_specs(cohort: str, folds, feats, HP, LAM, cfg) -> List[dict]:
    """Protocol 8: non-circular constraint analyses (repeat 0, seed index 0, same seed as the main CKINN of repeat 0)."""
    base = signs_for("T2", feats)
    specs = []
    tables = variant_tables(feats)
    lam_path = {f"lam_{l:g}": l for l in cfg.lam_grid}
    for k, (i, j) in enumerate(folds[0]):
        ex = explain_rows(cohort, j, k, cfg.n_explain)
        for v, table in list(tables.items()) + [(n, base) for n in lam_path]:
            lam = lam_path.get(v, LAM[(cohort, "T2")])
            specs.append({"id": tid("var", cohort, "T2", v, "r0", f"k{k}", "CKINN", "s0"), "model": "CKINN", "feats": list(feats),
                          "hp": HP[(cohort, "CKINN")], "seed": fit_seed(cohort, "T2", 0, k, 0), "signs": table, "lam": lam,
                          "train_frame": cohort, "train_idx": i, "tests": [{"tag": "none", "frame": cohort, "idx": j}],
                          "explain": {"frame": cohort, "idx": ex, "signs": base},
                          "extra_viol": [("own_table", cohort, ex, table)] + ([("dropped", cohort, ex, {v[5:]: base[v[5:]]})] if v.startswith("loco_") else []),
                          "_variant": v, "_k": k})
    return specs


def ipcw_weights(time: np.ndarray, event: np.ndarray, tau: float):
    """3-year endpoint and inverse-probability-of-censoring weights from the training rows (protocol 8)."""
    ut, G = mt.km_censoring(time, event)
    y3 = ((event == 1) & (time <= tau)).astype(int)
    w = np.where(y3 == 1, 1 / np.maximum(mt.G_at(ut, G, time, left=True), 1e-6), np.where(time > tau, 1 / max(float(mt.G_at(ut, G, tau)[0]), 1e-6), 0.0))
    return y3, w


def ipcw_specs(cohort: str, folds, feats, HP, LAM, cfg, fr: pd.DataFrame) -> List[dict]:
    specs = []
    sg = signs_for("T2", feats)
    for k, (i, j) in enumerate(folds[0]):
        y3, w = ipcw_weights(fr.follow_years.to_numpy()[i], fr.y.to_numpy()[i], cfg.horizon)
        for m in ("LR", "XGB_mono", "M2", "CKINN"):
            specs.append({"id": tid("ipcw", cohort, "T2", "r0", f"k{k}", m, "s0"), "model": m, "feats": list(feats), "hp": HP[(cohort, m)],
                          "seed": fit_seed(cohort, "T2", 0, k, 0), "signs": sg, "lam": LAM[(cohort, "T2")] if m == "CKINN" else 0.0,
                          "train_frame": cohort, "train_idx": i, "train_y": y3, "train_w": w,
                          "tests": [{"tag": "none", "frame": cohort, "idx": j}], "_k": k, "_model": m})
    return specs


def shared_features(fr_a: pd.DataFrame, fr_b: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    """v1 rule: observed in more than 90% of NAGALA and more than 30% of RICH."""
    return [c for c in candidates if fr_a[c].notna().mean() > 0.9 and fr_b[c].notna().mean() > 0.3]


def domain_weights(src: pd.DataFrame, tgt: pd.DataFrame, feats: Sequence[str], clip: Tuple[float, float]) -> np.ndarray:
    """Label-free importance weights for source rows: odds of target membership from an L2 LR on the shared predictors."""
    both = pd.concat([src[feats], tgt[feats]], ignore_index=True)
    z = np.r_[np.zeros(len(src)), np.ones(len(tgt))]
    pp = Prep(feats).fit(both)
    X = pp.transform(both)[0]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, z)
    p = lr.predict_proba(X[: len(src)])[:, 1]
    w = p / (1 - p) * (len(src) / len(tgt))
    w = np.clip(w, *clip)
    return w / w.mean()


def transport_specs(FR, FOLDS, shared, HP, LAM, cfg) -> List[dict]:
    specs = []
    for src, tgt in (("NAGALA", "RICH"), ("RICH", "NAGALA")):
        sg = signs_for("T2", shared)
        n_src, n_tgt = len(FR[src]), len(FR[tgt])
        ex_t = np.sort(np.random.default_rng(seed_for("trexpl", tgt)).choice(n_tgt, min(cfg.n_explain, n_tgt), replace=False))
        ex_s = np.sort(np.random.default_rng(seed_for("trexpl", src)).choice(n_src, min(cfg.n_explain, n_src), replace=False))
        w_rw = domain_weights(FR[src], FR[tgt], shared, cfg.weight_clip)
        for m in ("LR", "XGB", "XGB_mono") + NN_MODELS:
            seeds = range(cfg.n_seeds) if m in NN_MODELS else [0]
            for s in seeds:
                base = {"model": m, "feats": list(shared), "hp": HP[(src, m)], "signs": sg, "lam": LAM[(src, "T2")] if m == "CKINN" else 0.0}
                specs.append({**base, "id": tid("trext", src, tgt, m, f"s{s}"), "seed": seed_for("tr_ext", src, s), "train_frame": src,
                              "train_idx": np.arange(n_src), "tests": [{"tag": "tgt", "frame": tgt, "idx": np.arange(n_tgt)}],
                              "extra_viol": [("tgt", tgt, ex_t, sg), ("src", src, ex_s, sg)], "_src": src, "_tgt": tgt, "_model": m, "_s": s})
                for k, (i, j) in enumerate(FOLDS[tgt][0]):
                    bt = {"model": m, "feats": list(shared), "hp": HP[(tgt, m)], "signs": sg, "lam": LAM[(tgt, "T2")] if m == "CKINN" else 0.0}
                    specs.append({**bt, "id": tid("trref", tgt, m, f"k{k}", f"s{s}"), "seed": seed_for("tr_ref", tgt, k, s), "train_frame": tgt,
                                  "train_idx": i, "tests": [{"tag": "none", "frame": tgt, "idx": j}], "_tgt": tgt, "_model": m, "_s": s, "_k": k})
                if s == 0 and m in ("LR", "M2", "CKINN"):
                    specs.append({**base, "id": tid("trrw", src, tgt, m, "s0"), "seed": seed_for("tr_ext", src, 0), "train_frame": src,
                                  "train_idx": np.arange(n_src), "train_w": w_rw, "tests": [{"tag": "tgt", "frame": tgt, "idx": np.arange(n_tgt)}],
                                  "_src": src, "_tgt": tgt, "_model": m})
    return specs


def pooled_specs(FR, FOLDS, pooled_feats, HP, LAM, cfg, cap: int = 20000) -> List[dict]:
    """A17: pooled NAGALA + RICH training (each cohort's training rows capped at 20,000 as in v1); training-only standardisation."""
    off = {"NAGALA": 0, "RICH": len(FR["NAGALA"])}
    specs = []
    for k in range(cfg.n_splits):
        tr, tests = [], []
        for c in ("NAGALA", "RICH"):
            i, j = FOLDS[c][0][k]
            if len(i) > cap:
                i = np.sort(train_test_split(i, train_size=cap, stratify=FR[c].y.to_numpy()[i], random_state=seed_for("poolcap", c, k))[0])
            tr.append(i + off[c])
            tests.append({"tag": c, "frame": "POOL", "idx": j + off[c]})
        for m in ("M2", "CKINN"):
            specs.append({"id": tid("pool", m, f"k{k}"), "model": m, "feats": list(pooled_feats), "hp": HP[("RICH", m)],
                          "seed": seed_for("pool", k), "signs": signs_for("T2", pooled_feats), "lam": LAM[("RICH", "T2")] if m == "CKINN" else 0.0,
                          "train_frame": "POOL", "train_idx": np.concatenate(tr), "tests": tests, "_k": k, "_model": m})
    return specs


def site_specs(fr: pd.DataFrame, feats, HP, LAM, cfg) -> Tuple[List[dict], List[str]]:
    ev = fr.groupby("site").y.sum()
    sites = sorted([s for s in ev.index if ev[s] >= cfg.site_min_events], key=lambda s: int(s))
    specs = []
    site = fr.site.to_numpy()
    for s in sites:
        te, tr = np.flatnonzero(site == s), np.flatnonzero(site != s)
        for m in ("LR", "XGB", "XGB_mono") + NN_MODELS:
            specs.append({"id": tid("site", s, m), "model": m, "feats": list(feats), "hp": HP[("RICH", m)], "seed": seed_for("site", s, 0),
                          "signs": signs_for("T2", feats), "lam": LAM[("RICH", "T2")] if m == "CKINN" else 0.0, "train_frame": "RICH",
                          "train_idx": tr, "tests": [{"tag": "none", "frame": "RICH", "idx": te}], "_site": s, "_model": m})
    return specs, sites


# ------------------------------------------------------------------ assembly
def assemble(specs: Sequence[dict], n: int, cache_dir, tag: str = "none", key=lambda s: (s["_model"], s["_r"], s["_s"])) -> Dict[tuple, np.ndarray]:
    """Out-of-fold vectors: for each key, place each fold's predictions at its test rows. Every row must be filled."""
    out: Dict[tuple, np.ndarray] = {}
    for s in specs:
        k = key(s)
        if k not in out:
            out[k] = np.full(n, np.nan, np.float32)
        t = [x for x in s["tests"] if x["tag"] == tag]
        if not t:
            continue
        out[k][t[0]["idx"]] = load(cache_dir, s["id"])["preds"][tag]
    for k, v in out.items():
        assert np.isfinite(v).all(), f"incomplete out-of-fold predictions for {k} ({tag})"
    return out
