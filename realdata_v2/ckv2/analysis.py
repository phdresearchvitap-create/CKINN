"""Analysis of stored predictions (protocol_v2.md, sections 6 to 8). Pure functions of predictions, labels and configuration.

Bootstrap convention: in each draw the same participant weights are applied to every run and every model, the metric is computed
for each run and averaged over runs, and differences are taken between run averages (paired). Draws are split into chunks run in
worker processes; chunk c of a unit uses seed seed_for(<unit>, c), so results do not depend on the number of workers.
"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score

from . import metrics as mt
from .config import seed_for
from .engine import load

N_CHUNKS = 8


# ================================================================== assembly of stored predictions
def collect(specs: Sequence[dict], n: int, cache_dir, key=lambda s: (s["_model"], s["_r"], s["_s"])):
    """P[key][tag] = out-of-fold vector; THR[key] = (row threshold for specificity target, for sensitivity target); FOLD[key] = fold id."""
    P, THR, FOLD = {}, {}, {}
    for s in specs:
        k = key(s)
        r = load(cache_dir, s["id"])
        for t in s["tests"]:
            P.setdefault(k, {}).setdefault(t["tag"], np.full(n, np.nan, np.float32))[t["idx"]] = r["preds"][t["tag"]]
        none = [t for t in s["tests"] if t["tag"] in ("none", "tgt")]
        if none:
            ts, tn = THR.setdefault(k, (np.full(n, np.nan), np.full(n, np.nan)))
            ts[none[0]["idx"]], tn[none[0]["idx"]] = r["info"]["thr_spec"], r["info"]["thr_sens"]
            FOLD.setdefault(k, np.full(n, -1))[none[0]["idx"]] = s.get("_k", 0)
    for k, d in P.items():
        for tag, v in d.items():
            assert np.isfinite(v).all(), f"incomplete predictions {k} {tag}"
    return P, THR, FOLD


# ================================================================== bootstrap engine
def _weights(payload: dict, B: int, seed: int):
    n = len(payload["y"])
    base = payload.get("base_w")
    if payload.get("mode") == "cluster":
        it = mt.cluster_weight_iter(payload["cluster"], B, seed)
    elif payload.get("mode") == "survey":
        it = mt.strata_psu_weight_iter(payload["strata"], payload["psu"], base, B, seed)
        base = None
    else:
        it = mt.boot_weight_iter(n, B, seed)
    for w in it:
        yield w if base is None else w * base


class _Vec:
    """One prediction vector with its precomputed rank structures and point-estimate calibration (warm start)."""

    def __init__(self, y, p, groups, base_w, thr=None):
        self.p = np.asarray(p, float)
        self.hit = None if thr is None else (self.p >= np.asarray(thr, float))
        self.r = mt.Ranked(self.p, y)
        self.a, self.b = mt.calib_w(y, self.p, base_w)
        self.citl = mt.citl_w(y, self.p, base_w)
        self.edges = np.unique(np.quantile(self.p, np.linspace(0, 1, 11)))
        self.g = {gn: [(ix, mt.Ranked(self.p[ix], y[ix])) for ix in idxs] for gn, idxs in groups.items()}


def _stat(name: str, v: _Vec, y, w, payload) -> float:
    if name == "auroc":
        return mt.auroc_w(v.r, w)
    if name == "brier":
        return mt.brier_w(y, v.p, w)
    if name == "slope_err":
        return abs(mt.calib_w(y, v.p, w, iters=4, start=(v.a, v.b))[1] - 1)
    if name == "slope":
        return mt.calib_w(y, v.p, w, iters=4, start=(v.a, v.b))[1]
    if name == "citl":
        return _citl_warm(y, v.p, w, v.citl)
    if name == "citl_abs":
        return abs(_citl_warm(y, v.p, w, v.citl))
    if name == "ece":
        return mt.ece_w(y, v.p, w, edges=v.edges)
    if name in ("tpr_sex_gap", "tpr_age_range"):
        gn = "sex" if name == "tpr_sex_gap" else "age"
        vals = [mt.tpr_at_fpr_w(r, payload["fpr"], None if w is None else w[ix]) for ix, r in v.g[gn]]
        if len(vals) < 2:
            return float("nan")
        return float(abs(vals[0] - vals[1])) if gn == "sex" else float(max(vals) - min(vals))
    if name.startswith("gap_"):
        vals = []
        for ix, r in v.g["fair"]:
            wg = None if w is None else w[ix]
            yg = y[ix]
            if name == "gap_auroc":
                vals.append(mt.auroc_w(r, wg))
            elif name == "gap_tpr":
                vals.append(mt.tpr_at_fpr_w(r, payload["fpr"], wg))
            elif name in ("gap_tprthr", "gap_fprthr"):
                ww = np.ones(len(ix)) if wg is None else wg
                h = v.hit[ix]
                pos = yg == 1 if name == "gap_tprthr" else yg == 0
                vals.append(float((ww * h * pos).sum() / max((ww * pos).sum(), 1e-12)))
            elif name == "gap_citl":
                vals.append(_citl_warm(yg, v.p[ix], wg, 0.0, iters=8))
        vals = [x for x in vals if np.isfinite(x)]
        return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")
    if name.startswith("nb@"):
        return mt.net_benefit(y, v.p, float(name[3:]), w)
    raise ValueError(name)


def _citl_warm(y, p, w, start, iters=5):
    z = mt.logit(mt.clip(p))
    w = np.ones(len(y)) if w is None else w
    a = start
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(a + z)))
        a += (w * (y - pr)).sum() / ((w * pr * (1 - pr)).sum() + 1e-12)
    return float(a)


def boot_chunk(payload: dict, B: int, seed: int) -> Dict[Tuple[str, str], np.ndarray]:
    """Run-averaged statistics for B draws (B = 0: the point estimate, as an array of length 1)."""
    y = np.asarray(payload["y"], float)
    base = payload.get("base_w") if payload.get("mode") != "survey" else payload.get("base_w")
    groups = payload.get("groups", {})
    thrs = payload.get("thrs", {})
    vecs = {m: [_Vec(y, p, groups, base, (thrs.get(m) or [None] * len(ps))[i]) for i, p in enumerate(ps)] for m, ps in payload["vecs"].items()}
    stats = payload["stats"]
    if B == 0:
        draws = [base]
    else:
        draws = _weights(payload, B, seed)
    out = {(m, s): [] for m in vecs for s in stats}
    for w in draws:
        for m, vs in vecs.items():
            for s in stats:
                out[(m, s)].append(float(np.mean([_stat(s, v, y, w, payload) for v in vs])))
    return {k: np.asarray(v) for k, v in out.items()}


def bootstrap(payloads: Dict[str, dict], B: int, n_workers: int) -> Dict[str, Dict[str, Dict[Tuple[str, str], np.ndarray]]]:
    """For each unit: {'point': {...}, 'draws': {...}}. Draws are split into N_CHUNKS chunks with fixed seeds."""
    jobs = []
    for u, pl in payloads.items():
        jobs.append((u, "point", pl, 0, 0))
        sizes = [B // N_CHUNKS + (1 if c < B % N_CHUNKS else 0) for c in range(N_CHUNKS)]
        for c, b in enumerate(sizes):
            if b:
                jobs.append((u, c, pl, b, seed_for("boot", u, c)))
    res: Dict[str, Dict] = {u: {"point": None, "draws": []} for u in payloads}
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as ex:
            outs = list(ex.map(_job, jobs))
    else:
        outs = [_job(j) for j in jobs]
    for (u, kind, _, _, _), o in zip(jobs, outs):
        if kind == "point":
            res[u]["point"] = {k: float(v[0]) for k, v in o.items()}
        else:
            res[u]["draws"].append(o)
    for u in res:
        ch = res[u]["draws"]
        res[u]["draws"] = {k: np.concatenate([c[k] for c in ch]) for k in ch[0]} if ch else {}
    return res


def _job(j):
    u, kind, pl, b, seed = j
    return boot_chunk(pl, b, seed)


def ci(draws: np.ndarray) -> Tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))) if len(d) > 20 else (np.nan, np.nan)


# ================================================================== hypothesis families
def groups_for(fr: pd.DataFrame, y: np.ndarray, min_events: int):
    sex = [np.flatnonzero(fr.sex_female.to_numpy() == 1), np.flatnonzero(fr.sex_female.to_numpy() == 0)]
    if any(y[ix].sum() < min_events or (y[ix] == 0).sum() < min_events for ix in sex):
        sex = []                                             # protocol 7: groups need at least min_events events
    bands = []
    for b in ("20-39", "40-59", "60+"):
        ix = np.flatnonzero(fr.age_band.to_numpy() == b)
        if y[ix].sum() >= min_events and (y[ix] == 0).sum() >= min_events:
            bands.append(ix)
    return {"sex": sex, "age": bands}, len(bands)


def endpoint_row(fam, hyp, unit, endpoint, predicted, point_diff, draws_diff, margin=0.0, primary=True, note=""):
    if hyp == "H1":
        p_pred, p_opp = mt.p_one_sided(draws_diff, -margin, +1), mt.p_one_sided(draws_diff, -margin, -1)
    elif predicted == "higher":
        p_pred, p_opp = mt.p_one_sided(draws_diff, 0.0, +1), mt.p_one_sided(draws_diff, 0.0, -1)
    else:
        p_pred, p_opp = mt.p_one_sided(draws_diff, 0.0, -1), mt.p_one_sided(draws_diff, 0.0, +1)
    lo, hi = ci(draws_diff)
    return {"family": fam, "hypothesis": hyp, "unit": unit, "endpoint": endpoint, "predicted": predicted, "primary": primary,
            "difference": point_diff, "ci_lo": lo, "ci_hi": hi, "boot_sd": float(np.nanstd(draws_diff, ddof=1)), "p_pred": p_pred,
            "p_opp": p_opp, "margin": margin, "n_draws": int(np.isfinite(draws_diff).sum()), "note": note}


def verdicts(ep: pd.DataFrame, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    ep = ep.copy()
    ep["p_pred_holm"], ep["p_opp_holm"] = np.nan, np.nan
    for fam, g in ep[ep.primary].groupby("family"):
        ep.loc[g.index, "p_pred_holm"] = mt.holm(g.p_pred.to_numpy())
        ep.loc[g.index, "p_opp_holm"] = mt.holm(g.p_opp.to_numpy())
    half = alpha / 2
    ep["status"] = np.where(~ep.primary, "descriptive", np.where(ep.p_pred_holm < half, "predicted",
                                                                  np.where(ep.p_opp_holm < half, "opposite", "null")))
    for (fam, h), g in ep[ep.primary].groupby(["family", "hypothesis"]):
        st = []
        for _, gu in g.groupby("unit", sort=False):
            s = list(gu.status)
            st.append("predicted" if all(x == "predicted" for x in s) else "opposite" if any(x == "opposite" for x in s) else "null")
        n_u, n_p, n_o = len(st), st.count("predicted"), st.count("opposite")
        v = "PASS" if (n_o == 0 and (n_p == n_u if h == "H1" else n_p > n_u / 2)) else "FAIL" if n_o > n_u / 2 else "NULL"
        out.append({"family": fam, "hypothesis": h, "units": n_u, "units_predicted": n_p, "units_opposite": n_o, "verdict": v})
    return ep, pd.DataFrame(out)


# ================================================================== attribution stability (H4)
def stability_rows(attr_by_seed: List[np.ndarray]) -> np.ndarray:
    """Per row: mean Spearman correlation of attribution vectors over all seed pairs."""
    S = len(attr_by_seed)
    vals = [mt.spearman_rows(attr_by_seed[a], attr_by_seed[b]) for a in range(S) for b in range(a + 1, S)]
    return np.nanmean(np.vstack(vals), axis=0)


def mean_boot(x: Dict[str, np.ndarray], B: int, seed: int) -> Dict[str, np.ndarray]:
    """Bootstrap of row means for several aligned per-row arrays (same rows resampled for all)."""
    n = len(next(iter(x.values())))
    out = {k: [] for k in x}
    for w in mt.boot_weight_iter(n, B, seed):
        for k, v in x.items():
            ok = np.isfinite(v)
            out[k].append(float((w[ok] * v[ok]).sum() / w[ok].sum()))
    return {k: np.asarray(v) for k, v in out.items()}


# ================================================================== simple helpers used by several sections
def run_mean(P: dict, model: str, runs: Sequence[Tuple[int, int]], fn, tag: str = "none") -> float:
    return float(np.mean([fn(P[(model, r, s)][tag]) for r, s in runs]))


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


def fold_aurocs(y, p, fold) -> np.ndarray:
    return np.array([mt.auroc_w(mt.Ranked(p[fold == k], y[fold == k])) for k in np.unique(fold)])


def classify_interval(sign: int, lo: float, hi: float) -> str:
    """'supported' if the whole interval lies on the prior's side of zero, 'contradicted' if wholly on the other side."""
    lo_s, hi_s = sorted((sign * lo, sign * hi))
    return "supported" if lo_s > 0 else "contradicted" if hi_s < 0 else "uncertain"


def empirical_signs(fr: pd.DataFrame, feats: Sequence[str], signs: Dict[str, int]) -> pd.DataFrame:
    """Adjusted logistic regression on the evaluation rows (descriptive, protocol 8c): Wald 95% intervals on the standardised scale;
    and the Spearman correlation between quintile index and observed event rate."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from .prep import Prep
    pp = Prep(feats).fit(fr)
    X = pp.transform(fr)[0].astype(float)
    y = fr.y.to_numpy()
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    Xa = np.c_[np.ones(len(X)), X]
    cov = np.linalg.pinv(Xa.T @ (Xa * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))[1:]
    rows = []
    for f, s in signs.items():
        j = feats.index(f)
        b, lo, hi = lr.coef_[0, j], lr.coef_[0, j] - 1.96 * se[j], lr.coef_[0, j] + 1.96 * se[j]
        obs = fr[f].notna().to_numpy()
        if fr[f].nunique() > 5:
            q = pd.qcut(fr.loc[obs, f], 5, labels=False, duplicates="drop")
            rate = fr.loc[obs].groupby(q).y.mean()
            rho = float(spearmanr(rate.index, rate.values)[0]) if len(rate) > 2 else np.nan
        else:
            rate = fr.loc[obs].groupby(f).y.mean()
            rho = float(np.sign(rate.iloc[-1] - rate.iloc[0])) if len(rate) > 1 else np.nan
        cat = classify_interval(s, lo, hi)
        rows.append({"feature": f, "prior_sign": s, "adjusted_coef": b, "coef_lo": lo, "coef_hi": hi, "adjusted_evidence": cat,
                     "quintile_spearman": rho, "quintile_agrees": bool(np.sign(rho) == s) if np.isfinite(rho) else None})
    return pd.DataFrame(rows)


def conc_job(job):
    """Harrell's or Uno's C for one prediction vector (run in worker processes)."""
    c, m, r, kind, risk, t_, e_, w, tau = job
    return (c, m, r, kind, mt.concordance(risk, t_, e_, event_weight=w, tau=tau))


def parallel_map(fn, items, n_workers):
    if n_workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as ex:
            return list(ex.map(fn, items, chunksize=max(1, len(items) // (4 * n_workers))))
    return [fn(x) for x in items]
