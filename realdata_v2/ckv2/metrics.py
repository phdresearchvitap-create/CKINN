"""Metrics, weighted for the bootstrap (protocol_v2.md, sections 6 to 8).

Every metric accepts participant weights w (bootstrap counts; ones for the point estimate). Rank-based metrics take a `Ranked`
object so that the predictions are sorted once and each bootstrap draw costs O(n).
"""
import math
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, t as t_dist

EPS = 1e-6


def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


class Ranked:
    """Predictions sorted once (ascending) with tie groups, for O(n) weighted rank metrics."""

    def __init__(self, p: np.ndarray, y: np.ndarray):
        p = np.asarray(p, float)
        self.order = np.argsort(p, kind="mergesort")
        ps = p[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ps) > 0) + 1]
        self.ys = np.asarray(y, float)[self.order]
        self.n = len(p)

    def groups(self, w: Optional[np.ndarray]):
        ws = np.ones(self.n) if w is None else np.asarray(w, float)[self.order]
        pos = np.add.reduceat(ws * self.ys, self.starts)
        neg = np.add.reduceat(ws * (1 - self.ys), self.starts)
        return pos, neg


def auroc_w(r: Ranked, w: Optional[np.ndarray] = None) -> float:
    """Weighted AUROC with ties counted one half (equals roc_auc_score for unit weights)."""
    pos, neg = r.groups(w)
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def tpr_at_fpr_w(r: Ranked, fpr: float, w: Optional[np.ndarray] = None) -> float:
    """True-positive rate at a given false-positive rate on the (weighted) ROC curve, linear interpolation within tie groups."""
    pos, neg = r.groups(w)
    pos, neg = pos[::-1], neg[::-1]                       # descending score
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    F = np.r_[0.0, np.cumsum(neg) / N]
    T = np.r_[0.0, np.cumsum(pos) / P]
    i = int(np.searchsorted(F, fpr, side="left"))
    if i == 0:
        return 0.0
    if i >= len(F):
        return 1.0
    f0, f1 = F[i - 1], F[i]
    return float(T[i - 1] + (T[i] - T[i - 1]) * ((fpr - f0) / (f1 - f0) if f1 > f0 else 1.0))


def brier_w(y, p, w=None) -> float:
    w = np.ones(len(y)) if w is None else w
    return float((w * (np.asarray(p, float) - y) ** 2).sum() / w.sum())


def calib_w(y, p, w=None, iters: int = 25, start: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    """Logistic recalibration y ~ a + b * logit(p) (weighted Newton). Returns (intercept, slope); ideal (0, 1)."""
    z, y = logit(clip(p)), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    a, b = start
    for _ in range(iters):
        pr = expit(a + b * z)
        v = w * pr * (1 - pr) + 1e-12
        r = w * (y - pr)
        g0, g1 = r.sum(), (r * z).sum()
        h00, h01, h11 = v.sum(), (v * z).sum(), (v * z * z).sum()
        det = h00 * h11 - h01 ** 2
        if det <= 0:
            break
        da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        a, b = a + da, b + db
        if max(abs(da), abs(db)) < 1e-10:
            break
    return float(a), float(b)


def citl_w(y, p, w=None, iters: int = 30) -> float:
    """Calibration-in-the-large: intercept a with slope fixed at 1 (weighted Newton on the offset model)."""
    z, y = logit(clip(p)), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    a = 0.0
    for _ in range(iters):
        pr = expit(a + z)
        g, h = (w * (y - pr)).sum(), (w * pr * (1 - pr)).sum() + 1e-12
        a += g / h
        if abs(g / h) < 1e-12:
            break
    return float(a)


def ece_w(y, p, w=None, bins: int = 10, edges: Optional[np.ndarray] = None) -> float:
    """Expected calibration error with equal-frequency bins (edges from the unweighted predictions)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if edges is None:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    W = np.bincount(b, w, minlength=len(edges) - 1)
    ok = W > 0
    oy = np.bincount(b, w * y, minlength=len(edges) - 1)[ok] / W[ok]
    op = np.bincount(b, w * p, minlength=len(edges) - 1)[ok] / W[ok]
    return float((W[ok] / W.sum() * np.abs(oy - op)).sum())


def threshold_metrics(y, p, thr, w=None) -> Dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    hit = p >= thr
    tp, fp = (w * hit * y).sum(), (w * hit * (1 - y)).sum()
    fn, tn = (w * ~hit * y).sum(), (w * ~hit * (1 - y)).sum()
    return {"sens": tp / max(tp + fn, 1e-12), "spec": tn / max(tn + fp, 1e-12), "ppv": tp / max(tp + fp, 1e-12),
            "npv": tn / max(tn + fn, 1e-12), "f1": 2 * tp / max(2 * tp + fp + fn, 1e-12), "fpr": fp / max(fp + tn, 1e-12)}


def net_benefit(y, p, t: float, w=None) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    hit = p >= t
    return float(((w * hit * y).sum() - (w * hit * (1 - y)).sum() * t / (1 - t)) / w.sum())


def net_benefit_all(y, t: float, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    prev = (w * y).sum() / w.sum()
    return float(prev - (1 - prev) * t / (1 - t))


# ---------------------------------------------------------------- DeLong (paired, fast midrank version)
def _midrank(x):
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_paired(y, p1, p2) -> Dict[str, float]:
    """DeLong test for two correlated AUROCs (Sun and Xu fast algorithm). Returns aucs, difference, SE and two-sided p."""
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    aucs, v10, v01 = [], [], []
    for p in (np.asarray(p1, float), np.asarray(p2, float)):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.r_[p[pos], p[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    S = s10 / m + s01 / n
    d = aucs[0] - aucs[1]
    se = float(np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-300)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(d), "se": se, "p_two_sided": float(2 * norm.sf(abs(d) / se)),
            "se1": float(np.sqrt(S[0, 0])), "se2": float(np.sqrt(S[1, 1]))}


# ---------------------------------------------------------------- censoring-aware metrics
def km_censoring(time, event, w=None):
    """Kaplan-Meier estimate of the censoring survival G(t) (censoring = event 0). Returns (unique times, G right after each time)."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    w = np.ones(len(time)) if w is None else np.asarray(w, float)
    ut = np.unique(time)
    idx = np.searchsorted(ut, time)
    d_cens = np.bincount(idx, w * (event == 0), minlength=len(ut))
    n_at = np.cumsum(np.bincount(idx, w, minlength=len(ut))[::-1])[::-1]
    G = np.cumprod(1 - np.where(n_at > 0, d_cens / np.maximum(n_at, 1e-300), 0))
    return ut, G


def G_at(ut, G, t, left: bool = False):
    """G(t) (right-continuous) or G(t-) when left=True."""
    t = np.atleast_1d(np.asarray(t, float))
    i = np.searchsorted(ut, t, side="left" if left else "right") - 1
    return np.where(i >= 0, G[np.clip(i, 0, len(G) - 1)], 1.0)


def km_risk(time, event, tau: float, w=None) -> float:
    """Kaplan-Meier cumulative incidence 1 - S(tau) of the event."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    w = np.ones(len(time)) if w is None else np.asarray(w, float)
    ut = np.unique(time[time <= tau])
    idx = np.searchsorted(np.unique(time), time)
    allt = np.unique(time)
    d = np.bincount(idx, w * (event == 1), minlength=len(allt))
    n_at = np.cumsum(np.bincount(idx, w, minlength=len(allt))[::-1])[::-1]
    S = np.cumprod(1 - np.where(n_at > 0, d / np.maximum(n_at, 1e-300), 0))
    k = np.searchsorted(allt, tau, side="right") - 1
    return float(1 - S[k]) if k >= 0 else 0.0


def td_auc_weights(time, event, tau: float, ut, G):
    """Cases (event by tau) weighted 1/G(T-), controls (time > tau) weighted 1/G(tau); others weight 0 (cumulative/dynamic AUROC)."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    case = (event == 1) & (time <= tau)
    ctrl = time > tau
    wc = np.where(case, 1.0 / np.maximum(G_at(ut, G, time, left=True), 1e-12), 0.0)
    wk = np.where(ctrl, 1.0 / max(float(G_at(ut, G, tau)[0]), 1e-12), 0.0)
    return case.astype(float), wc + wk


def td_auc(p, time, event, tau: float, ut, G, w=None) -> float:
    lab, iw = td_auc_weights(time, event, tau, ut, G)
    keep = iw > 0
    ww = iw[keep] * (1.0 if w is None else np.asarray(w, float)[keep])
    return auroc_w(Ranked(np.asarray(p)[keep], lab[keep]), ww)


def ipcw_brier(p, time, event, tau: float, ut, G, w=None) -> float:
    """IPCW Brier score at tau for predicted risks p (Graf et al. type estimator)."""
    p, time, event = np.asarray(p, float), np.asarray(time, float), np.asarray(event, int)
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    case = (event == 1) & (time <= tau)
    ctrl = time > tau
    gc = G_at(ut, G, time, left=True)
    gt = float(G_at(ut, G, tau)[0])
    s = np.where(case, (1 - p) ** 2 / np.maximum(gc, 1e-12), 0.0) + np.where(ctrl, p ** 2 / max(gt, 1e-12), 0.0)
    return float((w * s).sum() / w.sum())


class _BIT:
    def __init__(self, n):
        self.n, self.t = n, [0.0] * (n + 1)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & (-i)

    def prefix(self, i):           # sum over ranks < i
        s, j = 0.0, i
        while j > 0:
            s += self.t[j]
            j -= j & (-j)
        return s


def concordance(risk, time, event, event_weight=None, tau: Optional[float] = None) -> float:
    """Harrell's C (event_weight None) or Uno's C (event_weight = 1/G(T)^2, truncated at tau).
    Comparable pair: i has an event and T_j > T_i (Uno: also T_i < tau). Risk ties count one half. O(n log n)."""
    risk, time, event = np.asarray(risk, float), np.asarray(time, float), np.asarray(event, int)
    n = len(risk)
    ranks = np.searchsorted(np.unique(risk), risk)
    nr = int(ranks.max()) + 1
    ew = np.ones(n) if event_weight is None else np.asarray(event_weight, float)
    order = np.argsort(-time, kind="mergesort")            # descending time
    bit_cnt = _BIT(nr)
    num = den = 0.0
    i = 0
    total = 0
    while i < n:
        k = i
        t0 = time[order[i]]
        while k < n and time[order[k]] == t0:
            k += 1
        grp = order[i:k]
        cens = [a for a in grp if event[a] == 0]
        for a in cens:                                       # censored at the same time count as outliving the event (Harrell)
            bit_cnt.add(int(ranks[a]), 1.0)
        total += len(cens)
        for a in grp:                                        # events in this time group vs later times and tied censored
            if event[a] == 1 and (tau is None or time[a] < tau) and total > 0:
                lower = bit_cnt.prefix(int(ranks[a]))
                equal = bit_cnt.prefix(int(ranks[a]) + 1) - lower
                num += ew[a] * (lower + 0.5 * equal)
                den += ew[a] * total
        for a in grp:
            if event[a] == 1:
                bit_cnt.add(int(ranks[a]), 1.0)
                total += 1
        i = k
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------- resampling and tests
def boot_weight_iter(n: int, B: int, seed: int) -> Iterator[np.ndarray]:
    """Participant bootstrap weights (counts of each row in a resample of size n)."""
    rng = np.random.default_rng(seed)
    for _ in range(B):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def cluster_weight_iter(cluster: np.ndarray, B: int, seed: int) -> Iterator[np.ndarray]:
    """Cluster bootstrap: resample clusters with replacement; each row gets its cluster's count."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cluster, return_inverse=True)
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(np.float64)
        yield cnt[inv]


def strata_psu_weight_iter(strata: np.ndarray, psu: np.ndarray, base_w: np.ndarray, B: int, seed: int) -> Iterator[np.ndarray]:
    """Survey bootstrap: within each stratum resample PSUs (n_h - 1 of n_h, Rao-Wu rescaling); returns replicate weights."""
    rng = np.random.default_rng(seed)
    key = np.char.add(np.asarray(strata).astype(str), "|" + np.asarray(psu).astype(str))
    ukey, inv = np.unique(key, return_inverse=True)
    st_of = np.array([k.split("|")[0] for k in ukey])
    ust = np.unique(st_of)
    members = {s: np.flatnonzero(st_of == s) for s in ust}
    for _ in range(B):
        mult = np.zeros(len(ukey))
        for s, ix in members.items():
            nh = len(ix)
            if nh < 2:
                mult[ix] = 1.0
                continue
            pick = rng.integers(0, nh, nh - 1)
            c = np.bincount(pick, minlength=nh)
            mult[ix] = c * nh / (nh - 1)
        yield base_w * mult[inv]


def p_one_sided(draws: np.ndarray, null: float, direction: int) -> float:
    """Percentile-bootstrap one-sided p for the alternative 'difference > null' (direction +1) or '< null' (-1):
    (1 + number of draws on the null side) / (B + 1)."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    k = (d <= null).sum() if direction > 0 else (d >= null).sum()
    return float((1 + k) / (len(d) + 1))


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out, ok = np.full(len(p), np.nan), np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok], kind="mergesort")]
    adj = np.minimum(1.0, (len(order) - np.arange(len(order))) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def corrected_t(diffs: np.ndarray, n_test: float, n_train: float) -> Dict[str, float]:
    """Nadeau-Bengio corrected resampled t-test over the R x K fold differences."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1) * (1.0 / J + n_test / n_train)
    se = math.sqrt(max(var, 1e-300))
    tt = d.mean() / se
    q = t_dist.ppf(0.975, J - 1)
    return {"mean": float(d.mean()), "se": se, "lo": float(d.mean() - q * se), "hi": float(d.mean() + q * se),
            "p_two_sided": float(2 * t_dist.sf(abs(tt), J - 1)), "df": J - 1}


def spearman_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two attribution matrices (average ranks for ties)."""
    from scipy.stats import rankdata
    ra, rb = rankdata(np.asarray(A, float), axis=1), rankdata(np.asarray(B, float), axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)
