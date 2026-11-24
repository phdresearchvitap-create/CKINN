import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize, brentq
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from data import status


class ReverseKM:

    def __init__(self, t, e):
        t = np.asarray(t, float)
        e = np.asarray(e)
        if len(t) == 0 or not np.isfinite(t).all() or (t <= 0).any() or (not np.isin(e, [0, 1]).all()):
            raise ValueError("Invalid survival data")
        self.times, idx, n = np.unique(t, return_inverse=True, return_counts=True)
        deaths = np.bincount(idx, weights=e, minlength=len(n))
        risk = len(t) - np.r_[0, np.cumsum(n)[:-1]]
        fraction = np.divide(n - deaths, risk - deaths, out=np.zeros(len(n)), where=risk > deaths)
        self.G = np.r_[1.0, np.cumprod(1.0 - fraction)]

    def at(self, q, left=False):
        return self.G[np.searchsorted(self.times, q, side="left" if left else "right")]


def weights(d, by_site=False):
    if "time" not in d:
        return (d.event.to_numpy(float), np.ones(len(d)), np.nan)
    st = status(d)
    y = (st == 1).astype(float)
    w = np.zeros(len(d))
    km = ReverseKM(d.time, d.event)
    groups = [np.arange(len(d))]
    if by_site:
        sites = d.site.fillna("missing").astype(str)
        tails = (d.time > 3).groupby(sites).sum()
        if tails.max() < 30:
            raise ValueError("No site with 30 observed beyond three years")
        pooled = sites.where(sites.map(tails) >= 30, tails.idxmax())
        groups = d.groupby(pooled).indices.values()
    for inds in groups:
        part = d.iloc[inds]
        g = ReverseKM(part.time, part.event) if by_site else km
        if g.at(3) <= 0 or not (part.time > 3).any():
            raise ValueError("No three-year censoring support")
        cases = inds[st[inds] == 1]
        w[cases] = 1 / g.at(d.time.to_numpy()[cases], left=True)
        w[inds[st[inds] == 0]] = 1 / g.at(3)
    return (y, w, float(km.at(3)))


def brier(y, w, p):
    return float(np.mean(w * np.square(y - np.asarray(p, float))))


def support(d):
    y, w, g = weights(d)
    known = w > 0
    return dict(
        n=len(y),
        events=int((y == 1).sum()),
        controls=int(sum(known & (y == 0))),
        unknown=int(sum(~known)),
        G3=g,
        at_risk_3=int((d.time >= 3).sum()) if "time" in d else None,
        ESS=w.sum() ** 2 / (w @ w),
        positive_w_min=w[known].min(),
        positive_w_median=np.median(w[known]),
        positive_w_p99=np.quantile(w[known], 0.99),
        w_max=w.max(),
    )


def point_threshold(y, w, p):
    c = (y == 0) & (w > 0)
    if not c.any():
        raise ValueError("No known controls to choose threshold")
    v = np.argsort(p[c], kind="stable")
    k = np.searchsorted(np.cumsum(w[c][v]), 0.9 * w[c].sum())
    return float(np.nextafter(float(p[c][v[k]]), np.inf))


def calibration(y, w, p):
    l = logit(np.clip(p, 1e-07, 1 - 1e-07))
    if np.unique(y[w > 0]).size < 2:
        return (np.nan, np.nan)
    intercept = brentq(lambda a: np.sum(w * (expit(a + l) - y)), -50, 50)
    if np.ptp(l[w > 0]) < 1e-10:
        return (intercept, np.nan)
    X = np.column_stack([np.ones(len(y)), l])

    def loss(beta):
        eta = X @ beta
        return (np.average(np.logaddexp(0, eta) - y * eta, weights=w), X.T @ (w * (expit(eta) - y)) / w.sum())

    opt = minimize(loss, [0, 1], jac=True, method="BFGS", options={"gtol": 1e-08})
    return (intercept, opt.x[1] if opt.success and abs(opt.x[1]) < 100 else np.nan)


def calbins(y, w, p):
    p = np.asarray(p)
    labels = np.searchsorted(np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1])), p, side="left")
    rows = []
    for k in np.unique(labels):
        pick = labels == k
        rows.append(
            dict(
                bin=int(k),
                n=int(pick.sum()),
                predicted=p[pick].mean(),
                observed=np.average(y[pick], weights=w[pick]) if w[pick].sum() > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def metrics(d, p, threshold, by_site=False):
    p = np.asarray(p, dtype=float)
    if len(p) != len(d) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    y, w, g = weights(d, by_site)
    k = w > 0
    s = p >= threshold
    tn, fp, fn, tp = [w[(y == a) & (s == b)].sum() for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    def ratio(a, b):
        return a / b if b else np.nan

    two = np.unique(y[k]).size == 2
    ci, cs = calibration(y, w, p)
    cb = calbins(y, w, p)
    return dict(
        **support(d),
        brier=brier(y, w, p),
        auroc=roc_auc_score(y[k], p[k], sample_weight=w[k]) if two else np.nan,
        average_precision=average_precision_score(y[k], p[k], sample_weight=w[k]) if two else np.nan,
        predicted=p.mean(),
        observed=np.mean(w * y),
        cal_intercept=ci,
        cal_slope=cs,
        ece=np.average(abs(cb.predicted - cb.observed), weights=cb.n),
        threshold=threshold if np.ndim(threshold) == 0 else "fold-specific",
        sensitivity=ratio(tp, tp + fn),
        specificity=ratio(tn, tn + fp),
        ppv=ratio(tp, tp + fp),
        npv=ratio(tn, tn + fn),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        fpr=ratio(fp, fp + tn),
        fnr=ratio(fn, fn + tp)
    )


def bootstrap(d, p1, p0, B, seed, threshold=None, all_metrics=False):
    rng = np.random.default_rng(seed)
    vals = []
    failures = 0
    for _ in range(B):
        i = rng.integers(len(d), size=len(d))
        dd = d.iloc[i].reset_index(drop=True)
        try:
            y, w, _ = weights(dd)
            if np.unique(y[w > 0]).size < 2:
                raise ValueError("single class")
            if all_metrics:
                vals.append(metrics(dd, p1[i], threshold if np.ndim(threshold) == 0 else threshold[i]))
            else:
                vals.append(dict(delta=brier(y, w, p1[i]) - brier(y, w, p0[i])))
        except (ValueError, FloatingPointError):
            failures += 1
    frame = pd.DataFrame(vals)
    result = {}
    for k in (
        [
            "brier",
            "auroc",
            "average_precision",
            "cal_intercept",
            "cal_slope",
            "ece",
            "predicted",
            "observed",
            "sensitivity",
            "specificity",
            "fpr",
            "fnr",
            "ppv",
            "npv",
            "f1",
        ]
        if all_metrics
        else ["delta"]
    ):
        v = frame[k].dropna() if k in frame else []
        lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) >= max(2, 0.8 * B) else (np.nan, np.nan)
        result.update({k + "_lo": lo, k + "_hi": hi, k + "_valid_boot": len(v)})
    return dict(**result, requested_boot=B, failed_boot=failures)
