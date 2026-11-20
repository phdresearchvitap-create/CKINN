"""Independent verification of stored CKINN v2 tables from saved out-of-fold predictions.

Uses numpy, pandas and scikit-learn only. Does not import or read the study's own code.
Prints aggregates only (never row-level values).

Usage: python -I verify.py <preds_dir> <tables_dir>
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

PRED = sys.argv[1]
TAB = sys.argv[2]
COHORTS = ["NAGALA", "RICH", "NHANES"]
MAIN = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
SEEDS_R0 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
TOL = 1e-6
TOL_SLOPE = 1e-3
TOL_BOOT = 0.003

n_match = 0
n_mismatch = 0
mismatches = []


def out(s=""):
    print(s, flush=True)


def cmp(label, ours, stored, tol=TOL):
    global n_match, n_mismatch
    try:
        stored_f = float(stored)
    except (TypeError, ValueError):
        stored_f = float("nan")
    diff = abs(ours - stored_f)
    ok = bool(np.isfinite(diff) and diff <= tol)
    tag = "MATCH" if ok else "MISMATCH"
    if ok:
        n_match += 1
    else:
        n_mismatch += 1
        mismatches.append(label)
    out(f"  [{tag}] {label}: ours={ours:.10g} stored={stored_f:.10g} |diff|={diff:.3g} (tol {tol:g})")
    return ok


def read_table(name):
    return pd.read_csv(f"{TAB}/{name}", keep_default_na=False, na_values=[""])


def logit(p, eps=1e-12):
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def cal_slope(y, p):
    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", tol=1e-12, max_iter=100000)
    lr.fit(x, y.astype(int))
    return float(lr.coef_[0, 0])


def cal_slope_newton(y, p, iters=50):
    """Plain Newton-Raphson for logit(P(y=1)) = a + b*logit(p); unpenalised. Cross-check only."""
    x = logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (y - mu)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return float(beta[1])


def load(cohort):
    return np.load(f"{PRED}/{cohort}_runs.npz", allow_pickle=False)


t0 = time.time()
t08 = read_table("table08_performance.csv")
t09 = read_table("table09_hypothesis_endpoints.csv")
t12 = read_table("table12_variance_components.csv")
t22 = read_table("table22_missingness_auroc.csv")
t31 = read_table("table31_transport.csv")

Z = {c: load(c) for c in COHORTS}
Y = {c: Z[c]["y"].astype(np.float64) for c in COHORTS}

# cache of AUROC/Brier per (cohort, model, r, s, tag)
AUC = {}
BRI = {}


def pred(c, m, r, s, tag="none"):
    return Z[c][f"{m}|r{r}|s{s}|{tag}"].astype(np.float64)


def auc(c, m, r, s, tag="none"):
    k = (c, m, r, s, tag)
    if k not in AUC:
        AUC[k] = roc_auc_score(Y[c], pred(c, m, r, s, tag))
    return AUC[k]


def brier(c, m, r, s, tag="none"):
    k = (c, m, r, s, tag)
    if k not in BRI:
        BRI[k] = brier_score_loss(Y[c], pred(c, m, r, s, tag))
    return BRI[k]


# ---------------------------------------------------------------- Check 8 (sanity) first
out("=" * 100)
out("CHECK 8: sanity of saved predictions")
out("=" * 100)
for c in COHORTS:
    z = Z[c]
    y = Y[c]
    n = len(y)
    row = t08[t08.cohort == c]
    n_tab = set(row["n"].astype(int))
    ev_tab = set(row["events"].astype(int))
    out(f"{c}: len(y)={n}, events={int(y.sum())}, labels binary={set(np.unique(y)) <= {0.0, 1.0}}")
    cmp(f"{c} n (len y) vs table08 n", float(n), list(n_tab)[0] if len(n_tab) == 1 else float("nan"), 0)
    cmp(f"{c} events (sum y) vs table08 events", float(y.sum()), list(ev_tab)[0] if len(ev_tab) == 1 else float("nan"), 0)
    pred_keys = [k for k in z.files if k != "y" and not k.startswith(("THRSPEC|", "FOLD|"))]
    bad_len = bad_nan = bad_rng = 0
    gmin, gmax = np.inf, -np.inf
    for k in pred_keys:
        p = z[k]
        if p.shape != (n,):
            bad_len += 1
        if np.isnan(p).any():
            bad_nan += 1
        if (p < 0).any() or (p > 1).any():
            bad_rng += 1
        gmin = min(gmin, float(np.nanmin(p)))
        gmax = max(gmax, float(np.nanmax(p)))
    ok = bad_len == 0 and bad_nan == 0 and bad_rng == 0
    out(f"  [{'MATCH' if ok else 'MISMATCH'}] {c}: {len(pred_keys)} prediction arrays; wrong length={bad_len}, "
        f"with NaN={bad_nan}, outside [0,1]={bad_rng}; global min={gmin:.3g}, max={gmax:.6g}")
    if ok:
        n_match += 1
    else:
        n_mismatch += 1
        mismatches.append(f"{c} prediction sanity")
    # fold ids within each main run
    fold_keys = [k for k in z.files if k.startswith("FOLD|")]
    bad = []
    nf = set()
    for k in fold_keys:
        f = z[k]
        _, m, rr, ss = k.split("|")
        rs = (int(rr[1:]), int(ss[1:]))
        okf = (f.shape == (n,) and np.issubdtype(f.dtype, np.integer) and f.min() >= 0
               and np.isfinite(f).all())
        u = np.unique(f)
        nf.add(len(u))
        if not okf:
            bad.append(k)
    main_fold_keys = [k for k in fold_keys
                      if (int(k.split("|")[2][1:]), int(k.split("|")[3][1:])) in MAIN]
    ok = len(bad) == 0
    out(f"  [{'MATCH' if ok else 'MISMATCH'}] {c}: {len(fold_keys)} fold-id arrays ({len(main_fold_keys)} for main runs); "
        f"each is one integer fold id per row (shape (n,)), none negative/missing; bad={len(bad)}; "
        f"distinct fold counts per array={sorted(nf)}")
    if ok:
        n_match += 1
    else:
        n_mismatch += 1
        mismatches.append(f"{c} fold ids")
    # do models share the same partition within a run? (informative)
    for (r, s) in MAIN:
        ks = [k for k in main_fold_keys if k.endswith(f"|r{r}|s{s}")]
        ref = z[ks[0]]
        same = all(np.array_equal(ref, z[k]) for k in ks[1:])
        sizes = np.bincount(ref)
        ev = np.bincount(ref, weights=y).astype(int)
        out(f"     run r{r}s{s}: {len(ks)} models, identical fold partition across models={same}; "
            f"fold sizes={sizes.tolist()} events per fold={ev.tolist()}")
    # are the main-run partitions different between repeats? (informative)
    parts = [z[f"FOLD|CKINN|r{r}|s{s}"] for (r, s) in MAIN]
    diffs = [int(np.sum(parts[0] != p)) for p in parts[1:]]
    out(f"     CKINN partitions r1..r4 differ from r0 in {diffs} rows (repeats re-split)")
    seeds_same = all(np.array_equal(z["FOLD|CKINN|r0|s0"], z[f"FOLD|CKINN|r0|s{s}"]) for s in range(1, 5))
    out(f"     seed study (r0, s1..s4) uses the same partition as r0s0: {seeds_same}")

zt = np.load(f"{PRED}/transport.npz", allow_pickle=False)
bad = []
for k in zt.files:
    if k.startswith("y_"):
        continue
    p = zt[k]
    tgt = k.split("|")[2] if k.startswith("ext|") else k.split("|")[1]
    if p.shape != zt[f"y_{tgt}"].shape or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        bad.append(k)
ok = len(bad) == 0
out(f"  [{'MATCH' if ok else 'MISMATCH'}] transport.npz: {len(zt.files) - 2} prediction arrays, "
    f"bad (length/NaN/range)={len(bad)}; len(y_NAGALA)={len(zt['y_NAGALA'])} (events {int(zt['y_NAGALA'].sum())}), "
    f"len(y_RICH)={len(zt['y_RICH'])} (events {int(zt['y_RICH'].sum())})")
if ok:
    n_match += 1
else:
    n_mismatch += 1
    mismatches.append("transport sanity")
for c in ["NAGALA", "RICH"]:
    same = len(zt[f"y_{c}"]) == len(Y[c]) and np.array_equal(zt[f"y_{c}"].astype(float), Y[c])
    out(f"     transport y_{c} identical to {c}_runs y: {same}")

# ---------------------------------------------------------------- Check 1: Table 08
out()
out("=" * 100)
out("CHECK 1: table08_performance.csv (run-averaged AUROC, Brier, AUPRC; scaled Brier)")
out("=" * 100)
for c in COHORTS:
    y = Y[c]
    prev = y.mean()
    rows = t08[(t08.cohort == c)]
    for _, row in rows.iterrows():
        m = row["model"]
        runs = int(row["runs"])
        rs_list = MAIN if runs == 5 else [(0, 0)]
        aurocs = [auc(c, m, r, s) for r, s in rs_list]
        briers = [brier(c, m, r, s) for r, s in rs_list]
        auprcs = [average_precision_score(y, pred(c, m, r, s)) for r, s in rs_list]
        a, b, ap = np.mean(aurocs), np.mean(briers), np.mean(auprcs)
        sb = 1 - b / (prev * (1 - prev))
        lab = f"{c} {m}" + ("" if runs == 5 else " [single-run model, r0s0; supplementary]")
        out(f"{lab}:")
        cmp(f"{c} {m} AUROC", a, row["auroc"])
        cmp(f"{c} {m} Brier", b, row["brier"])
        cmp(f"{c} {m} AUPRC", ap, row["auprc"])
        cmp(f"{c} {m} scaled Brier", sb, row["scaled_brier"])

# ---------------------------------------------------------------- Check 2: Table 09
out()
out("=" * 100)
out("CHECK 2: table09_hypothesis_endpoints.csv (H1 AUROC and H2 Brier point estimates)")
out("=" * 100)
for fam, comp in [("A: CKINN-M2", "M2"), ("B: CKINN-M1", "M1")]:
    for c in COHORTS:
        d_auc = np.mean([auc(c, "CKINN", r, s) for r, s in MAIN]) - np.mean([auc(c, comp, r, s) for r, s in MAIN])
        d_bri = np.mean([brier(c, "CKINN", r, s) for r, s in MAIN]) - np.mean([brier(c, comp, r, s) for r, s in MAIN])
        sel = t09[(t09.family == fam) & (t09.unit == f"{c} (T2)")]
        h1 = sel[(sel.hypothesis == "H1")]
        h2 = sel[(sel.hypothesis == "H2") & (sel.endpoint == "Brier score")]
        for _, rw in h1.iterrows():
            cmp(f"{fam} H1 {c} [{rw['endpoint']}] CKINN-{comp} AUROC", d_auc, rw["difference"])
        for _, rw in h2.iterrows():
            cmp(f"{fam} H2 {c} Brier CKINN-{comp}", d_bri, rw["difference"])

# ---------------------------------------------------------------- Check 3: calibration slope
out()
out("=" * 100)
out("CHECK 3: calibration slope (sklearn LogisticRegression C=1e10 on logit(p)), mean over 5 main runs; tol 1e-3")
out("=" * 100)
for c in COHORTS:
    y = Y[c]
    for m in ["CKINN", "M2"]:
        sl = [cal_slope(y, pred(c, m, r, s)) for r, s in MAIN]
        sl_nr = [cal_slope_newton(y, pred(c, m, r, s)) for r, s in MAIN]
        stored = t08[(t08.cohort == c) & (t08.model == m)]["slope"].iloc[0]
        out(f"{c} {m}: per-run slopes (sklearn) = {np.round(sl, 5).tolist()}; "
            f"max |sklearn - Newton| = {np.max(np.abs(np.array(sl) - np.array(sl_nr))):.2e}")
        cmp(f"{c} {m} mean calibration slope", float(np.mean(sl)), stored, TOL_SLOPE)

# ---------------------------------------------------------------- Check 4: Table 31
out()
out("=" * 100)
out("CHECK 4: table31_transport.csv (seed-averaged ref and ext AUROC; transport gap)")
out("=" * 100)
GAP = {}
for src, tgt in [("NAGALA", "RICH"), ("RICH", "NAGALA")]:
    yt = zt[f"y_{tgt}"].astype(np.float64)
    for m in ["M2", "CKINN"]:
        ref = np.mean([roc_auc_score(yt, zt[f"ref|{tgt}|{m}|s{s}"].astype(np.float64)) for s in range(5)])
        ext = np.mean([roc_auc_score(yt, zt[f"ext|{src}|{tgt}|{m}|s{s}"].astype(np.float64)) for s in range(5)])
        GAP[(src, tgt, m)] = ref - ext
        row = t31[(t31.source == src) & (t31.target == tgt) & (t31.model == m)].iloc[0]
        cmp(f"{src}->{tgt} {m} target-CV (ref) AUROC", ref, row["auroc_target_cv"])
        cmp(f"{src}->{tgt} {m} external AUROC", ext, row["auroc_external"])
        cmp(f"{src}->{tgt} {m} transport gap", ref - ext, row["transport_gap"])
    # supplementary: table 09 H3 gap difference CKINN - M2
    d = GAP[(src, tgt, "CKINN")] - GAP[(src, tgt, "M2")]
    rw = t09[(t09.family == "A: CKINN-M2") & (t09.hypothesis == "H3") & (t09.unit == f"{src} -> {tgt}")
             & (t09.endpoint.str.startswith("transport gap"))].iloc[0]
    cmp(f"[supplementary] table09 H3 {src}->{tgt} gap difference CKINN-M2", d, rw["difference"])

# ---------------------------------------------------------------- Check 5: Table 22
out()
out("=" * 100)
out("CHECK 5: table22_missingness_auroc.csv (run-averaged AUROC under mcar30)")
out("=" * 100)
for c in COHORTS:
    for m in ["M2", "CKINN"]:
        a = np.mean([auc(c, m, r, s, "mcar30") for r, s in MAIN])
        row = t22[(t22.cohort == c) & (t22.model == m)].iloc[0]
        cmp(f"{c} {m} AUROC mcar30", a, row["mcar30"])

# ---------------------------------------------------------------- Check 6: Table 12
out()
out("=" * 100)
out("CHECK 6: table12_variance_components.csv (CKINN AUROC SD across seeds of repeat 0 and across main runs)")
out("=" * 100)
for c in COHORTS:
    a_seed = np.array([auc(c, "CKINN", r, s) for r, s in SEEDS_R0])
    a_main = np.array([auc(c, "CKINN", r, s) for r, s in MAIN])
    row = t12[(t12.cohort == c) & (t12.model == "CKINN") & (t12.metric == "auroc")].iloc[0]
    for ddof in (1, 0):
        st = "" if ddof == 1 else " [ddof=0, informative only]"
        if ddof == 1:
            cmp(f"{c} CKINN AUROC SD across seeds r0 (ddof=1)", float(np.std(a_seed, ddof=1)), row["sd_across_seeds_repeat0"])
            cmp(f"{c} CKINN AUROC SD across main runs (ddof=1)", float(np.std(a_main, ddof=1)), row["sd_between_main_runs"])
        else:
            out(f"  {c}{st}: seeds SD={np.std(a_seed, ddof=0):.10g}, runs SD={np.std(a_main, ddof=0):.10g}")

# ---------------------------------------------------------------- Check 7: bootstrap
out()
out("=" * 100)
out("CHECK 7: independent paired percentile bootstrap, NAGALA, run-averaged AUROC CKINN - M2 (1,000 draws, seed 20261120)")
out("=" * 100)
c = "NAGALA"
y = Y[c]
n = len(y)
PC = np.column_stack([pred(c, "CKINN", r, s) for r, s in MAIN])
PM = np.column_stack([pred(c, "M2", r, s) for r, s in MAIN])
point = np.mean([roc_auc_score(y, PC[:, j]) - roc_auc_score(y, PM[:, j]) for j in range(5)])
rng = np.random.default_rng(20261120)
B = 1000
draws = np.empty(B)
i = 0
while i < B:
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if yb.min() == yb.max():
        continue
    draws[i] = np.mean([roc_auc_score(yb, PC[idx, j]) - roc_auc_score(yb, PM[idx, j]) for j in range(5)])
    i += 1
lo, hi = np.percentile(draws, [2.5, 97.5])
row = t09[(t09.family == "A: CKINN-M2") & (t09.hypothesis == "H1") & (t09.unit == "NAGALA (T2)")].iloc[0]
out(f"  point estimate (ours) = {point:.10g}; stored difference = {row['difference']:.10g}")
out(f"  our 95% CI = [{lo:.6f}, {hi:.6f}], bootstrap SD = {draws.std(ddof=1):.6f}, P(diff <= -0.01) = {np.mean(draws <= -0.01):.4f}")
out(f"  stored 95% CI = [{row['ci_lo']:.6f}, {row['ci_hi']:.6f}], stored boot_sd = {row['boot_sd']:.6f}, stored n_draws = {row['n_draws']}")
cmp("bootstrap ci_lo (ours vs stored)", lo, row["ci_lo"], TOL_BOOT)
cmp("bootstrap ci_hi (ours vs stored)", hi, row["ci_hi"], TOL_BOOT)

out()
out("=" * 100)
out(f"SUMMARY: {n_match} comparisons matched, {n_mismatch} mismatched. Runtime {time.time() - t0:.0f}s")
if mismatches:
    out("Mismatches:")
    for m in mismatches:
        out(f"  - {m}")
out("=" * 100)
