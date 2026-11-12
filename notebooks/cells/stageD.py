# %%
# CELL 22 - preprocessing fitted on training rows only, and every data split of the protocol      [status L; protocol sections 5, 8, 12]
# Why: test data must never inform how missing values are filled or how variables are scaled, and every analysis must
# use exactly the splits the protocol names. All tests in stage D run on small hand-made toy arrays, never on study data.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 22)
require_gate("C")
A = Audit(22, "Preprocessing and data splits")
from sklearn.model_selection import StratifiedKFold


def toy_survival(n=300, ties=True):
    """Deterministic toy survival data (not study data). It uses no random generator and only + - * /, so every
    machine produces exactly the same numbers. x1 raises the hazard, x2 lowers it; about half are censored."""
    i = np.arange(1, n + 1, dtype=float)
    u1, u2, u3, u4 = [(i * a) % 1.0 for a in (0.6180339887, 0.7548776662, 0.5698402910, 0.4142135623)]
    t_event = 4.0 * (u3 / (1.02 - u3)) / ((0.5 + 1.5 * u1) * (1.25 - 0.5 * u2))
    t_cens = 1.0 + 5.0 * u4
    time = np.minimum(t_event, t_cens)
    if ties:
        time = np.maximum(np.round(time, 1), 0.1)
    return pd.DataFrame({"x1": 2 * u1 - 1, "x2": 2 * u2 - 1, "time_y": time, "event": (t_event <= t_cens).astype(int)})


CATEGORICAL = {"sex_male", "famhx", "exercise", "fatty_liver", "smoke", "drink"}     # filled with the training mode; the rest with the median


class Prep:
    """Protocol section 5. Fitted on training rows only: fill value (median, or mode for categorical variables), a
    missingness indicator for every column that has a missing value in training, then standardization."""

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, df):
        X = df[self.cols]
        self.fill = pd.Series({c: (X[c].mode().iloc[0] if c in CATEGORICAL else X[c].median()) for c in self.cols})
        self.ind = [c for c in self.cols if X[c].isna().any()]
        Xf = X.fillna(self.fill)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        self.names = self.cols + [c + "_missing" for c in self.ind]
        return self

    def transform(self, df):
        X = df[self.cols]
        out = ((X.fillna(self.fill) - self.mu) / self.sd).to_numpy(np.float64)
        if self.ind:
            out = np.hstack([out, X[self.ind].isna().to_numpy(np.float64)])
        return out


def cv_folds(df, k=5, seed=0):
    """Stratified k-fold on the event indicator, shuffled with the given seed. Yields (train, test) data frames."""
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(df, df["event"]):
        yield df.iloc[tr], df.iloc[te]


def repeated_cv(df, repeats=None, k=None):
    """R1: yields (repeat, fold, model_seed, train, test). CV seed = repeat; model seed = 1000 x repeat + fold (section 12)."""
    for rep in range(EVAL["cv_repeats"] if repeats is None else repeats):
        for fold, (tr, te) in enumerate(cv_folds(df, EVAL["cv_folds"] if k is None else k, rep)):
            yield rep, fold, 1000 * rep + fold, tr, te


def loso_folds(df):
    """R2: leave one site out. Tiny sites (0, 16) and site 14 (deviation D-1) are always in training, never tested."""
    for s in sorted(df["site"].unique()):
        if s in SMALL_SITES or s in NO_TEST_SITES:
            continue
        yield int(s), df[df["site"] != s], df[df["site"] == s]


def shift_split(df, var, cut):
    """R4: train below the cut, test at or above it (age 45; BMI 25)."""
    return df[df[var] < cut], df[df[var] >= cut]


SHIFT_SPLITS = [("age", 45), ("bmi", 25)]


def inner_split(df, seed):
    """Inner 80/20 split of a training set, used only to choose lambda (section 6). Returns (fit part, validation part)."""
    return next(cv_folds(df, 5, seed + 1000))


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_d["site"] = (np.arange(300) % 6) * 2 + np.where(np.arange(300) % 6 == 5, 4, 0)       # toy site codes 0, 2, 4, 6, 8, 14
_d.loc[::7, "x2"] = np.nan
_d["smoke"] = (np.arange(300) % 4 == 0) * 2.0
_d.loc[5::11, "smoke"] = np.nan
A.check("toy data are identical on every machine (300 rows, 150 events, 129 by 3 years, 60 distinct times)",
        (len(_d), int(_d["event"].sum()), int(((_d["event"] == 1) & (_d["time_y"] <= 3)).sum()), _d["time_y"].nunique()) == (300, 150, 129, 60)
        and abs(float(_d["x1"].iloc[0]) - 0.2360679774) < 1e-9)
_tr, _te = _d.iloc[:200], _d.iloc[200:].copy()
_p = Prep(["x1", "x2", "smoke"]).fit(_tr)
_Z = _p.transform(_tr)
A.check("fill values: training median for continuous, training mode for categorical",
        _p.fill["x2"] == _tr["x2"].median() and _p.fill["smoke"] == 0.0 and _p.fill["x1"] == _tr["x1"].median())
A.check("a missingness indicator is added for each column with a missing value in training, and only for those",
        _p.names == ["x1", "x2", "smoke", "x2_missing", "smoke_missing"] and _Z.shape == (200, 5)
        and (_Z[:, 3] == _tr["x2"].isna().to_numpy()).all())
A.check("standardized training columns have mean 0 and SD 1; nothing is left missing",
        np.allclose(_Z[:, :3].mean(0), 0, atol=1e-12) and np.allclose(_Z[:, :3].std(0, ddof=1), 1, atol=1e-12) and not np.isnan(_Z).any())
_before = (_p.fill.copy(), _p.mu.copy(), _p.sd.copy())
_te.loc[:, "x1"] = _te["x1"] * 50 + 7                         # change the test rows drastically
_p.transform(_te)
A.check("no leakage: changing the test rows leaves every fitted statistic unchanged",
        _p.fill.equals(_before[0]) and _p.mu.equals(_before[1]) and _p.sd.equals(_before[2]))
A.check("a statistic fitted on training rows differs from one fitted on all rows (so the separation matters)",
        Prep(["x1"]).fit(pd.concat([_tr, _te])).mu["x1"] != _p.mu["x1"])
_p1 = Prep(["x1"]).fit(_tr)
_te.loc[_te.index[:3], "x1"] = np.nan
A.check("a value missing only at test time is filled with the training value, without creating a new column",
        _p1.transform(_te).shape == (100, 1) and np.allclose(_p1.transform(_te)[:3, 0], (_p1.fill["x1"] - _p1.mu["x1"]) / _p1.sd["x1"]))

_folds = list(cv_folds(_d, 5, 0))
_test_idx = [set(te.index) for _, te in _folds]
A.check("5-fold: test folds are disjoint, cover every row once, and train and test never overlap",
        sum(len(s) for s in _test_idx) == 300 and len(set().union(*_test_idx)) == 300 and all(not (set(tr.index) & set(te.index)) for tr, te in _folds))
A.check("5-fold: stratified, each test fold holds 30 of the 150 events", [int(te["event"].sum()) for _, te in _folds] == [30] * 5)
A.check("5-fold: the same seed gives the same folds, another seed gives different ones",
        [list(te.index) for _, te in cv_folds(_d, 5, 0)] == [list(te.index) for _, te in _folds]
        and [list(te.index) for _, te in cv_folds(_d, 5, 1)] != [list(te.index) for _, te in _folds])
_rc = [(r, f, s) for r, f, s, _a, _b in repeated_cv(_d)]
A.check("repeated CV: 3 repeats x 5 folds = 15 units; model seed = 1000 x repeat + fold",
        len(_rc) == 15 and all(s == 1000 * r + f for r, f, s in _rc) and _rc[0] == (0, 0, 0) and _rc[-1] == (2, 4, 2004))
_lo = [(s, set(tr["site"]), set(te["site"])) for s, tr, te in loso_folds(_d)]
A.check("leave-one-site-out: sites 0, 16 and 14 are never test sites; the test site is absent from training",
        [s for s, _a, _b in _lo] == [2, 4, 6, 8] and all(te == {s} and s not in tr and {0, 14} <= tr for s, tr, te in _lo))
_a, _b = shift_split(_d, "x1", 0.0)
A.check("shift split: training below the cut, test at or above it, nothing lost",
        (_a["x1"] < 0).all() and (_b["x1"] >= 0).all() and len(_a) + len(_b) == 300 and SHIFT_SPLITS == [("age", 45), ("bmi", 25)])
_fi, _va = inner_split(_tr, 0)
A.check("inner split: 80% to fit, 20% to choose lambda, disjoint, stratified", (len(_fi), len(_va)) == (160, 40) and not (set(_fi.index) & set(_va.index))
        and abs(_va["event"].mean() - _tr["event"].mean()) < 0.03)
_ = A.close(outputs={"toy": {"n": 300, "events": 150}, "categorical": sorted(CATEGORICAL)})
keep_for_session(22)

# %%
# CELL 23 - Cox partial likelihood, baseline hazard and absolute risk                              [status L; protocol section 6; conditional risk: A1.7]
# Why: this is the estimand of the whole study. risk(3 | x) = 1 - exp(-H0(3) exp(lp(x))), with H0 the Breslow baseline
# cumulative hazard from training data. The code is checked against a hand calculation and against reference values
# produced once with the lifelines package (0.30.3) on the same toy data.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 23)
A = Audit(23, "Cox partial likelihood, Breslow hazard, absolute risk")


def cox_nll_grad_hess(beta, X, t, e, penalizer=0.0, pen_scale=None, need_hess=True):
    """Negative log partial likelihood (Efron's correction for tied event times), divided by n, plus the ridge penalty
    0.5 * penalizer * sum_j (pen_scale_j * beta_j)^2. Returns value, gradient and Hessian with respect to beta."""
    n, p = X.shape
    pen_scale = np.ones(p) if pen_scale is None else np.asarray(pen_scale, dtype=float)
    o = np.argsort(-t, kind="stable")                    # descending time: the risk set of a time is every row up to its block end
    X, t, e = X[o], t[o], e[o].astype(float)
    lp = X @ beta
    lp = lp - lp.max()
    r = np.exp(lp)
    change = np.flatnonzero(np.diff(t) != 0)
    starts, ends = np.append(0, change + 1), np.append(change, n - 1)
    d_all = np.add.reduceat(e, starts)
    kb = np.flatnonzero(d_all > 0)                       # tie blocks that contain at least one event
    d = d_all[kb]
    rX, re = r[:, None] * X, r * e
    s0, s1 = np.cumsum(r)[ends[kb]], np.cumsum(rX, axis=0)[ends[kb]]
    e0, e1 = np.add.reduceat(re, starts)[kb], np.add.reduceat(re[:, None] * X, starts, axis=0)[kb]
    xsum, lpsum = np.add.reduceat(e[:, None] * X, starts, axis=0)[kb], np.add.reduceat(e * lp, starts)[kb]
    if need_hess:
        s2, e2, run, prev = np.empty((len(kb), p, p)), np.empty((len(kb), p, p)), np.zeros((p, p)), 0
        for j, k in enumerate(kb):
            run = run + rX[prev:ends[k] + 1].T @ X[prev:ends[k] + 1]
            prev = ends[k] + 1
            s2[j] = run
            b = slice(starts[k], ends[k] + 1)
            e2[j] = (re[b][:, None] * X[b]).T @ X[b]
    nll, grad, hess = -lpsum.sum(), -xsum.sum(0), np.zeros((p, p))
    for l in range(int(d.max())):                        # Efron: the l-th of d tied events sees the risk set minus l/d of the tied events
        m = d > l
        f = l / d[m]
        w0 = s0[m] - f * e0[m]
        w1 = s1[m] - f[:, None] * e1[m]
        mu = w1 / w0[:, None]
        nll += np.log(w0).sum()
        grad += mu.sum(0)
        if need_hess:
            w2 = s2[m] - f[:, None, None] * e2[m]
            hess += (w2 / w0[:, None, None] - mu[:, :, None] * mu[:, None, :]).sum(0)
    pen = penalizer * pen_scale ** 2
    return nll / n + 0.5 * (pen * beta * beta).sum(), grad / n + pen * beta, hess / n + np.diag(pen)


def cox_newton(X, t, e, penalizer=0.0, pen_scale=None, max_iter=50, tol=1e-9):
    """Newton-Raphson with step halving. Returns (beta, objective, iterations)."""
    X, t, e = np.asarray(X, dtype=float), np.asarray(t, dtype=float), np.asarray(e, dtype=float)
    beta = np.zeros(X.shape[1])
    nll, g, Hm = cox_nll_grad_hess(beta, X, t, e, penalizer, pen_scale)
    for it in range(max_iter):
        step, s = np.linalg.solve(Hm, g), 1.0
        while True:
            nb = beta - s * step
            nll2 = cox_nll_grad_hess(nb, X, t, e, penalizer, pen_scale, need_hess=False)[0]
            if nll2 <= nll + 1e-12 or s < 1e-4:
                break
            s *= 0.5
        done = np.abs(nb - beta).max() < tol
        beta = nb
        nll, g, Hm = cox_nll_grad_hess(beta, X, t, e, penalizer, pen_scale)
        if done:
            break
    return beta, nll, it + 1


def cox_slope(z, t, e):
    """Coefficient of a one-variable unpenalized Cox model (used for the calibration slope and for recalibration)."""
    z = np.asarray(z, dtype=float)
    sd = z.std(ddof=1) or 1.0
    b, _nll, _it = cox_newton(((z - z.mean()) / sd)[:, None], t, e)
    return float(b[0] / sd)


class Breslow:
    """Breslow baseline cumulative hazard H0(t), estimated from training times, events and log relative hazards."""

    def __init__(self, time, event, lp):
        o = np.argsort(time, kind="stable")
        t, e, r = np.asarray(time, dtype=float)[o], np.asarray(event)[o], np.exp(np.clip(np.asarray(lp, dtype=float)[o], -20, 20))
        at_risk = np.cumsum(r[::-1])[::-1]
        ut, cnt = np.unique(t[e == 1], return_counts=True)
        self.t, self.H = ut, np.cumsum(cnt / at_risk[np.searchsorted(t, ut, side="left")])

    def __call__(self, h):
        i = np.searchsorted(self.t, h, side="right") - 1
        return 0.0 if i < 0 else float(self.H[i])


def risk_at(H0, lp, h=H, h_from=None):
    """Absolute risk by time h: 1 - exp(-H0(h) exp(lp)). With h_from, the conditional risk between h_from and h among
    people event-free at h_from: 1 - exp(-(H0(h) - H0(h_from)) exp(lp))  (used only by amendment item A1.7)."""
    base = H0(h) - (H0(h_from) if h_from is not None else 0.0)
    return 1.0 - np.exp(-base * np.exp(np.clip(np.asarray(lp, dtype=float), -20, 20)))


class Fitted:
    """A fitted model together with the Breslow baseline of its training data: gives absolute risk for new rows."""

    def __init__(self, model, train):
        self.model, self.train = model, train
        self.H0 = Breslow(train["time_y"].to_numpy(), train["event"].to_numpy(), model.lp(train))

    def risk(self, df, h=H, h_from=None):
        return risk_at(self.H0, self.model.lp(df), h, h_from)


def fit_with_baseline(model, train):
    return Fitted(model.fit(train), train)


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]
_X = _tr[["x1", "x2"]].to_numpy()
_mu, _sd = _X.mean(0), _X.std(0, ddof=1)
_Z, _t, _e = (_X - _mu) / _sd, _tr["time_y"].to_numpy(), _tr["event"].to_numpy()

_v = cox_nll_grad_hess(np.zeros(1), np.zeros((3, 1)), np.array([1.0, 1.0, 2.0]), np.ones(3))[0] * 3
A.check("Efron tie correction equals a hand calculation: times 1, 1, 2, all events, gives log(3) + log(2) + log(1)", abs(_v - np.log(6.0)) < 1e-12, round(float(_v), 6))
_b0 = np.array([0.3, -0.2])
_f0, _g0, _H0m = cox_nll_grad_hess(_b0, _Z, _t, _e, 0.01)
_gn = np.array([(cox_nll_grad_hess(_b0 + dk, _Z, _t, _e, 0.01)[0] - cox_nll_grad_hess(_b0 - dk, _Z, _t, _e, 0.01)[0]) / 2e-6 for dk in np.eye(2) * 1e-6])
_Hn = np.array([(cox_nll_grad_hess(_b0 + dk, _Z, _t, _e, 0.01)[1] - cox_nll_grad_hess(_b0 - dk, _Z, _t, _e, 0.01)[1]) / 2e-6 for dk in np.eye(2) * 1e-6])
A.check("analytic gradient equals the numerical derivative of the likelihood", np.allclose(_g0, _gn, atol=1e-8), float(np.abs(_g0 - _gn).max()))
A.check("analytic Hessian equals the numerical derivative of the gradient", np.allclose(_H0m, _Hn, atol=1e-7), float(np.abs(_H0m - _Hn).max()))

REF_COX = {"unpenalized": [0.48966514, -0.32296458], "ridge 0.01": [0.47960590, -0.31679536]}       # lifelines 0.30.3 on the same toy rows
_fit = {}
for name, pen in (("unpenalized", 0.0), ("ridge 0.01", 0.01)):
    b, nll, it = cox_newton(_Z, _t, _e, pen)
    _fit[name] = b / _sd
    A.check(f"Cox coefficients, {name}: equal to lifelines to 6 decimals", np.allclose(_fit[name], REF_COX[name], atol=1e-6),
            f"{np.round(_fit[name], 8).tolist()} in {it} iterations")
A.check("the ridge penalty shrinks both coefficients towards zero", (np.abs(_fit["ridge 0.01"]) < np.abs(_fit["unpenalized"])).all())
A.check("x1 raises the hazard and x2 lowers it, as the toy data were built", _fit["unpenalized"][0] > 0 > _fit["unpenalized"][1])

BETA_TOY = np.array([0.47960590, -0.31679536])                # fixed coefficients used by the metric tests of cells 23 to 28
_lp_tr, _lp_te = _tr[["x1", "x2"]].to_numpy() @ BETA_TOY, _te[["x1", "x2"]].to_numpy() @ BETA_TOY
_H = Breslow(_t, _e, _lp_tr)
_r3 = risk_at(_H, _lp_te, 3.0)
A.check("baseline cumulative hazard starts at 0 and never decreases", _H(0.0) == 0.0 and _H(1) <= _H(3) <= _H(6) and (np.diff(_H.H) > 0).all())
A.check("3-year risks lie strictly between 0 and 1 and rise with the log relative hazard", ((_r3 > 0) & (_r3 < 1)).all() and (np.diff(_r3[np.argsort(_lp_te)]) >= 0).all())
A.check("mean 3-year risk on the toy test rows equals the lifelines prediction (0.482906)", abs(_r3.mean() - 0.4829061392600477) < 1e-6, round(float(_r3.mean()), 6))
A.check("risk is invariant to adding a constant to lp (the baseline absorbs it)", np.allclose(risk_at(Breslow(_t, _e, _lp_tr + 5.0), _lp_te + 5.0, 3.0), _r3, atol=1e-12))
_r2, _rc = risk_at(_H, _lp_te, 2.0), risk_at(_H, _lp_te, 3.0, h_from=2.0)
A.check("conditional risk between 2 and 3 years equals 1 - S(3)/S(2)", np.allclose(_rc, 1 - (1 - _r3) / (1 - _r2), atol=1e-12) and (_rc < _r3).all())


class _ToyModel:
    cols = ["x1", "x2"]

    def fit(self, df):
        return self

    def lp(self, df):
        return df[["x1", "x2"]].to_numpy() @ BETA_TOY


A.check("fit_with_baseline returns the same risks as the direct calculation", np.allclose(fit_with_baseline(_ToyModel(), _tr).risk(_te), _r3, atol=1e-15))
# live check against lifelines installed on this machine (in addition to the stored reference values above)
import lifelines
from lifelines import CoxPHFitter
_ll = {name: CoxPHFitter(penalizer=pen).fit(_tr[["x1", "x2", "time_y", "event"]], "time_y", "event") for name, pen in (("unpenalized", 0.0), ("ridge 0.01", 0.01))}
A.check(f"live: lifelines {lifelines.__version__} on this machine gives the same Cox coefficients (6 decimals) and log-likelihood",
        all(np.allclose(_fit[k], _ll[k].params_.to_numpy(), atol=1e-6) for k in _ll)
        and abs(_ll["unpenalized"].log_likelihood_ + cox_newton(_Z, _t, _e, 0.0)[1] * len(_tr)) < 1e-6)
_m = _ll["ridge 0.01"]
_risk_ll = 1 - _m.predict_survival_function(_te[["x1", "x2"]], times=[3.0]).to_numpy().ravel()
_risk_me = risk_at(Breslow(_t, _e, _tr[["x1", "x2"]].to_numpy() @ _m.params_.to_numpy()), _te[["x1", "x2"]].to_numpy() @ _m.params_.to_numpy(), 3.0)
A.check("live: Breslow baseline and 3-year risk equal lifelines' predicted survival", np.allclose(_risk_ll, _risk_me, atol=1e-10), float(np.abs(_risk_ll - _risk_me).max()))
_ = A.close(outputs={"cox_toy": {k: v.tolist() for k, v in _fit.items()}, "reference": f"lifelines 0.30.3 (stored); lifelines {lifelines.__version__} (live)"})
keep_for_session(23)

# %%
# CELL 24 - Kaplan-Meier, follow-up, and inverse-probability-of-censoring weights                  [status L; protocol section 9]
# Why: every 3-year metric needs the observed risk (Kaplan-Meier) and, because half of RICH is censored before 3 years,
# weights that make the participants with known 3-year status stand in for the censored ones. Nobody censored early is
# ever counted as a non-event: their weight is zero and the others are weighted up.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 24)
A = Audit(24, "Kaplan-Meier, censoring distribution, censoring weights")


def km_survival(time, event, at):
    """Kaplan-Meier probability of being event-free at each time in `at` (array in, array out)."""
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq, first = np.unique(t, return_index=True)
    surv = np.cumprod(1.0 - np.add.reduceat(e, first) / (len(t) - first))
    pos = np.searchsorted(uniq, np.atleast_1d(np.asarray(at, dtype=float)), side="right") - 1
    return np.where(pos >= 0, surv[np.clip(pos, 0, None)], 1.0)


def km_risk(df, h=H):
    """Observed risk by time h (1 - Kaplan-Meier). NaN when it cannot be estimated: under 20 rows, no event, or no follow-up to h."""
    if len(df) < 20 or df["time_y"].max() < h or df["event"].sum() == 0:
        return float("nan")
    return float(1.0 - km_survival(df["time_y"], df["event"], h)[0])


def censoring_survival(time, event, at):
    """G(t): Kaplan-Meier probability of still being uncensored at t. At a tied time, events are taken to happen before
    censorings (the convention of scikit-survival), so an event at t is not at risk of censoring at t."""
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq, first = np.unique(t, return_index=True)
    d, c = np.add.reduceat(e, first), np.add.reduceat(1.0 - e, first)
    at_risk = (len(t) - first) - d
    G = np.cumprod(np.where(at_risk > 0, 1.0 - c / np.maximum(at_risk, 1), 1.0))
    pos = np.searchsorted(uniq, np.atleast_1d(np.asarray(at, dtype=float)), side="right") - 1
    return np.where(pos >= 0, G[np.clip(pos, 0, None)], 1.0)


def ipcw(train, test, h=H):
    """Status at h and weights for the test rows, with the censoring distribution estimated on the training rows.
    Returns (case, control, weight): case = event by h (weight 1/G(T)); control = followed beyond h (weight 1/G(h));
    censored before h = neither, weight 0."""
    t, e = test["time_y"].to_numpy(dtype=float), test["event"].to_numpy()
    case, control = (t <= h) & (e == 1), t > h
    w = np.zeros(len(t))
    w[case] = 1.0 / censoring_survival(train["time_y"], train["event"], t[case])
    w[control] = 1.0 / censoring_survival(train["time_y"], train["event"], h)[0]
    return case, control, w


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]
A.check("Kaplan-Meier reproduces a hand calculation (1, 0.75, 0.75, 0.375, 0)",
        np.allclose(km_survival([1, 2, 3, 4], [1, 0, 1, 1], [0.5, 1, 2, 3, 4]), [1, 0.75, 0.75, 0.375, 0.0]))
REF_KM = [0.86, 0.7466666666666666, 0.615941095579149, 0.5293783467157591, 0.38064782793776253]       # lifelines 0.30.3
A.check("Kaplan-Meier on the toy data equals lifelines at 0.5, 1, 2, 3 and 5 years", np.allclose(km_survival(_d["time_y"], _d["event"], [0.5, 1, 2, 3, 5]), REF_KM, atol=1e-12))
A.check("Kaplan-Meier never increases and starts at 1", (np.diff(km_survival(_d["time_y"], _d["event"], np.linspace(0, 6, 61))) <= 0).all() and km_survival(_d["time_y"], _d["event"], 0.0)[0] == 1.0)
A.check("observed risk is 1 - Kaplan-Meier, and is not estimated without events or without follow-up to the horizon",
        abs(km_risk(_d, 3.0) - (1 - REF_KM[3])) < 1e-12 and np.isnan(km_risk(_d[_d["event"] == 0], 3.0)) and np.isnan(km_risk(_d[_d["time_y"] < 2], 3.0)))
# tie convention: one event and one censoring at time 1 among 4; then one censoring at time 2 among the 2 left
A.check("censoring distribution with a tie: the event leaves first, so G(1) = 1 - 1/3 and G(2) = (2/3)(1/2)",
        np.allclose(censoring_survival([1, 1, 2, 3], [1, 0, 0, 1], [0.5, 1, 2, 3]), [1, 2 / 3, 1 / 3, 1 / 3]))
_case, _ctrl, _w = ipcw(_tr, _te[_te["time_y"] < _tr["time_y"].max()], 3.0)
_tk = _te[_te["time_y"] < _tr["time_y"].max()]
A.check("toy test rows: 40 events by 3 years, 35 followed beyond, 25 censored before (weight 0)",
        (int(_case.sum()), int(_ctrl.sum()), int((_w == 0).sum())) == (40, 35, 25) and not (_case & _ctrl).any())
A.check("nobody censored before the horizon is counted as an event or a non-event",
        (_w[(_tk["event"].to_numpy() == 0) & (_tk["time_y"].to_numpy() <= 3.0)] == 0).all())
A.check("weights are at least 1; every participant followed beyond the horizon has the same weight 1/G(3), G(3) as in scikit-survival",
        (_w[_case | _ctrl] >= 1).all() and np.allclose(_w[_ctrl], 1 / 0.590283704066138, atol=1e-9), round(float(1 / _w[_ctrl][0]), 6))
_all1 = toy_survival(300, ties=False).assign(event=1)
A.check("without any censoring every weight is exactly 1", (ipcw(_all1, _all1, 3.0)[2] == 1.0).all())
# live checks against lifelines and scikit-survival installed on this machine
import sksurv
import lifelines
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
_kmf = KaplanMeierFitter().fit(_d["time_y"], _d["event"])
_grid = np.linspace(0, 6, 61)
A.check(f"live: Kaplan-Meier equals lifelines {lifelines.__version__} at 61 time points", np.allclose(km_survival(_d["time_y"], _d["event"], _grid), _kmf.survival_function_at_times(_grid).to_numpy(), atol=1e-12))
_cde = CensoringDistributionEstimator().fit(Surv.from_arrays(_tr["event"].astype(bool).to_numpy(), _tr["time_y"].to_numpy()))
_tt = _tk["time_y"].to_numpy()
A.check(f"live: censoring distribution equals scikit-survival {sksurv.__version__} at every toy test time", np.allclose(censoring_survival(_tr["time_y"], _tr["event"], _tt), _cde.predict_proba(_tt), atol=1e-12))
_ = A.close(outputs={"reference": f"lifelines 0.30.3, scikit-survival 0.28.0 (stored); lifelines {lifelines.__version__}, scikit-survival {sksurv.__version__} (live)"})
keep_for_session(24)

# %%
# CELL 25 - discrimination and overall accuracy at 3 years                                         [status L; protocol section 9; AUPRC: A1.1]
# Why: C-index, time-dependent AUC and the Brier score are the protocol's discrimination and accuracy metrics. Each is
# checked against reference values from lifelines and scikit-survival on the toy data. The average precision (AUPRC,
# proposal pp. 15, 22) is defined and tested here but is computed on study data only if amendment item A1.1 is approved.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 25)
A = Audit(25, "Discrimination and Brier score")


def concordance_index(time, score, event, block=4_000_000):
    """Harrell's C: among comparable pairs, the share in which the person with the earlier event has the higher score
    (ties in score count one half). A pair is comparable when one has an event and the other is known to be event-free
    for longer: a later time, or censored at the same time. Two events at the same time are not compared (lifelines)."""
    t, s, e = np.asarray(time, dtype=float), np.asarray(score, dtype=float), np.asarray(event).astype(bool)
    o = np.argsort(t, kind="stable")
    t, s, e = t[o], s[o], e[o]
    ev = np.flatnonzero(e)
    left, right = np.searchsorted(t, t[ev], side="left"), np.searchsorted(t, t[ev], side="right")
    n, col, cens = len(t), np.arange(len(t)), ~e
    conc = ties = pairs = 0.0
    rows = max(1, block // n)
    for a in range(0, len(ev), rows):
        lo, hi = left[a:a + rows, None], right[a:a + rows, None]
        comparable = (col[None, :] >= hi) | ((col[None, :] >= lo) & (col[None, :] < hi) & cens[None, :])
        si = s[ev[a:a + rows], None]
        conc += (comparable & (s[None, :] < si)).sum()
        ties += (comparable & (s[None, :] == si)).sum()
        pairs += comparable.sum()
    return float((conc + 0.5 * ties) / pairs) if pairs else float("nan")


def auc_at(train, test, score, h=H):
    """Cumulative/dynamic AUC at h with censoring weights (Uno): events by h against those followed beyond h."""
    case, control, w = ipcw(train, test, h)
    s = np.asarray(score, dtype=float)
    sk = np.sort(s[control])
    less, le = np.searchsorted(sk, s[case], side="left"), np.searchsorted(sk, s[case], side="right")
    return float((w[case] * (less + 0.5 * (le - less))).sum() / (w[case].sum() * len(sk)))


def brier_at(train, test, risk, h=H):
    """Brier score at h with censoring weights (Graf): mean squared difference between predicted risk and status at h."""
    case, control, w = ipcw(train, test, h)
    r = np.asarray(risk, dtype=float)
    return float((w * np.where(case, (1.0 - r) ** 2, r ** 2)).sum() / len(r))


def scaled_brier_at(train, test, risk, h=H):
    """1 - Brier / Brier of the uninformative prediction (the Kaplan-Meier risk of the training data for everyone)."""
    null = np.full(len(test), 1.0 - km_survival(train["time_y"], train["event"], h)[0])
    return 1.0 - brier_at(train, test, risk, h) / brier_at(train, test, null, h)


def average_precision_at(train, test, score, h=H):
    """AUPRC at h with censoring weights: area under precision against recall, events by h being the positives."""
    case, control, w = ipcw(train, test, h)
    known = case | control
    s, y, w = np.asarray(score, dtype=float)[known], case[known].astype(float), w[known]
    o = np.argsort(-s, kind="stable")
    s, y, w = s[o], y[o], w[o]
    tp, fp = np.cumsum(w * y), np.cumsum(w * (1 - y))
    last = np.append(np.flatnonzero(np.diff(s) != 0), len(s) - 1)          # last row of each group of tied scores
    precision, recall = tp[last] / (tp[last] + fp[last]), tp[last] / tp[-1]
    return float((np.diff(np.append(0.0, recall)) * precision).sum())


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]
_lp_tr, _lp_te = _tr[["x1", "x2"]].to_numpy() @ BETA_TOY, _te[["x1", "x2"]].to_numpy() @ BETA_TOY
_r3 = risk_at(Breslow(_tr["time_y"].to_numpy(), _tr["event"].to_numpy(), _lp_tr), _lp_te, 3.0)
REF = {"c_test": 0.6137734052681688, "c_train": 0.6007552870090634, "c_tied_scores": 0.6091716915264995,          # lifelines 0.30.3
       "auc": 0.5788494750946951, "auc_tied_scores": 0.5782694183177958, "brier": 0.25217341693181056,              # scikit-survival 0.28.0
       "auprc": 0.5180786953590525, "auprc_tied_scores": 0.5021647565831647}                                                                                     # scikit-learn 1.9.1
A.check("C-index: one event before one censored at the same time is a comparable pair; two events at the same time are not",
        concordance_index([1, 1], [2, 1], [1, 0]) == 1.0 and concordance_index([1, 1], [1, 2], [1, 0]) == 0.0
        and np.isnan(concordance_index([1, 1], [2, 1], [1, 1])) and concordance_index([1, 2], [1, 1], [1, 1]) == 0.5)
A.check("C-index equals lifelines on the toy test rows, the toy training rows, and with tied scores",
        abs(concordance_index(_te["time_y"], _lp_te, _te["event"]) - REF["c_test"]) < 1e-12
        and abs(concordance_index(_tr["time_y"], _lp_tr, _tr["event"]) - REF["c_train"]) < 1e-12
        and abs(concordance_index(_te["time_y"], np.round(_lp_te, 1), _te["event"]) - REF["c_tied_scores"]) < 1e-12)
A.check("C-index does not depend on how the calculation is split into blocks",
        concordance_index(_tr["time_y"], _lp_tr, _tr["event"], block=1500) == concordance_index(_tr["time_y"], _lp_tr, _tr["event"]))
A.check("C-index is 1 for a perfect ranking, 0 for the reverse, and unchanged by any increasing transformation of the score",
        concordance_index([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 0]) == 1.0 and concordance_index([1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 0]) == 0.0
        and concordance_index(_te["time_y"], np.exp(_lp_te), _te["event"]) == concordance_index(_te["time_y"], _lp_te, _te["event"]))
_keep = (_te["time_y"] < _tr["time_y"].max()).to_numpy()
_tk, _lpk, _rk = _te[_keep], _lp_te[_keep], _r3[_keep]
A.check("time-dependent AUC at 3 years equals scikit-survival, also with tied scores",
        abs(auc_at(_tr, _tk, _lpk, 3.0) - REF["auc"]) < 1e-12 and abs(auc_at(_tr, _tk, np.round(_lpk, 1), 3.0) - REF["auc_tied_scores"]) < 1e-12,
        round(auc_at(_tr, _tk, _lpk, 3.0), 6))
A.check("Brier score at 3 years equals scikit-survival", abs(brier_at(_tr, _tk, _rk, 3.0) - REF["brier"]) < 1e-9, round(brier_at(_tr, _tk, _rk, 3.0), 6))
A.check("scaled Brier is exactly 0 for the uninformative prediction and below 1 for the model",
        abs(scaled_brier_at(_tr, _tk, np.full(len(_tk), 1 - km_survival(_tr["time_y"], _tr["event"], 3.0)[0]), 3.0)) < 1e-15 and scaled_brier_at(_tr, _tk, _rk, 3.0) < 1)
_u = toy_survival(300, ties=False).assign(event=1)                     # no censoring: status at 3 years is known for everyone
_yu, _su = (_u["time_y"] <= 3.0).to_numpy(), _u[["x1", "x2"]].to_numpy() @ BETA_TOY
A.check("without censoring the Brier score is the ordinary mean squared error and the AUC the ordinary rank statistic",
        abs(brier_at(_u, _u, np.full(300, 0.3), 3.0) - np.mean((_yu - 0.3) ** 2)) < 1e-15
        and abs(auc_at(_u, _u, _su, 3.0) - np.mean(_su[_yu][:, None] > _su[~_yu][None, :])) < 1e-12)
A.check("AUPRC equals the scikit-learn reference on the toy rows (also with tied scores), and the event share for a constant score",
        abs(average_precision_at(_tr, _tk, _lpk, 3.0) - REF["auprc"]) < 1e-12 and abs(average_precision_at(_tr, _tk, np.round(_lpk, 1), 3.0) - REF["auprc_tied_scores"]) < 1e-12
        and abs(average_precision_at(_u, _u, np.zeros(300), 3.0) - _yu.mean()) < 1e-12,
        round(average_precision_at(_tr, _tk, _lpk, 3.0), 6))
# live checks against lifelines, scikit-survival and scikit-learn installed on this machine
import sklearn
import sksurv
from lifelines.utils import concordance_index as _ll_cindex
from sksurv.metrics import cumulative_dynamic_auc as _sk_auc, brier_score as _sk_brier
from sksurv.util import Surv
from sklearn.metrics import average_precision_score
A.check("live: C-index equals lifelines on the toy test rows, also with tied scores",
        abs(concordance_index(_te["time_y"], _lp_te, _te["event"]) - _ll_cindex(_te["time_y"], -_lp_te, _te["event"])) < 1e-12
        and abs(concordance_index(_te["time_y"], np.round(_lp_te, 1), _te["event"]) - _ll_cindex(_te["time_y"], -np.round(_lp_te, 1), _te["event"])) < 1e-12)
_ytr = Surv.from_arrays(_tr["event"].astype(bool).to_numpy(), _tr["time_y"].to_numpy())
_ytk = Surv.from_arrays(_tk["event"].astype(bool).to_numpy(), _tk["time_y"].to_numpy())
A.check(f"live: AUC and Brier score at 3 years equal scikit-survival {sksurv.__version__}",
        abs(auc_at(_tr, _tk, _lpk, 3.0) - float(_sk_auc(_ytr, _ytk, _lpk, [3.0])[0][0])) < 1e-12
        and abs(brier_at(_tr, _tk, _rk, 3.0) - float(_sk_brier(_ytr, _ytk, (1 - _rk)[:, None], [3.0])[1][0])) < 1e-12)
_c, _k, _w = ipcw(_tr, _tk, 3.0)
_kn = _c | _k
A.check(f"live: AUPRC equals scikit-learn {sklearn.__version__} average precision with the same weights",
        abs(average_precision_at(_tr, _tk, _lpk, 3.0) - average_precision_score(_c[_kn].astype(int), _lpk[_kn], sample_weight=_w[_kn])) < 1e-12)
_ = A.close(outputs={"reference_values": REF, "references": "stored: lifelines 0.30.3, scikit-survival 0.28.0, scikit-learn 1.9.1; live: versions on this machine"})
keep_for_session(25)

# %%
# CELL 26 - calibration and clinical usefulness                                                    [status L; protocol section 9; intercept: A1.1]
# Why: a risk of 5% should mean that about 5 in 100 such people develop diabetes within 3 years (proposal pp. 15-16).
# Calibration slope, calibration distance |slope - 1|, observed-to-expected ratio, the table by tenths of risk and the
# decision curve are the protocol's measures. The calibration intercept is an amendment metric (A1.1).
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 26)
A = Audit(26, "Calibration and decision curve")


def cloglog(risk):
    return np.log(-np.log1p(-np.clip(np.asarray(risk, dtype=np.float64), 1e-9, 1 - 1e-9)))


def cal_slope(risk, df):
    """Calibration slope: coefficient of a Cox model with log(-log(1 - risk)) as its only covariate. 1 is ideal; below 1
    means the predictions are too extreme, above 1 too timid."""
    return cox_slope(cloglog(risk), df["time_y"].to_numpy(dtype=float), df["event"].to_numpy())


def oe_ratio(risk, df, h=H):
    """Observed (Kaplan-Meier) risk by h divided by the mean predicted risk. 1 is ideal; above 1 means under-prediction."""
    return km_risk(df, h) / float(np.mean(risk))


def cal_intercept(risk, df, h=H):
    """Calibration-in-the-large on the complementary log-log scale: cloglog(observed risk) - cloglog(mean predicted risk).
    0 is ideal. Amendment item A1.1."""
    return float(cloglog(km_risk(df, h)) - cloglog(np.mean(risk)))


def calibration_table(risk, df, h=H, q=10):
    """Mean predicted and observed (Kaplan-Meier) risk within q groups of predicted risk."""
    risk = np.asarray(risk, dtype=np.float64)
    g = pd.qcut(risk, q, labels=False, duplicates="drop")
    rows = []
    for k, part in df.reset_index(drop=True).groupby(g):
        rows.append({"group": int(k) + 1, "n": len(part), "events": int(part["event"].sum()), "predicted": float(risk[part.index].mean()), "observed": km_risk(part, h)})
    return pd.DataFrame(rows)


def decision_curve(risk, df, thresholds=None, h=H):
    """Net benefit of acting on everyone whose predicted risk is at least the threshold, with observed risk by Kaplan-Meier."""
    risk = np.asarray(risk, dtype=np.float64)
    prev, out = km_risk(df, h), []
    for th in (EVAL["net_benefit_thresholds"] if thresholds is None else thresholds):
        m = risk >= th
        share = m.mean()
        obs = km_risk(df[m], h) if m.sum() >= 20 else float("nan")
        out.append({"threshold": float(th), "share_above": float(share), "net_benefit": share * obs - share * (1 - obs) * th / (1 - th),
                    "treat_all": prev - (1 - prev) * th / (1 - th), "treat_none": 0.0})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]
_lp_tr, _lp_te = _tr[["x1", "x2"]].to_numpy() @ BETA_TOY, _te[["x1", "x2"]].to_numpy() @ BETA_TOY
_H = Breslow(_tr["time_y"].to_numpy(), _tr["event"].to_numpy(), _lp_tr)
_r3 = risk_at(_H, _lp_te, 3.0)
A.check("calibration slope on the toy test rows equals lifelines (1.013683)", abs(cal_slope(_r3, _te) - 1.013683044824473) < 1e-5, round(cal_slope(_r3, _te), 6))
_sharp = risk_at(_H, 2.0 * _lp_te, 3.0)
A.check("doubling the log relative hazard (over-confident predictions) halves the slope", abs(cal_slope(_sharp, _te) - cal_slope(_r3, _te) / 2) < 1e-6, round(cal_slope(_sharp, _te), 6))
_const = np.full(len(_te), km_risk(_te, 3.0))
A.check("predicting the observed risk for everyone gives observed/expected = 1 and intercept = 0",
        abs(oe_ratio(_const, _te, 3.0) - 1) < 1e-12 and abs(cal_intercept(_const, _te, 3.0)) < 1e-9)
A.check("halving every prediction gives observed/expected = 2 and a positive intercept (risk is under-predicted)",
        abs(oe_ratio(_const / 2, _te, 3.0) - 2) < 1e-12 and cal_intercept(_const / 2, _te, 3.0) > 0)
_ct = calibration_table(risk_at(_H, _d[["x1", "x2"]].to_numpy() @ BETA_TOY, 3.0), _d, 3.0, q=5)
A.check("calibration table: five groups of 60, all 150 events placed, predicted risk rising from group to group",
        _ct["n"].tolist() == [60] * 5 and int(_ct["events"].sum()) == 150 and (np.diff(_ct["predicted"]) > 0).all() and _ct["observed"].between(0, 1).all())
_dc = decision_curve(_r3, _te, thresholds=[0.05, 0.3, 0.5], h=3.0)
A.check("decision curve: when every predicted risk exceeds the threshold, the model equals 'act on everyone'",
        _r3.min() > 0.05 and abs(_dc.loc[0, "net_benefit"] - _dc.loc[0, "treat_all"]) < 1e-12 and _dc.loc[0, "share_above"] == 1.0)
A.check("decision curve: net benefit of acting on nobody is 0, and the default thresholds are 1% to 10%",
        (_dc["treat_none"] == 0).all() and decision_curve(_r3, _te)["threshold"].round(2).tolist() == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
from lifelines import CoxPHFitter
_ll_slope = float(CoxPHFitter().fit(pd.DataFrame({"z": cloglog(_r3), "T": _te["time_y"].to_numpy(), "E": _te["event"].to_numpy()}), "T", "E").params_["z"])
A.check("live: calibration slope equals lifelines' one-variable Cox fit to 5 decimals", abs(cal_slope(_r3, _te) - _ll_slope) < 1e-5, round(_ll_slope, 6))
_ = A.close(outputs={"reference": "lifelines 0.30.3 (stored) and the installed lifelines (live)"})
keep_for_session(26)

# %%
# CELL 27 - classification metrics at a risk threshold, with censoring weights                     [status A: amendment item A1.1; proposal pp. 15, 22]
# Why: the proposal asks for sensitivity, specificity, PPV, NPV, F1 and false-negative rate. With censored follow-up these
# need the weights of cell 24. The function is defined and tested on toy arrays here; it is applied to study data only
# in cells that first call require_approval("A1.1").
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 27)
A = Audit(27, "Threshold metrics with censoring weights", status="A")

THRESHOLDS_FIXED = [0.02, 0.05]                    # predicted 3-year risk of 2% and 5% (amendment A1.1)
TOP_SHARE = 0.10                                   # and the top tenth of predicted risk, cut-off taken from training data


def threshold_metrics(train, test, risk, threshold, h=H):
    """Weighted 2 x 2 table at h for 'predicted risk >= threshold' against 'event by h'."""
    case, control, w = ipcw(train, test, h)
    pos = np.asarray(risk, dtype=float) >= threshold
    tp, fn = (w * (case & pos)).sum(), (w * (case & ~pos)).sum()
    fp, tn = (w * (control & pos)).sum(), (w * (control & ~pos)).sum()
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    sens, spec, ppv, npv = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp), div(tn, tn + fn)
    return {"threshold": float(threshold), "share_flagged": float(pos.mean()), "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "f1": div(2 * ppv * sens, ppv + sens) if ppv == ppv and sens == sens else float("nan"), "accuracy": div(tp + tn, tp + fn + fp + tn),
            "false_negative_rate": 1 - sens if sens == sens else float("nan"), "false_positive_rate": 1 - spec if spec == spec else float("nan")}


def top_share_cutoff(train_risk, share=TOP_SHARE):
    """The predicted risk above which the top `share` of the training participants lie."""
    return float(np.quantile(np.asarray(train_risk, dtype=float), 1 - share))


# ---------------------------------------------------------------- tests on toy arrays
_u = toy_survival(300, ties=False).assign(event=1)                     # no censoring: plain formulas must be reproduced
_y, _s = (_u["time_y"] <= 3.0).to_numpy(), 1 / (1 + np.exp(-(_u[["x1", "x2"]].to_numpy() @ BETA_TOY)))
_m, _pos = threshold_metrics(_u, _u, _s, 0.5, 3.0), _s >= 0.5
_tp, _fn, _fp, _tn = (_y & _pos).sum(), (_y & ~_pos).sum(), (~_y & _pos).sum(), (~_y & ~_pos).sum()
A.check("without censoring: sensitivity, specificity, PPV and NPV equal the ordinary 2 x 2 table",
        np.allclose([_m["sensitivity"], _m["specificity"], _m["ppv"], _m["npv"]], [_tp / (_tp + _fn), _tn / (_tn + _fp), _tp / (_tp + _fp), _tn / (_tn + _fn)], atol=1e-12))
A.check("without censoring: F1 = 2TP/(2TP+FP+FN), accuracy = (TP+TN)/n, FNR = 1 - sensitivity, FPR = 1 - specificity",
        np.allclose([_m["f1"], _m["accuracy"], _m["false_negative_rate"], _m["false_positive_rate"]],
                    [2 * _tp / (2 * _tp + _fp + _fn), (_tp + _tn) / 300, _fn / (_tp + _fn), _fp / (_fp + _tn)], atol=1e-12))
_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]
_te = _te[_te["time_y"] < _tr["time_y"].max()]
_r = risk_at(Breslow(_tr["time_y"].to_numpy(), _tr["event"].to_numpy(), _tr[["x1", "x2"]].to_numpy() @ BETA_TOY), _te[["x1", "x2"]].to_numpy() @ BETA_TOY, 3.0)
_lo, _hi = threshold_metrics(_tr, _te, _r, 0.0, 3.0), threshold_metrics(_tr, _te, _r, 1.1, 3.0)
A.check("threshold 0 flags everyone (sensitivity 1, specificity 0); a threshold above 1 flags nobody (sensitivity 0, specificity 1)",
        (_lo["sensitivity"], _lo["specificity"], _hi["sensitivity"], _hi["specificity"]) == (1.0, 0.0, 0.0, 1.0) and np.isnan(_hi["ppv"]))
_grid = [threshold_metrics(_tr, _te, _r, th, 3.0) for th in (0.3, 0.45, 0.6)]
A.check("raising the threshold lowers sensitivity and raises specificity", np.all(np.diff([g["sensitivity"] for g in _grid]) <= 0) and np.all(np.diff([g["specificity"] for g in _grid]) >= 0))
_case, _ctrl, _w = ipcw(_tr, _te, 3.0)
A.check("with censoring: PPV at threshold 0 equals the weighted share of events, and censored rows carry no weight",
        abs(_lo["ppv"] - _w[_case].sum() / _w.sum()) < 1e-12 and (_w[~(_case | _ctrl)] == 0).all())
_cut = top_share_cutoff(np.linspace(0, 1, 1001))
A.check("top-tenth cut-off: one tenth of the training risks lie at or above it", abs(_cut - 0.9) < 1e-12 and abs((np.linspace(0, 1, 1001) >= _cut).mean() - 0.1) < 2e-3)
_ = A.close(outputs={"thresholds": THRESHOLDS_FIXED, "top_share": TOP_SHARE})
keep_for_session(27)

# %%
# CELL 28 - inference, constraint violations, the metric bundle and the resumable unit runner      [status L; protocol sections 8, 9, 11, 12]
# Why: the decision rules rest on the Nadeau-Bengio interval (cross-validation) and on the paired bootstrap (cross-cohort),
# and claim C4 on the violation rate. The unit runner makes long runs resumable: a stored result is reused only if its
# record shows the same settings, data, code and file content; a file merely existing on disk proves nothing.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 28)
A = Audit(28, "Inference, violations, metric bundle, unit runner")
from scipy import stats as _stats


def approved(item):
    """True if an amendment item is approved in cell 3 (never raises; used to add amendment metrics to a table)."""
    rec = read_json(APPROVAL_FILE)["items"].get(item)
    return bool(rec and rec["approved"] and rec["amendment_sha"] == A1_SHA)


def nadeau_bengio(diffs, n_train, n_test, level=0.95):
    """Corrected resampled t interval for the mean of paired fold differences (repeated cross-validation). The variance
    is inflated by n_test/n_train because the folds share training data."""
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    se = np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
    tq = float(_stats.t.ppf(0.5 + level / 2, J - 1))
    return {"mean": float(d.mean()), "lo95": float(d.mean() - tq * se), "hi95": float(d.mean() + tq * se), "J": J}


def violation_rate(model, df, delta=None, n=None, seed=0, priors=None):
    """Protocol section 9. For up to 5,000 participants, raise one constrained continuous variable by 0.25 SD and see
    whether the model's log relative hazard moves against the prior. Returns (mean rate over variables, rate per
    variable, mean size of the adverse change per variable). Works for any model with .cols and .lp()."""
    delta = EVAL["viol_delta_sd"] if delta is None else delta
    priors = PRIORS if priors is None else priors
    s = df.sample(min(EVAL["viol_n"] if n is None else n, len(df)), random_state=seed).reset_index(drop=True)
    base, rate, size = model.lp(s), {}, {}
    for c in model.cols:
        if c in priors and s[c].nunique() > 2:
            p = s.copy()
            p[c] = p[c] + delta * df[c].std()
            change = priors[c] * (model.lp(p) - base)
            rate[c] = float(np.mean(change < -1e-6))
            size[c] = float(np.mean(np.maximum(0.0, -change)))
    return (float(np.mean(list(rate.values()))) if rate else float("nan")), rate, size


def metrics_from_arrays(train, test, lp, risk, h=H, viol=float("nan")):
    """The protocol's metric bundle (section 9) from a log relative hazard and a 3-year risk for the test rows. Test rows
    beyond the training follow-up are left out, because the censoring weights are not defined there."""
    keep = (test["time_y"] < train["time_y"].max()).to_numpy()
    test, lp, risk = test[keep], np.asarray(lp)[keep], np.asarray(risk, dtype=np.float64)[keep]
    bs, slope, obs = brier_at(train, test, risk, h), cal_slope(risk, test), km_risk(test, h)
    out = {"n": len(test), "events": int(test["event"].sum()), "events_by_h": int(((test["event"] == 1) & (test["time_y"] <= h)).sum()),
           "cindex": concordance_index(test["time_y"], lp, test["event"]), "auc_3y": auc_at(train, test, lp, h),
           "brier": bs, "scaled_brier": scaled_brier_at(train, test, risk, h), "obs_risk": obs, "mean_pred": float(risk.mean()),
           "oe": obs / float(risk.mean()), "cal_slope": slope, "cal_slope_dist": abs(slope - 1.0), "viol_rate": viol}
    if approved("A1.1"):                                # amendment metrics, only when approved
        out.update({"auprc": average_precision_at(train, test, lp, h), "cal_intercept": cal_intercept(risk, test, h)})
    return out


def evaluate(fit, test, h=H):
    """All metrics for a fitted model (a Fitted object) on test rows, including its violation rate."""
    test = test[test["time_y"] < fit.train["time_y"].max()]
    return metrics_from_arrays(fit.train, test, fit.model.lp(test), fit.risk(test, h), h, viol=violation_rate(fit.model, test)[0])


def subgroup_metrics(fit, test, by, h=H):
    """The bundle within each level of `by`; a level needs 20 events and follow-up to h (otherwise it is skipped)."""
    rows = []
    for g, d in test.groupby(by, observed=True):
        if d["event"].sum() >= 20 and d["time_y"].max() >= h:
            r = evaluate(fit, d, h)
            rows.append({by: g, **{k: r[k] for k in ("n", "events", "events_by_h", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope")}})
    return pd.DataFrame(rows)


class ScaledLP:
    """A model whose log relative hazard is multiplied by a slope (used by recalibration)."""

    def __init__(self, model, slope):
        self.model, self.slope, self.cols = model, slope, model.cols

    def lp(self, df):
        return self.slope * self.model.lp(df)


def recalibrate(fit, target_sample):
    """Protocol section 8: intercept-and-slope recalibration on a small target sample: a new slope for lp (one-variable
    Cox model) and a new Breslow baseline. Returns (recalibrated Fitted, slope). This is model updating, not validation."""
    slope = cox_slope(fit.model.lp(target_sample), target_sample["time_y"].to_numpy(dtype=float), target_sample["event"].to_numpy())
    return Fitted(ScaledLP(fit.model, slope), target_sample), slope


BOOT_METRICS = {"cindex": lambda tr, te, lp, r, h: concordance_index(te["time_y"], lp, te["event"]),
                "scaled_brier": lambda tr, te, lp, r, h: scaled_brier_at(tr, te, r, h),
                "cal_slope_dist": lambda tr, te, lp, r, h: abs(cal_slope(r, te) - 1.0)}


def boot_diff_arrays(train, test, arrs, ref, n_boot=None, seed=0, h=H, max_n=None, metrics=("cindex", "scaled_brier", "cal_slope_dist")):
    """Paired bootstrap (section 9): metric(model) - metric(ref) on the same resampled participants, percentile interval.
    arrs = {name: (lp, risk)} aligned with `test`; max_n takes a fixed random subsample of the test rows first."""
    n_boot = EVAL["boot_n"] if n_boot is None else n_boot
    keep = (test["time_y"] < train["time_y"].max()).to_numpy()
    test, arrs = test[keep].reset_index(drop=True), {k: (np.asarray(v[0])[keep], np.asarray(v[1])[keep]) for k, v in arrs.items()}
    rng = np.random.default_rng(seed)
    if max_n and len(test) > max_n:
        sel = rng.choice(len(test), max_n, replace=False)
        test, arrs = test.iloc[sel].reset_index(drop=True), {k: (v[0][sel], v[1][sel]) for k, v in arrs.items()}
    draws = {k: {m: [] for m in metrics} for k in arrs if k != ref}
    for _b in range(n_boot):
        idx = rng.integers(0, len(test), len(test))
        b = test.iloc[idx]
        base = {m: BOOT_METRICS[m](train, b, arrs[ref][0][idx], arrs[ref][1][idx], h) for m in metrics}
        for k in draws:
            for m in metrics:
                draws[k][m].append(BOOT_METRICS[m](train, b, arrs[k][0][idx], arrs[k][1][idx], h) - base[m])
    return pd.DataFrame([{"model": k, "ref": ref, "metric": m, "mean": float(np.mean(v)), "lo95": float(np.percentile(v, 2.5)),
                          "hi95": float(np.percentile(v, 97.5)), "n_boot": n_boot, "n": len(test)} for k, d in draws.items() for m, v in d.items()])


def code_fingerprint(cells):
    """SHA-256 over the saved sources of the given definition cells: a result depends on the code that produced it."""
    return sha256_text("".join((SESSION / f"cell_{int(c):02d}.py").read_text(encoding="utf-8") for c in sorted(cells)))


def unit_valid(name, code_cells, folder=None):
    """A stored unit may be reused only if its record matches the current settings, harmonized data and code, and the
    result file still has the recorded content and number of rows."""
    folder = DIR["parts"] if folder is None else folder
    p, m = folder / f"{name}.csv", folder / f"{name}.json"
    if not (p.exists() and m.exists()):
        return False
    meta = read_json(m)
    return (meta.get("settings_sha") == settings_fingerprint() and meta.get("harm_sha") == HARM_SHA and meta.get("code_sha") == code_fingerprint(code_cells)
            and meta.get("file_sha") == sha256_file(p) and meta.get("rows") == len(pd.read_csv(p)))


def run_unit(name, fn, code_cells, folder=None, force=False, quiet=False):
    """Run one resumable unit of work. fn() must return a DataFrame. Returns (DataFrame, 'computed' or 'reused')."""
    folder = DIR["parts"] if folder is None else folder
    p, m = folder / f"{name}.csv", folder / f"{name}.json"
    if not force and unit_valid(name, code_cells, folder):
        if not quiet:
            print(f"   reused   {name}  (record validated)")
        return pd.read_csv(p), "reused"
    t0 = time.time()
    out = fn()
    out.to_csv(p, index=False)
    write_json(m, {"unit": name, "time": now(), "seconds": round(time.time() - t0, 1), "rows": len(out), "columns": list(out.columns),
                   "settings_sha": settings_fingerprint(), "harm_sha": HARM_SHA, "code_sha": code_fingerprint(code_cells), "code_cells": sorted(code_cells),
                   "file_sha": sha256_file(p), "protocol_sha": PROTOCOL_SHA})
    with open(OUT / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"unit": name, "time": now(), "passed": True, "protocol_sha": PROTOCOL_SHA, "settings_sha": settings_fingerprint(), "source_sha": code_fingerprint(code_cells)}) + "\n")
    if not quiet:
        print(f"   computed {name}  ({time.time() - t0:.0f} s)")
    return pd.read_csv(p), "computed"


# ---------------------------------------------------------------- tests on toy arrays
_dd = np.array([0.010, 0.020, 0.015, 0.000, 0.020, 0.010, 0.012, 0.018, 0.005, 0.020, 0.011, 0.014, 0.016, 0.009, 0.013])
_nb = nadeau_bengio(_dd, 1000, 250)
_se = np.sqrt((1 / 15 + 250 / 1000) * _dd.var(ddof=1))
A.check("Nadeau-Bengio interval equals mean +/- t(0.975, 14) x sqrt((1/15 + n_test/n_train) x variance), t = 2.1448",
        abs(_nb["lo95"] - (_dd.mean() - 2.1447866879 * _se)) < 1e-9 and abs(_nb["hi95"] - (_dd.mean() + 2.1447866879 * _se)) < 1e-9 and _nb["J"] == 15)
A.check("it is wider than the uncorrected t interval (the correction never narrows)", _nb["hi95"] - _nb["lo95"] > 2 * 2.1447866879 * _dd.std(ddof=1) / np.sqrt(15))

_d = toy_survival(300)
_tr, _te = _d.iloc[:200], _d.iloc[200:]


class _Lin:
    def __init__(self, beta):
        self.cols, self.beta = ["x1", "x2"], np.asarray(beta, dtype=float)

    def fit(self, df):
        return self

    def lp(self, df):
        return df[["x1", "x2"]].to_numpy() @ self.beta


_pri = {"x1": +1, "x2": -1}
_v_ok, _v_bad = violation_rate(_Lin([0.5, -0.3]), _te, priors=_pri), violation_rate(_Lin([-0.5, -0.3]), _te, priors=_pri)
A.check("violation rate is 0 for a model that follows both priors, and 1 for the variable whose effect is reversed",
        _v_ok[0] == 0.0 and _v_bad[1] == {"x1": 1.0, "x2": 0.0} and _v_bad[0] == 0.5)
A.check("violation size is the mean adverse change: 0.5 x 0.25 SD of x1 for the reversed model, 0 otherwise",
        abs(_v_bad[2]["x1"] - 0.5 * 0.25 * _te["x1"].std()) < 1e-12 and _v_bad[2]["x2"] == 0.0 and _v_ok[2] == {"x1": 0.0, "x2": 0.0})
A.check("a variable without a prior, or a yes/no variable, is not tested", violation_rate(_Lin([0.5, -0.3]), _te, priors={"x9": 1})[1] == {}
        and violation_rate(_Lin([0.5, -0.3]), _te.assign(x1=(_te["x1"] > 0) * 1.0), priors=_pri)[1].keys() == {"x2"})

_fit = fit_with_baseline(_Lin(BETA_TOY), _tr)
_mt = evaluate(_fit, _te)
A.check("metric bundle on the toy rows reproduces the reference values of cells 25 and 26",
        abs(_mt["cindex"] - REF["c_test"]) < 1e-9 and abs(_mt["auc_3y"] - REF["auc"]) < 1e-9 and abs(_mt["brier"] - REF["brier"]) < 1e-9
        and abs(_mt["cal_slope"] - 1.0136) < 1e-3 and _mt["n"] == 100 and _mt["events_by_h"] == 40)
A.check("bundle is internally consistent: O/E = observed / mean predicted; distance = |slope - 1|; no violation rate without a constrained variable",
        abs(_mt["oe"] - _mt["obs_risk"] / _mt["mean_pred"]) < 1e-15 and abs(_mt["cal_slope_dist"] - abs(_mt["cal_slope"] - 1)) < 1e-15 and np.isnan(_mt["viol_rate"]))
A.check("amendment metrics appear in the bundle only when A1.1 is approved", ("auprc" in _mt) == approved("A1.1") and ("cal_intercept" in _mt) == approved("A1.1"))
_sg = subgroup_metrics(_fit, _te.assign(grp=np.where(_te["x1"] > 0, "high", "low")), "grp")
A.check("subgroup metrics: one row per level with at least 20 events", sorted(_sg["grp"]) == ["high", "low"] and (_sg["events"] >= 20).all())

_rf, _slope = recalibrate(fit_with_baseline(_Lin(2.0 * BETA_TOY), _tr), _te)
A.check("recalibration: a model whose log hazard is doubled gets slope one half and then predicts as the original model",
        abs(_slope - cox_slope(_Lin(BETA_TOY).lp(_te), _te["time_y"].to_numpy(), _te["event"].to_numpy()) / 2) < 1e-6
        and np.allclose(_rf.model.lp(_te), _slope * 2.0 * _Lin(BETA_TOY).lp(_te)))

_arr = {"good": (_Lin(BETA_TOY).lp(_te), _fit.risk(_te)), "same": (_Lin(BETA_TOY).lp(_te), _fit.risk(_te)),
        "reversed": (_Lin(-BETA_TOY).lp(_te), fit_with_baseline(_Lin(-BETA_TOY), _tr).risk(_te))}
_bt = boot_diff_arrays(_tr, _te, _arr, "good", n_boot=40, seed=0).set_index(["model", "metric"])
A.check("paired bootstrap: a model compared with itself differs by exactly 0 in every resample",
        (_bt.loc["same", ["mean", "lo95", "hi95"]].to_numpy() == 0).all())
A.check("paired bootstrap: a reversed model is worse in C-index and the 95% interval excludes 0", _bt.loc[("reversed", "cindex"), "hi95"] < 0)
A.check("paired bootstrap: the same seed reproduces the same interval",
        boot_diff_arrays(_tr, _te, _arr, "good", n_boot=40, seed=0).set_index(["model", "metric"]).equals(_bt))

_tmp = DIR["cache"] / "unit_selftest"
_tmp.mkdir(exist_ok=True)
for _f in _tmp.glob("toy_unit.*"):
    _f.unlink()
_calls = []


def _toy_unit():
    _calls.append(1)
    return pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 0.25, 0.125]})


_o1, _s1 = run_unit("toy_unit", _toy_unit, code_cells=[22, 23], folder=_tmp, quiet=True)
_o2, _s2 = run_unit("toy_unit", _toy_unit, code_cells=[22, 23], folder=_tmp, quiet=True)
A.check("unit runner: computed the first time, reused the second time with identical content", (_s1, _s2, len(_calls)) == ("computed", "reused", 1) and _o1.equals(_o2))
(_tmp / "toy_unit.csv").write_text("a,b\n1,0.5\n2,0.25\n3,9.0\n")              # tamper with the stored result
_o3, _s3 = run_unit("toy_unit", _toy_unit, code_cells=[22, 23], folder=_tmp, quiet=True)
A.check("unit runner: a result file changed on disk is detected and recomputed", _s3 == "computed" and len(_calls) == 2 and _o3.equals(_o1))
_o4, _s4 = run_unit("toy_unit", _toy_unit, code_cells=[22, 24], folder=_tmp, quiet=True)
A.check("unit runner: a result made with different code is not reused", _s4 == "computed" and len(_calls) == 3)
for _f in _tmp.glob("toy_unit.*"):
    _f.unlink()
_ = A.close(outputs={"boot_metrics": sorted(BOOT_METRICS), "amendment_metrics_on": approved("A1.1")})
keep_for_session(28)

# %%
# CELL 29 - GATE D: the statistical toolkit                                                       [status L]
# Why: a stop point before any model is fitted to study data. It confirms that cells 22 to 28 passed all their tests on
# toy arrays, re-runs one end-to-end toy analysis, and confirms that the toolkit has not touched the study outcome.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 29)
require_gate("C")
A = Audit("gate_D", "Gate D: statistical toolkit")

_recs = {c: load_record(c) for c in range(22, 29)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else f"{len(r['checks'])} tests, {r['time']}")
A.check("every toolkit function is defined in this session",
        all(callable(globals().get(f)) for f in ("Prep", "cv_folds", "repeated_cv", "loso_folds", "shift_split", "inner_split", "cox_newton", "cox_slope", "Breslow",
                                                 "risk_at", "fit_with_baseline", "km_survival", "km_risk", "censoring_survival", "ipcw", "concordance_index", "auc_at",
                                                 "brier_at", "scaled_brier_at", "average_precision_at", "cal_slope", "oe_ratio", "cal_intercept", "calibration_table",
                                                 "decision_curve", "threshold_metrics", "nadeau_bengio", "violation_rate", "metrics_from_arrays", "evaluate",
                                                 "subgroup_metrics", "recalibrate", "boot_diff_arrays", "run_unit", "unit_valid")))

# one end-to-end toy analysis through the public functions only: split, preprocess, fit, risk, metrics
_d = toy_survival(300)
_rows = []
for _rep, _fold, _seed, _tr, _te in repeated_cv(_d, repeats=1):
    _p = Prep(["x1", "x2"]).fit(_tr)
    _b, _nll, _it = cox_newton(_p.transform(_tr), _tr["time_y"], _tr["event"], HP["cox_ridge"])


    class _M:
        cols = ["x1", "x2"]

        def fit(self, df):
            return self

        def lp(self, df, _p=_p, _b=_b):
            return _p.transform(df) @ _b


    _rows.append(evaluate(fit_with_baseline(_M(), _tr), _te))
_cv = pd.DataFrame(_rows)
print("  End-to-end toy analysis (5 folds of the 300 toy rows; these are not study results):")
print(_cv[["n", "events", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate"]].round(3).to_string(index=False))
A.check("toy 5-fold analysis: every metric is finite in every fold", np.isfinite(_cv[["cindex", "auc_3y", "brier", "scaled_brier", "oe", "cal_slope"]].to_numpy()).all())
A.check("toy 5-fold analysis: five folds evaluated, the model discriminates (mean C-index above 0.55)", len(_cv) == 5 and _cv["cindex"].mean() > 0.55,
        round(float(_cv["cindex"].mean()), 3))
A.check("the harmonized study data are unchanged by stage D", all(content_sha(HARM[k]) == HARM_SHA[k] for k in HARM))
A.check("no result part or prediction file has been written: no model has been fitted to study data", not any(p.suffix == ".csv" for p in DIR["parts"].iterdir()) and not any(DIR["preds"].iterdir()))
_missing = [p for p in ("xgboost", "torch") if PKG.set_index("package").loc[p, "version"] == "NOT INSTALLED"]
print(f"\n  Packages absent on this machine that the protocol's software list names: {_missing if _missing else 'none'}.")
print("  Stage E (baseline models) and stage F (network and CKINN) are written for the packages that are present here.")
_ = A.close(outputs={"stage": "D", "toy_cv": _cv[["cindex", "auc_3y", "scaled_brier", "oe", "cal_slope"]].mean().round(6).to_dict(),
                     "tests": {str(c): len(r["checks"]) for c, r in _recs.items() if r}}, extra_path=DIR["gates"] / "gate_D.json")
print("\n  GATE D PASSED. Stage E (cells 30 to 36, baseline models) may start.")
