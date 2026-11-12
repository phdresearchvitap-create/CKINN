# %%
# CELL 30 - baseline 1: Cox regression with ridge penalty 0.01                                    [status L; protocol section 6; proposal p. 17]
# Why: Cox regression is the counterpart of logistic regression for censored follow-up, and the first comparator of
# claim C1. This cell defines the model, checks it against lifelines on toy data, then fits it once to one fold of
# RICH (tier T1) as a smoke test. That fold is unit (repeat 0, fold 0) of the internal validation run in stage G.
# session guard (same three lines in every cell): restores the earlier set-up cells by itself if the kernel was restarted
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 30)
require_gate("D")
A = Audit(30, "Cox regression (ridge 0.01)")


class CoxRidge:
    """Protocol section 6, model 1. Preprocessing is fitted on the training rows (cell 22); the ridge penalty acts on
    coefficients per standard deviation of each prepared column, as in lifelines."""

    def __init__(self, cols, penalizer=None):
        self.cols, self.penalizer = list(cols), HP["cox_ridge"] if penalizer is None else penalizer

    def fit(self, df):
        self.prep = Prep(self.cols).fit(df)
        X = self.prep.transform(df)
        scale = X.std(0, ddof=1)
        scale[scale == 0] = 1.0
        self.beta, self.objective_, self.iterations_ = cox_newton(X, df["time_y"].to_numpy(), df["event"].to_numpy(), self.penalizer, scale)
        return self

    def lp(self, df):
        return self.prep.transform(df) @ self.beta


class FittedDirect:
    """For a model that predicts the 3-year risk directly (a classifier) instead of a log relative hazard."""

    def __init__(self, model, train):
        self.model, self.train, self.H0 = model, train, None

    def risk(self, df, h=H, h_from=None):
        if h != H or h_from is not None:
            raise ValueError("a direct-risk model predicts the risk by the fixed horizon only")
        return self.model.risk3(df)


MODELS = {"Cox": lambda cols, seed=0: CoxRidge(cols)}                 # name -> builder; later cells add the other models


def fit_model(name, cols, train, seed=0):
    """Fit a registered model on training rows and return it with its way of producing absolute 3-year risk."""
    m = MODELS[name](cols, seed)
    return FittedDirect(m.fit(train), train) if hasattr(m, "risk3") else fit_with_baseline(m, train)


def skip_unapproved(audit, item, what):
    """For a cell marked A. If the amendment item is not approved in cell 3: say so, record the skip, return True."""
    if approved(item):
        return False
    print(f"  SKIPPED on study data: {what} belongs to amendment item {item}, which is not approved in cell 3.")
    print(f"  To run it: set \"{item}\": True in cell 3, run cell 3, then run this cell again. Nothing was fitted to study data here.")
    audit.check(f"{item} is not approved, so no study data were used by this cell", True)
    return True


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_d.loc[::7, "x2"] = np.nan
_tr, _te = _d.iloc[:200], _d.iloc[200:]
_m = CoxRidge(["x1", "x2"]).fit(_tr)
REF_COX_PREPARED = [0.27446562, -0.14468310, 0.46652407]             # lifelines 0.30.3, ridge 0.01, on the same prepared toy columns
A.check("Cox ridge with a missingness indicator equals lifelines to 5 decimals (x1, x2, x2_missing)",
        _m.prep.names == ["x1", "x2", "x2_missing"] and np.allclose(_m.beta, REF_COX_PREPARED, atol=1e-5), np.round(_m.beta, 6).tolist())
_f = fit_model("Cox", ["x1", "x2"], _tr)
A.check("toy fit: risks lie strictly between 0 and 1 and the model ranks better than chance",
        ((_f.risk(_te) > 0) & (_f.risk(_te) < 1)).all() and concordance_index(_te["time_y"], _f.model.lp(_te), _te["event"]) > 0.55)
A.check("a stronger penalty shrinks the coefficients", np.abs(CoxRidge(["x1", "x2"], 1.0).fit(_tr).beta).sum() < np.abs(_m.beta).sum())

# ---------------------------------------------------------------- smoke test on one fold of RICH, tier T1 (skipped while a session is being restored)
_rep, _fold, _seed, SMOKE_TRAIN, SMOKE_TEST = next(repeated_cv(HARM["RICH"]))
A.check("smoke fold = repeat 0, fold 0 of RICH: 169,466 training and 42,367 test participants, 3,339 and 835 events",
        (len(SMOKE_TRAIN), len(SMOKE_TEST), int(SMOKE_TRAIN["event"].sum()), int(SMOKE_TEST["event"].sum())) == (169466, 42367, 3339, 835))
A.check("training and test participants of the fold do not overlap", not (set(SMOKE_TRAIN["id"]) & set(SMOKE_TEST["id"])))
if not restoring():
    _t0 = time.time()
    _cols = predictors("RICH", "T1")
    _fit = fit_model("Cox", _cols, SMOKE_TRAIN)
    _mt = evaluate(_fit, SMOKE_TEST)
    print(f"\n  Cox ridge, RICH tier T1, fold (repeat 0, fold 0): fitted in {time.time() - _t0:.1f} s, {_fit.model.iterations_} Newton iterations")
    print(pd.DataFrame({"coefficient per SD": _fit.model.beta, "hazard ratio per SD": np.exp(_fit.model.beta)}, index=_fit.model.prep.names).round(4).to_string())
    print("\n  Test-fold metrics (one fold only; the full internal validation is stage G):")
    print(pd.Series({k: _mt[k] for k in ("n", "events", "events_by_h", "cindex", "auc_3y", "brier", "scaled_brier", "obs_risk", "mean_pred", "oe", "cal_slope", "viol_rate")}).round(4).to_string())
    A.check("the fit converged: all coefficients finite, under 20 Newton iterations", np.isfinite(_fit.model.beta).all() and _fit.model.iterations_ < 20)
    A.check("missingness indicators were created for SBP and DBP only (the only tier T1 variables with missing values)",
            _fit.model.prep.names == _cols + ["sbp_missing", "dbp_missing"])
    A.check("3-year risks lie strictly between 0 and 1", ((_fit.risk(SMOKE_TEST) > 0) & (_fit.risk(SMOKE_TEST) < 1)).all())
    A.check("age, BMI and SBP raise the hazard, in line with their priors", all(_fit.model.beta[_cols.index(v)] > 0 for v in ("age", "bmi", "sbp")))
    A.check("discrimination on the test fold is far above chance (C-index over 0.7) and the Brier score beats the uninformative prediction",
            _mt["cindex"] > 0.7 and _mt["scaled_brier"] > 0, round(_mt["cindex"], 4))
    A.check("a linear model with positive coefficients violates no prior (rate exactly 0)", _mt["viol_rate"] == 0.0)
    A.check("mean predicted 3-year risk is within a factor of 1.5 of the observed risk", 1 / 1.5 < _mt["oe"] < 1.5, round(_mt["oe"], 3))
_ = A.close(outputs={"model": "Cox", "penalizer": HP["cox_ridge"], "reference": "lifelines 0.30.3 (toy data)"})
keep_for_session(30)

# %%
# CELL 31 - baselines 2 and 3: gradient-boosted Cox, without and with monotone constraints         [status L, D; protocol sections 6, 7; deviation D-6]
# Why: boosted trees are the strongest conventional comparator (XGB), and the constrained version (XGB-mono) is the
# conventional way of imposing the clinical priors: CKINN must be compared with it. XGBoost cannot be installed on this
# tablet, so the same algorithm is written here in NumPy (deviation D-6). On toy data without tied times it reproduces
# the predictions of XGBoost 3.4.1 to six decimals; with tied times it uses the exact Breslow gradient.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 31)
A = Audit(31, "Gradient-boosted Cox, with and without monotone constraints", status="L, D")

BOOST = {"lam": 1.0, "min_child_weight": 20.0, "max_bin": 4096}       # fixed settings (D-6); trees, depth and learning rate are in HP


class HistGBM:
    """Gradient-boosted regression trees in NumPy, after the 'hist' algorithm of XGBoost: Newton boosting, L2 penalty
    `lam` on leaf values, a minimum Hessian sum per leaf, missing values sent to whichever side gains more, and optional
    monotone constraints. Every distinct value is its own bin (up to max_bin), so split search is exact. No random
    subsampling: the result is identical on every machine.
    objective 'cox': y = (time, event), Cox partial likelihood with Breslow ties. objective 'logistic': y in {0, 1}."""

    def __init__(self, objective="cox", n_trees=300, depth=3, eta=0.05, lam=1.0, min_child_weight=1.0, max_bin=4096, monotone=None):
        self.objective, self.n_trees, self.depth, self.eta, self.lam = objective, n_trees, depth, eta, lam
        self.mcw, self.max_bin, self.monotone = min_child_weight, max_bin, monotone

    def _bin(self, X):
        B = np.empty(X.shape, dtype=np.int32, order="F")
        for j, cuts in enumerate(self.cuts):
            x = X[:, j]
            b = np.clip(np.searchsorted(cuts, x, side="right") - 1, 0, len(cuts) - 1)       # bin k holds cuts[k] <= x < cuts[k+1]
            B[:, j] = np.where(np.isnan(x), len(cuts), b)                                   # the last bin index means missing
        return B

    def fit(self, X, time=None, event=None, y=None, weight=None):
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        self.cuts = []
        for j in range(p):
            v = X[~np.isnan(X[:, j]), j]
            u = np.unique(v)
            self.cuts.append(u if len(u) <= self.max_bin else np.unique(np.quantile(v, np.linspace(0, 1, self.max_bin + 1)[:-1])))
        B = self._bin(X)
        nb = [len(c) + 1 for c in self.cuts]
        mono = np.zeros(p) if self.monotone is None else np.asarray(self.monotone, dtype=float)
        if self.objective == "cox":
            o = np.argsort(time, kind="stable")                      # ascending time
            B, delta = np.asfortranarray(B[o]), np.asarray(event, dtype=float)[o]
            _ut, starts, inv = np.unique(np.asarray(time, dtype=float)[o], return_index=True, return_inverse=True)
            d_at = np.add.reduceat(delta, starts)
            f = np.zeros(n)
        else:
            y = np.asarray(y, dtype=float)
            w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
            self.base = float(np.log((w * y).sum() / (w * (1 - y)).sum()))
            f = np.full(n, self.base)
        self.trees = []
        for _k in range(self.n_trees):
            if self.objective == "cox":                              # gradient and Hessian of the negative partial log-likelihood
                r = np.exp(f - f.max())
                s0 = np.cumsum(r[::-1])[::-1][starts]                # risk-set sum at each distinct time
                ra = r * np.cumsum(d_at / s0)[inv]
                g, h = ra - delta, ra - r * r * np.cumsum(d_at / s0 ** 2)[inv]
            else:
                pr = 1.0 / (1.0 + np.exp(-f))
                g, h = w * (pr - y), w * pr * (1.0 - pr)
            tree = self._grow(B, g, h, nb, mono)
            f = f + self.eta * self._apply(tree, B)
            self.trees.append(tree)
        return self

    def _weight(self, G, Hs, lo, hi):
        return np.clip(-G / (Hs + self.lam), lo, hi)

    def _grow(self, B, g, h, nb, mono):
        n_nodes = 2 ** (self.depth + 1) - 1                          # complete binary tree; node k has children 2k+1 and 2k+2
        feat, thr, miss_left = np.full(n_nodes, -1), np.zeros(n_nodes, dtype=int), np.zeros(n_nodes, dtype=bool)
        value, leaf = np.zeros(n_nodes), np.ones(n_nodes, dtype=bool)
        lo, hi = np.full(n_nodes, -np.inf), np.full(n_nodes, np.inf)
        rows = {0: np.arange(len(g))}
        for k in range(n_nodes):
            if k not in rows:
                continue
            idx = rows.pop(k)
            gi, hi_ = g[idx], h[idx]
            G, Hs = gi.sum(), hi_.sum()
            value[k] = self._weight(G, Hs, lo[k], hi[k])
            if 2 * k + 2 >= n_nodes or len(idx) < 2:
                continue
            base = -(2 * G * value[k] + (Hs + self.lam) * value[k] ** 2)
            best_gain, best = 1e-6, None
            for j in range(B.shape[1]):
                if nb[j] < 3:
                    continue
                b = B[idx, j]
                gh, hh = np.bincount(b, weights=gi, minlength=nb[j]), np.bincount(b, weights=hi_, minlength=nb[j])
                gm, hm = gh[-1], hh[-1]                              # sums over rows with a missing value
                cg, ch = np.cumsum(gh[:-2]), np.cumsum(hh[:-2])      # left sums for 'bin <= c'
                for left_missing in ((False, True) if hm > 0 else (False,)):
                    GL, HL = (cg + gm, ch + hm) if left_missing else (cg, ch)
                    GR, HR = G - GL, Hs - HL
                    wl, wr = self._weight(GL, HL, lo[k], hi[k]), self._weight(GR, HR, lo[k], hi[k])
                    gain = -(2 * GL * wl + (HL + self.lam) * wl ** 2) - (2 * GR * wr + (HR + self.lam) * wr ** 2) - base
                    ok = (HL >= self.mcw) & (HR >= self.mcw)
                    if mono[j] > 0:
                        ok &= wl <= wr
                    if mono[j] < 0:
                        ok &= wl >= wr
                    gain = np.where(ok, gain, -np.inf)
                    # among equal gains (empty bins in between): lowest threshold if missing goes right, highest if left (as XGBoost)
                    c = int(len(gain) - 1 - np.argmax(gain[::-1])) if left_missing else int(np.argmax(gain))
                    if gain[c] > best_gain * (1 + 1e-9) + 1e-12:     # an exact tie keeps the earlier feature
                        best_gain, best = gain[c], (j, c, left_missing, wl[c], wr[c])
            if best is None:
                continue
            j, c, left_missing, wl, wr = best
            feat[k], thr[k], miss_left[k], leaf[k] = j, c, left_missing, False
            b = B[idx, j]
            go_left = np.where(b == nb[j] - 1, left_missing, b <= c)
            L, R = 2 * k + 1, 2 * k + 2
            rows[L], rows[R] = idx[go_left], idx[~go_left]
            lo[L], hi[L], lo[R], hi[R] = lo[k], hi[k], lo[k], hi[k]
            mid = (wl + wr) / 2
            if mono[j] > 0:
                hi[L], lo[R] = mid, mid
            if mono[j] < 0:
                lo[L], hi[R] = mid, mid
        return {"feat": feat, "thr": thr, "miss_left": miss_left, "value": value, "leaf": leaf, "nb": np.asarray(nb)}

    def _apply(self, tree, B):
        node, row = np.zeros(B.shape[0], dtype=np.int64), np.arange(B.shape[0])
        for _level in range(self.depth):
            inner = ~tree["leaf"][node]
            fj = np.where(inner, tree["feat"][node], 0)
            b = B[row, fj]
            go_left = np.where(b == tree["nb"][fj] - 1, tree["miss_left"][node], b <= tree["thr"][node])
            node = np.where(inner, np.where(go_left, 2 * node + 1, 2 * node + 2), node)
        return tree["value"][node]

    def margin(self, X):
        B = self._bin(np.asarray(X, dtype=float))
        f = np.zeros(B.shape[0]) + (self.base if self.objective == "logistic" else 0.0)
        for tree in self.trees:
            f += self.eta * self._apply(tree, B)
        return f


class BoostedCox:
    """Protocol section 6, models 2 (XGB) and 3 (XGB-mono). Missing values are handled natively (section 5). With
    monotone=True the priors of section 7 that apply to the predictors become hard monotone constraints."""

    def __init__(self, cols, monotone=False):
        self.cols, self.monotone = list(cols), monotone

    def fit(self, df):
        mono = [PRIORS.get(c, 0) if self.monotone else 0 for c in self.cols]
        self.gbm = HistGBM("cox", HP["xgb_trees"], HP["xgb_depth"], HP["xgb_eta"], BOOST["lam"], BOOST["min_child_weight"], BOOST["max_bin"], mono)
        self.gbm.fit(df[self.cols].to_numpy(dtype=float), df["time_y"].to_numpy(), df["event"].to_numpy())
        return self

    def lp(self, df):
        return self.gbm.margin(df[self.cols].to_numpy(dtype=float))


MODELS["XGB"] = lambda cols, seed=0: BoostedCox(cols, monotone=False)
MODELS["XGB-mono"] = lambda cols, seed=0: BoostedCox(cols, monotone=True)

A.check("deviation D-6 (software on the tablet) is on the register of cell 3", any(d[0] == "D-6" for d in DEVIATIONS),
        "if this fails: replace cell 3 with its new version and run it")
A.check("boosting settings equal the locked ones: 300 trees, depth 3, learning rate 0.05", (HP["xgb_trees"], HP["xgb_depth"], HP["xgb_eta"]) == (300, 3, 0.05))

# ---------------------------------------------------------------- tests on toy arrays, against XGBoost 3.4.1 (tree_method hist, lambda 1, min_child_weight 1)
_d = toy_survival(250, ties=False)
_d.loc[::9, "x2"] = np.nan                                           # missing values in one predictor
_tr, _te = _d.iloc[:166], _d.iloc[166:]
_X, _Xt, _t, _e = _tr[["x1", "x2"]].to_numpy(), _te[["x1", "x2"]].to_numpy(), _tr["time_y"].to_numpy(), _tr["event"].to_numpy()
_m300 = HistGBM("cox", 300, 3, 0.05).fit(_X, _t, _e).margin(_Xt)
A.check("boosted Cox, 300 trees, toy data with missing values: equal to XGBoost on unseen rows to 5 decimals",
        np.allclose(_m300[:5], [-0.4353852, 0.7293296, -0.2031798, -0.9950941, 1.6108009], atol=1e-5) and abs(_m300.sum() - 4.609698) < 1e-4, np.round(_m300[:3], 6).tolist())
_gm = HistGBM("cox", 20, 3, 0.05, monotone=(1, -1)).fit(_X, _t, _e)
_m20 = _gm.margin(_Xt)
A.check("monotone boosted Cox, 20 trees: equal to XGBoost with monotone_constraints (1, -1) to 6 decimals",
        np.allclose(_m20[:5], [0.1433601, -0.0044428, -0.0167303, -0.1436132, 0.1890372], atol=1e-6) and abs(_m20.sum() + 1.384737) < 1e-5)
_y, _w = (_tr["time_y"] <= 3).astype(int).to_numpy(), 1 + (np.arange(len(_tr)) % 3) * 0.5
_ml = HistGBM("logistic", 300, 3, 0.05).fit(_X, y=_y, weight=_w).margin(_Xt)
A.check("boosted logistic model with weights, 300 trees: equal to XGBoost to 5 decimals",
        np.allclose(_ml[:5], [0.4211563, 1.8292954, -0.4477710, 2.1648521, 2.6879263], atol=1e-5) and abs(_ml.sum() - 91.229327) < 1e-4)
_grid = np.linspace(-1, 1, 81)
_mm = HistGBM("cox", 300, 3, 0.05, monotone=(1, -1)).fit(_X, _t, _e)
_up = np.array([_mm.margin(np.column_stack([_grid, np.full(81, c)])) for c in (-0.5, 0.0, 0.5, np.nan)])
_dn = np.array([_mm.margin(np.column_stack([np.full(81, c), _grid])) for c in (-0.5, 0.0, 0.5)])
A.check("monotone constraints hold everywhere on a grid: the log hazard never falls as x1 rises and never rises as x2 rises",
        (np.diff(_up, axis=1) >= 0).all() and (np.diff(_dn, axis=1) <= 0).all())
_free = HistGBM("cox", 300, 3, 0.05).fit(_X, _t, _e)
A.check("without constraints the same data give a non-monotone fit (so the constraint is doing something)",
        (np.diff(_free.margin(np.column_stack([_grid, np.zeros(81)]))) < 0).any())
A.check("the fit is deterministic: fitting twice gives identical predictions", np.array_equal(HistGBM("cox", 50, 3, 0.05).fit(_X, _t, _e).margin(_Xt), HistGBM("cox", 50, 3, 0.05).fit(_X, _t, _e).margin(_Xt)))
A.check("a missing value is accepted at prediction time and gives a finite log hazard", np.isfinite(_free.margin(np.array([[0.2, np.nan], [np.nan, np.nan]]))).all())
_tt = toy_survival(250)                                              # tied times: structural checks only (XGBoost approximates the tie gradient)
_ft = fit_with_baseline(BoostedCox(["x1", "x2"]), _tt.iloc[:166])
A.check("with tied times the model still fits, gives risks in (0, 1) and ranks unseen rows better than chance",
        ((_ft.risk(_tt.iloc[166:]) > 0) & (_ft.risk(_tt.iloc[166:]) < 1)).all() and concordance_index(_tt.iloc[166:]["time_y"], _ft.model.lp(_tt.iloc[166:]), _tt.iloc[166:]["event"]) > 0.5)

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2 (skipped while a session is being restored)
if not restoring():
    _cols, _rows = predictors("RICH", "T2"), []
    for _name in ("XGB", "XGB-mono"):
        _t0 = time.time()
        _fit = fit_model(_name, _cols, SMOKE_TRAIN)
        _mt = evaluate(_fit, SMOKE_TEST)
        _rows.append({"model": _name, "seconds": round(time.time() - _t0), **{k: _mt[k] for k in ("events", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate")}})
        A.check(f"{_name}: risks strictly between 0 and 1; C-index over 0.7; Brier score beats the uninformative prediction",
                ((_fit.risk(SMOKE_TEST) > 0) & (_fit.risk(SMOKE_TEST) < 1)).all() and _mt["cindex"] > 0.7 and _mt["scaled_brier"] > 0, round(_mt["cindex"], 4))
        if _name == "XGB-mono":
            A.check("XGB-mono: constraints on age, BMI, SBP and triglycerides; its violation rate is exactly 0",
                    [PRIORS.get(c, 0) for c in _cols] == [1, 0, 1, 1, 0, 0, 1, 0] and _mt["viol_rate"] == 0.0)
    print("\n  RICH tier T2, fold (repeat 0, fold 0); one fold only, the full internal validation is stage G:")
    print(pd.DataFrame(_rows).round(4).to_string(index=False))
    print(f"\n  Time per fit here: about {int(np.mean([r['seconds'] for r in _rows]))} s. Stage G fits these two models 120 times (60 folds), in resumable units.")
_ = A.close(outputs={"models": ["XGB", "XGB-mono"], "settings": {**BOOST, "trees": HP["xgb_trees"], "depth": HP["xgb_depth"], "eta": HP["xgb_eta"]},
                     "reference": "XGBoost 3.4.1 (toy data)"})
keep_for_session(31)

# %%
# CELL 32 - random survival forest                                                              [status A: amendment item A1.2; proposal p. 17]
# Why: the proposal names random forest. Its survival version needs the scikit-survival package, which cannot be
# installed on this tablet. The cell reports that honestly; the random-forest family is represented in cell 34 by a
# forest classifier on the 3-year endpoint. Nothing is substituted for a model that cannot be run.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 32)
A = Audit(32, "Random survival forest", status="A")
RSF_STATUS = "not approved"
if not skip_unapproved(A, "A1.2", "the random survival forest"):
    try:
        import sksurv                                                 # noqa: F401
        RSF_STATUS = "package present: this cell must be completed before use"
    except Exception:
        RSF_STATUS = "not run: scikit-survival is not installed on this machine"
    print(f"  Random survival forest: {RSF_STATUS}.")
    print("  It is recorded as 'not run on this machine' in the comparison tables. To include it, the notebook has to be")
    print("  run once on a computer where scikit-survival can be installed.")
    A.check("the status of the random survival forest is recorded, with no substitute model", RSF_STATUS.startswith(("not run", "package present")))
_ = A.close(outputs={"rsf_status": RSF_STATUS})
keep_for_session(32)

# %%
# CELL 33 - the 3-year classification endpoint, with censoring weights                           [status A: amendment item A1.2; proposal p. 17]
# Why: logistic regression, random forest, boosted classifiers and SVM predict a yes/no label, and have no way of using
# a participant who was censored before 3 years. A valid label is: event by 3 years = 1; followed beyond 3 years = 0;
# censored earlier = unknown, weight 0, with the others weighted up by the inverse probability of remaining uncensored.
# Nobody censored early is ever labelled a non-event. The function is defined and tested on toy arrays here.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 33)
A = Audit(33, "Three-year classification endpoint", status="A")


def status_at_horizon(train, h=H):
    """For the rows of a training set: (known, y, w). known = status at h is known; y = 1 if event by h; w = censoring
    weight (0 where the status is unknown). The censoring distribution is estimated on the same training rows."""
    case, control, w = ipcw(train, train, h)
    return case | control, case.astype(float), w


# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(300)
_known, _y, _w = status_at_horizon(_d, 3.0)
_cens_early = ((_d["event"] == 0) & (_d["time_y"] <= 3.0)).to_numpy()
A.check("toy data: 129 events by 3 years, 95 followed beyond, 76 censored before; the three groups cover everyone",
        (int(_y.sum()), int((_known & (_y == 0)).sum()), int((~_known).sum())) == (129, 95, 76) and int(_known.sum()) + int((~_known).sum()) == 300)
A.check("a participant censored before 3 years is never labelled: weight 0, excluded from fitting", (_w[_cens_early] == 0).all() and (~_known[_cens_early]).all())
A.check("the weighted share of events equals the Kaplan-Meier 3-year risk (the weights recover what censoring hid)",
        abs((_w * _y).sum() / _w.sum() - km_risk(_d, 3.0)) < 0.01, f"{(_w * _y).sum() / _w.sum():.4f} against {km_risk(_d, 3.0):.4f}")
A.check("the naive share of events among everyone (censored counted as non-events) is too low", _y.mean() < km_risk(_d, 3.0) - 0.02,
        f"{_y.mean():.3f} against {km_risk(_d, 3.0):.3f}")
ENDPOINT_COUNTS = None
if not skip_unapproved(A, "A1.2", "the 3-year endpoint count in RICH and NAGALA"):
    _rows = []
    for _c in ("RICH", "NAGALA"):
        _k, _yy, _ww = status_at_horizon(HARM[_c])
        _rows.append({"cohort": _c, "known_status": int(_k.sum()), "events_by_3y": int(_yy.sum()), "followed_beyond_3y": int((_k & (_yy == 0)).sum()),
                      "unknown_weight_0": int((~_k).sum()), "weighted_event_percent": round(100 * (_ww * _yy).sum() / _ww.sum(), 3), "km_risk_percent": round(100 * km_risk(HARM[_c]), 3)})
    ENDPOINT_COUNTS = pd.DataFrame(_rows)
    print(ENDPOINT_COUNTS.to_string(index=False))
    A.check("RICH: status at 3 years known for 105,778; unknown (weight 0) for 106,055", ENDPOINT_COUNTS.loc[0, ["known_status", "unknown_weight_0"]].tolist() == [105778, 106055])
    A.check("in both cohorts the weighted event share is within 0.05 percentage points of the Kaplan-Meier risk",
            (ENDPOINT_COUNTS["weighted_event_percent"] - ENDPOINT_COUNTS["km_risk_percent"]).abs().max() < 0.05)
_ = A.close(outputs={"endpoint": "event by 3 years, censoring weights", "counts": None if ENDPOINT_COUNTS is None else ENDPOINT_COUNTS.to_dict("records")})
keep_for_session(33)

# %%
# CELL 34 - classifiers on the 3-year endpoint: logistic regression, random forest, boosted trees   [status A: amendment item A1.2; proposal p. 17]
# Why: the proposal's comparison list. On this tablet: logistic regression and random forest from scikit-learn; the
# XGBoost-type model is the NumPy engine of cell 31 with a logistic loss; the LightGBM-type model is scikit-learn's
# histogram gradient boosting (the same leaf-wise algorithm). Settings are fixed, not tuned. These models are exploratory
# comparators: they are not part of the "best baseline" of claim C1.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 34)
A = Audit(34, "Weighted classifiers on the 3-year endpoint", status="A")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

CLASSIFIER_SETTINGS = {"LR": "L2, C = 1", "RF": "300 trees, minimum leaf 50", "GB-xgb": "300 trees, depth 3, rate 0.05, lambda 1, min child weight 20",
                       "GB-lgbm": "300 trees, 8 leaves, rate 0.05"}


class WeightedClassifier:
    """A classifier for 'event by 3 years', fitted on the training rows whose status is known, with censoring weights.
    risk3() is the predicted 3-year risk; lp() is its complementary log-log, used only for ranking and violation checks."""

    def __init__(self, cols, kind, seed=0):
        self.cols, self.kind, self.seed = list(cols), kind, seed

    def _X(self, df):
        return self.prep.transform(df) if self.kind in ("LR", "RF") else df[self.cols].to_numpy(dtype=float)

    def fit(self, df):
        known, y, w = status_at_horizon(df)
        if self.kind in ("LR", "RF"):
            self.prep = Prep(self.cols).fit(df)                      # median or mode fill, indicators, scaling (section 5)
        d, y, w = df[known], y[known], w[known] / w[known].mean()
        X = self._X(d)
        if self.kind == "LR":
            self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y, sample_weight=w)
        elif self.kind == "RF":
            self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, random_state=self.seed, n_jobs=1).fit(X, y, sample_weight=w)
        elif self.kind == "GB-lgbm":
            self.clf = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=8, learning_rate=0.05, early_stopping=False, random_state=self.seed).fit(X, y, sample_weight=w)
        elif self.kind == "GB-xgb":
            self.clf = HistGBM("logistic", 300, 3, 0.05, BOOST["lam"], BOOST["min_child_weight"], BOOST["max_bin"]).fit(X, y=y, weight=w)
        else:
            raise KeyError(self.kind)
        self.n_fit_, self.events_fit_ = int(known.sum()), int(y.sum())
        return self

    def risk3(self, df):
        X = self._X(df)
        return 1.0 / (1.0 + np.exp(-self.clf.margin(X))) if self.kind == "GB-xgb" else self.clf.predict_proba(X)[:, 1]

    def lp(self, df):
        return cloglog(self.risk3(df))


for _k in CLASSIFIER_SETTINGS:
    MODELS[_k] = (lambda kind: lambda cols, seed=0: WeightedClassifier(cols, kind, seed))(_k)

# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(600)
_d.loc[::9, "x2"] = np.nan
_tr, _te = _d.iloc[:400], _d.iloc[400:]
_te = _te[_te["time_y"] < _tr["time_y"].max()]
_res = {}
for _k in CLASSIFIER_SETTINGS:
    _f = fit_model(_k, ["x1", "x2"], _tr, seed=0)
    _r = _f.risk(_te)
    _res[_k] = (bool(((_r >= 0) & (_r <= 1)).all()), auc_at(_tr, _tr, _f.model.lp(_tr)), _f.model.n_fit_, float(np.mean(_r)), auc_at(_tr, _te, _f.model.lp(_te)))
A.check("every classifier fits on the toy rows with known status only (293 of 400) and returns risks between 0 and 1",
        all(v[0] and v[2] == 293 for v in _res.values()), {k: v[2] for k, v in _res.items()})
A.check("every classifier has learned from its training rows (training AUC over 0.6); logistic regression also ranks unseen rows above chance",
        all(v[1] > 0.6 for v in _res.values()) and _res["LR"][4] > 0.55, {k: (round(v[1], 3), round(v[4], 3)) for k, v in _res.items()})
A.check("mean predicted risk is near the Kaplan-Meier 3-year risk of the toy test rows, not the naive (too low) event share",
        all(abs(v[3] - km_risk(_te, 3.0)) < 0.12 for v in _res.values()), {k: round(v[3], 3) for k, v in _res.items()})
try:
    fit_model("LR", ["x1", "x2"], _tr).risk(_te, h=5.0)
    _refused = False
except ValueError:
    _refused = True
A.check("a classifier gives the risk by 3 years only; asking for another horizon raises an error", _refused)
A.check("the same seed gives the same forest", np.array_equal(fit_model("RF", ["x1", "x2"], _tr, seed=3).risk(_te), fit_model("RF", ["x1", "x2"], _tr, seed=3).risk(_te)))

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2
if not skip_unapproved(A, "A1.2", "the classifier smoke test on the RICH fold") and not restoring():
    _cols, _rows = predictors("RICH", "T2"), []
    for _k in CLASSIFIER_SETTINGS:
        _t0 = time.time()
        _fit = fit_model(_k, _cols, SMOKE_TRAIN, seed=0)
        _mt = evaluate(_fit, SMOKE_TEST)
        _rows.append({"model": _k, "seconds": round(time.time() - _t0), "rows_used": _fit.model.n_fit_, **{k: _mt[k] for k in ("cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate")}})
        A.check(f"{_k}: risks between 0 and 1 and AUC over 0.7 on the test fold", ((_fit.risk(SMOKE_TEST) >= 0) & (_fit.risk(SMOKE_TEST) <= 1)).all() and _mt["auc_3y"] > 0.7, round(_mt["auc_3y"], 4))
    print("\n  RICH tier T2, fold (repeat 0, fold 0); exploratory comparators, one fold only:")
    print(pd.DataFrame(_rows).round(4).to_string(index=False))
_ = A.close(outputs={"classifiers": CLASSIFIER_SETTINGS})
keep_for_session(34)

# %%
# CELL 35 - support vector machines on the 3-year endpoint                                        [status A: amendment item A1.2; proposal p. 17]
# Why: SVM is in the proposal's comparison list. An exact kernel SVM does not scale to 85,000 training rows, so two
# feasible versions are used: a linear SVM, and a radial-kernel SVM through the Nystroem approximation (300 components).
# An SVM gives a score, not a probability; the score is turned into a 3-year risk by weighted Platt scaling fitted on
# an inner 20% of the training rows, never on test rows.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 35)
A = Audit(35, "Support vector machines with Platt scaling", status="A")
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem


class WeightedSVM:
    """Linear SVM (kind 'SVM-linear') or Nystroem radial-kernel SVM (kind 'SVM-rbf') for 'event by 3 years', with
    censoring weights, C = 1, and Platt scaling on an inner 20% split."""

    def __init__(self, cols, kind, seed=0):
        self.cols, self.kind, self.seed = list(cols), kind, seed

    def _features(self, df):
        Z = self.prep.transform(df)
        return self.map.transform(Z) if self.kind == "SVM-rbf" else Z

    def fit(self, df):
        known, y, w = status_at_horizon(df)
        self.prep = Prep(self.cols).fit(df)
        d = df[known].assign(_y=y[known], _w=w[known] / w[known].mean())
        fit_part, cal_part = next(iter(StratifiedKFold(5, shuffle=True, random_state=self.seed + 1000).split(d, d["_y"])))
        fit_part, cal_part = d.iloc[fit_part], d.iloc[cal_part]
        if self.kind == "SVM-rbf":
            self.map = Nystroem(kernel="rbf", gamma=1.0 / len(self.prep.names), n_components=min(300, len(fit_part)), random_state=self.seed).fit(self.prep.transform(fit_part))
        self.svm = LinearSVC(C=1.0, dual=False, max_iter=5000).fit(self._features(fit_part), fit_part["_y"], sample_weight=fit_part["_w"])
        score = self.svm.decision_function(self._features(cal_part))
        self.platt = LogisticRegression(C=1e6, max_iter=2000).fit(score[:, None], cal_part["_y"], sample_weight=cal_part["_w"])
        self.n_fit_, self.n_cal_ = len(fit_part), len(cal_part)
        return self

    def risk3(self, df):
        return self.platt.predict_proba(self.svm.decision_function(self._features(df))[:, None])[:, 1]

    def lp(self, df):
        return cloglog(self.risk3(df))


MODELS["SVM-linear"] = lambda cols, seed=0: WeightedSVM(cols, "SVM-linear", seed)
MODELS["SVM-rbf"] = lambda cols, seed=0: WeightedSVM(cols, "SVM-rbf", seed)

# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(600)
_d.loc[::9, "x2"] = np.nan
_tr, _te = _d.iloc[:400], _d.iloc[400:]
_te = _te[_te["time_y"] < _tr["time_y"].max()]
_res = {}
for _k in ("SVM-linear", "SVM-rbf"):
    _f = fit_model(_k, ["x1", "x2"], _tr, seed=0)
    _r = _f.risk(_te)
    _res[_k] = (bool(((_r > 0) & (_r < 1)).all()), auc_at(_tr, _tr, _f.model.lp(_tr)), _f.model.n_fit_, _f.model.n_cal_, float(_f.model.platt.coef_[0, 0]), float(np.mean(_r)))
A.check("both SVMs fit on 80% of the toy rows with known status and are scaled on the other 20% (234 and 59)",
        all(v[0] and (v[2], v[3]) == (234, 59) for v in _res.values()), {k: (v[2], v[3]) for k, v in _res.items()})
A.check("both SVMs have learned from their training rows (training AUC over 0.55), and a higher score means a higher risk (positive Platt slope)",
        all(v[1] > 0.55 and v[4] > 0 for v in _res.values()), {k: (round(v[1], 3), round(v[4], 3)) for k, v in _res.items()})
A.check("Platt scaling returns probabilities near the observed 3-year risk on average", all(abs(v[5] - km_risk(_te, 3.0)) < 0.12 for v in _res.values()), {k: round(v[5], 3) for k, v in _res.items()})

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2
if not skip_unapproved(A, "A1.2", "the SVM smoke test on the RICH fold") and not restoring():
    _cols, _rows = predictors("RICH", "T2"), []
    for _k in ("SVM-linear", "SVM-rbf"):
        _t0 = time.time()
        _fit = fit_model(_k, _cols, SMOKE_TRAIN, seed=0)
        _mt = evaluate(_fit, SMOKE_TEST)
        _rows.append({"model": _k, "seconds": round(time.time() - _t0), "rows_fit": _fit.model.n_fit_, "rows_platt": _fit.model.n_cal_, **{k: _mt[k] for k in ("cindex", "auc_3y", "scaled_brier", "oe", "cal_slope")}})
        A.check(f"{_k}: risks strictly between 0 and 1 and AUC over 0.65 on the test fold", ((_fit.risk(SMOKE_TEST) > 0) & (_fit.risk(SMOKE_TEST) < 1)).all() and _mt["auc_3y"] > 0.65, round(_mt["auc_3y"], 4))
    print("\n  RICH tier T2, fold (repeat 0, fold 0); exploratory comparators, one fold only:")
    print(pd.DataFrame(_rows).round(4).to_string(index=False))
_ = A.close(outputs={"svm": {"C": 1.0, "nystroem_components": 300, "platt_split": "inner 20%"}})
keep_for_session(35)

# %%
# CELL 36 - GATE E: baseline models                                                              [status L]
# Why: a stop point before the network and CKINN. It confirms that every baseline is defined, tested and registered,
# states which exploratory comparators are active, and confirms that no result file has been written yet: the smoke
# tests of this stage printed one fold only and stored nothing.
_g = __import__("pathlib").Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 2 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 36)
require_gate("D")
A = Audit("gate_E", "Gate E: baseline models")

_recs = {c: load_record(c) for c in range(30, 36)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else f"{len(r['checks'])} checks, {r['time']}")
LOCKED_BASELINES = ["Cox", "XGB", "XGB-mono"]
EXPLORATORY_COMPARATORS = ["LR", "RF", "GB-xgb", "GB-lgbm", "SVM-linear", "SVM-rbf"]
A.check("the three locked baselines are registered", all(m in MODELS for m in LOCKED_BASELINES))
A.check("the six exploratory comparators are defined; they enter an analysis only if A1.2 is approved", all(m in MODELS for m in EXPLORATORY_COMPARATORS))
ACTIVE_MODELS = LOCKED_BASELINES + (EXPLORATORY_COMPARATORS if approved("A1.2") else [])
_toy = toy_survival(300)
for _m in LOCKED_BASELINES:
    _f = fit_model(_m, ["x1", "x2"], _toy.iloc[:200])
    A.check(f"{_m}: fits the toy rows through the registry and returns a log hazard and a 3-year risk for unseen rows",
            np.isfinite(_f.model.lp(_toy.iloc[200:])).all() and ((_f.risk(_toy.iloc[200:]) > 0) & (_f.risk(_toy.iloc[200:]) < 1)).all())
A.check("the study data are unchanged by stage E", all(content_sha(HARM[k]) == HARM_SHA[k] for k in HARM))
A.check("no result part or prediction file has been written yet", not any(p.suffix == ".csv" for p in DIR["parts"].iterdir()) and not any(DIR["preds"].iterdir()))
print(f"\n  Locked baselines: {LOCKED_BASELINES}")
print(f"  Exploratory comparators (amendment A1.2): {'ACTIVE' if approved('A1.2') else 'defined but not active, because A1.2 is not approved'}: {EXPLORATORY_COMPARATORS}")
print(f"  Random survival forest: {RSF_STATUS}.")
print("  Deviation D-6: boosted Cox and (in stage F) the network are fitted with NumPy code instead of XGBoost and PyTorch.")
_ = A.close(outputs={"stage": "E", "locked_baselines": LOCKED_BASELINES, "active_models": ACTIVE_MODELS, "rsf_status": RSF_STATUS},
            extra_path=DIR["gates"] / "gate_E.json")
print("\n  GATE E PASSED. Stage F (cells 37 to 42, network and CKINN) may start.")
