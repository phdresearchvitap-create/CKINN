# %%
# CELL 30 - baseline 1: Cox regression with ridge penalty 0.01                                    [status L; protocol section 6; proposal p. 17]
# Why: Cox regression is the counterpart of logistic regression for censored follow-up, and the first comparator of
# claim C1. This cell defines the model, checks it against lifelines on toy data, then fits it once to one fold of
# RICH (tier T1) as a smoke test. That fold is unit (repeat 0, fold 0) of the internal validation run in stage G.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
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
    # the same model fitted by lifelines on the same prepared columns of the real fold (implementation check)
    from lifelines import CoxPHFitter
    _Zs = pd.DataFrame(_fit.model.prep.transform(SMOKE_TRAIN), columns=_fit.model.prep.names)
    _Zs["T"], _Zs["E"] = SMOKE_TRAIN["time_y"].to_numpy(), SMOKE_TRAIN["event"].to_numpy()
    _ll = CoxPHFitter(penalizer=HP["cox_ridge"]).fit(_Zs, "T", "E").params_.reindex(_fit.model.prep.names).to_numpy()
    A.check("live: lifelines fitted to the same RICH fold gives the same coefficients (4 decimals)", np.allclose(_fit.model.beta, _ll, atol=1e-4),
            float(np.abs(_fit.model.beta - _ll).max()))
_ = A.close(outputs={"model": "Cox", "penalizer": HP["cox_ridge"], "reference": "lifelines 0.30.3 (toy data, stored) and installed lifelines (live)"})
keep_for_session(30)

# %%
# CELL 31 - baselines 2 and 3: gradient-boosted Cox (XGBoost), without and with monotone constraints     [status L; protocol sections 5-7]
# Why: boosted trees are the strongest conventional comparator (XGB), and the constrained version (XGB-mono) imposes the
# clinical priors as hard monotone constraints, the conventional way of doing so: CKINN has to be compared with it.
# Settings are those of the locked reference code: 300 trees, depth 3, learning rate 0.05, row and column subsampling
# 0.8, minimum child weight 20, histogram method, seed = model seed, one thread. Missing values are handled by XGBoost
# itself (protocol section 5). This cell tests the models on toy data, then fits both once to the RICH smoke fold.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 31)
A = Audit(31, "Gradient-boosted Cox (XGBoost), with and without monotone constraints")
import xgboost as xgb


class BoostedCox:
    """Protocol section 6, models 2 (XGB) and 3 (XGB-mono). lp() is XGBoost's margin, the log relative hazard. With
    monotone=True every predictor that has a prior (section 7) gets a hard monotone constraint in its direction."""

    def __init__(self, cols, monotone=False, seed=0, priors=None):
        self.cols, self.monotone, self.seed = list(cols), monotone, seed
        self.priors = PRIORS if priors is None else priors

    def _X(self, df):
        return df[self.cols].to_numpy(np.float32)                    # NaN is passed through: XGBoost learns where it goes

    def params(self):
        p = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "eta": HP["xgb_eta"], "max_depth": HP["xgb_depth"],
             "subsample": HP["xgb_subsample"], "colsample_bytree": HP["xgb_colsample"], "min_child_weight": HP["xgb_min_child_weight"],
             "tree_method": HP["xgb_tree_method"], "seed": self.seed, "nthread": ENV["threads"]}
        if self.monotone:
            p["monotone_constraints"] = "(" + ",".join(str(self.priors.get(c, 0)) for c in self.cols) + ")"
        return p

    def fit(self, df):
        y = np.where(df["event"].to_numpy() == 1, df["time_y"].to_numpy(), -df["time_y"].to_numpy())   # XGBoost Cox label: negative = censored
        self.bst = xgb.train(self.params(), xgb.DMatrix(self._X(df), label=y), num_boost_round=HP["xgb_trees"])
        return self

    def lp(self, df):
        return self.bst.predict(xgb.DMatrix(self._X(df)), output_margin=True).astype(np.float64)


MODELS["XGB"] = lambda cols, seed=0: BoostedCox(cols, monotone=False, seed=seed)
MODELS["XGB-mono"] = lambda cols, seed=0: BoostedCox(cols, monotone=True, seed=seed)

A.check(f"XGBoost {xgb.__version__} is in use, with the locked settings: 300 trees, depth 3, rate 0.05, subsampling 0.8 and 0.8, minimum child weight 20",
        (HP["xgb_trees"], HP["xgb_depth"], HP["xgb_eta"], HP["xgb_subsample"], HP["xgb_colsample"], HP["xgb_min_child_weight"]) == (300, 3, 0.05, 0.8, 0.8, 20))

# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(6000)                                              # boosted trees need a few thousand rows to beat noise on these toy data
_d.loc[::9, "x2"] = np.nan                                           # missing values in one predictor
_tr, _te = _d.iloc[:4000], _d.iloc[4000:]
_pri = {"x1": +1, "x2": -1}
_mono = BoostedCox(["x1", "x2"], monotone=True, seed=0, priors=_pri).fit(_tr)
_free = BoostedCox(["x1", "x2"], monotone=False, seed=0, priors=_pri).fit(_tr)
A.check("the constraint string follows the priors: '(1,-1)' for x1 up and x2 down; none without monotone", _mono.params()["monotone_constraints"] == "(1,-1)"
        and "monotone_constraints" not in _free.params())
_grid = np.linspace(-1, 1, 81)
_up = np.array([_mono.lp(pd.DataFrame({"x1": _grid, "x2": np.full(81, c)})) for c in (-0.5, 0.0, 0.5, np.nan)])
_dn = np.array([_mono.lp(pd.DataFrame({"x1": np.full(81, c), "x2": _grid})) for c in (-0.5, 0.0, 0.5)])
A.check("monotone constraints hold everywhere on a grid: the log hazard never falls as x1 rises and never rises as x2 rises",
        (np.diff(_up, axis=1) >= 0).all() and (np.diff(_dn, axis=1) <= 0).all())
A.check("without constraints the same data give a non-monotone fit (so the constraint is doing something)",
        (np.diff(_free.lp(pd.DataFrame({"x1": _grid, "x2": np.zeros(81)}))) < 0).any() or (np.diff(_free.lp(pd.DataFrame({"x1": np.zeros(81), "x2": _grid}))) > 0).any())
A.check("the same seed gives identical predictions; another seed gives different ones (row and column subsampling)",
        np.array_equal(BoostedCox(["x1", "x2"], seed=0).fit(_tr).lp(_te), _free.lp(_te)) and not np.array_equal(BoostedCox(["x1", "x2"], seed=1).fit(_tr).lp(_te), _free.lp(_te)))
A.check("a missing value is accepted at prediction time and gives a finite log hazard", np.isfinite(_free.lp(pd.DataFrame({"x1": [0.2, np.nan], "x2": [np.nan, np.nan]}))).all())
_ft = fit_with_baseline(BoostedCox(["x1", "x2"]), _tr)
A.check("risks lie strictly between 0 and 1 and the model ranks unseen toy rows better than chance",
        ((_ft.risk(_te) > 0) & (_ft.risk(_te) < 1)).all() and concordance_index(_te["time_y"], _ft.model.lp(_te), _te["event"]) > 0.55,
        round(concordance_index(_te["time_y"], _ft.model.lp(_te), _te["event"]), 3))
A.check("the constrained toy model violates neither prior (violation rate exactly 0)", violation_rate(_mono, _te, priors=_pri)[0] == 0.0)

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2 (skipped while a session is being restored)
if not restoring():
    _cols, _rows = predictors("RICH", "T2"), []
    for _name in ("XGB", "XGB-mono"):
        _t0 = time.time()
        _fit = fit_model(_name, _cols, SMOKE_TRAIN, seed=0)
        _mt = evaluate(_fit, SMOKE_TEST)
        _rows.append({"model": _name, "seconds": round(time.time() - _t0), **{k: _mt[k] for k in ("events", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate")}})
        A.check(f"{_name}: risks strictly between 0 and 1; C-index over 0.7; Brier score beats the uninformative prediction",
                ((_fit.risk(SMOKE_TEST) > 0) & (_fit.risk(SMOKE_TEST) < 1)).all() and _mt["cindex"] > 0.7 and _mt["scaled_brier"] > 0, round(_mt["cindex"], 4))
        if _name == "XGB-mono":
            A.check("XGB-mono: constraints on age, BMI, SBP and triglycerides only; its violation rate is exactly 0",
                    _fit.model.params()["monotone_constraints"] == "(1,0,1,1,0,0,1,0)" and _mt["viol_rate"] == 0.0)
    print("\n  RICH tier T2, fold (repeat 0, fold 0); one fold only, the full internal validation is stage G:")
    print(pd.DataFrame(_rows).round(4).to_string(index=False))
_ = A.close(outputs={"models": ["XGB", "XGB-mono"], "xgboost": xgb.__version__,
                     "settings": {k: HP[k] for k in ("xgb_trees", "xgb_depth", "xgb_eta", "xgb_subsample", "xgb_colsample", "xgb_min_child_weight", "xgb_tree_method")}})
keep_for_session(31)

# %%
# CELL 32 - random survival forest (scikit-survival)                                            [status A: amendment item A1.2; proposal p. 17]
# Why: the proposal names random forest; for censored follow-up its proper form is the random survival forest. Settings
# (amendment A1.2): 200 trees, at least 50 participants per leaf, each tree grown on 20,000 participants drawn with
# replacement; the other settings are scikit-survival's defaults. Missing values: training-fold median or mode plus
# missingness indicators (cell 22). It is an exploratory comparator, not part of the "best baseline" of claim C1.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 32)
A = Audit(32, "Random survival forest", status="A")
import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

RSF_SETTINGS = {"n_estimators": 200, "min_samples_leaf": 50, "max_samples": 20000, "bootstrap": True}


class SurvivalForest:
    """3-year risk from the forest's survival function at the training event times. lp() = log(-log(1 - risk)), used
    for ranking and violation checks, as for the other direct-risk models."""

    def __init__(self, cols, seed=0, n_estimators=None):
        self.cols, self.seed = list(cols), seed
        self.n_estimators = RSF_SETTINGS["n_estimators"] if n_estimators is None else n_estimators

    def fit(self, df):
        self.prep = Prep(self.cols).fit(df)
        y = Surv.from_arrays(df["event"].astype(bool).to_numpy(), df["time_y"].to_numpy(dtype=float))
        self.rsf = RandomSurvivalForest(n_estimators=self.n_estimators, min_samples_leaf=RSF_SETTINGS["min_samples_leaf"],
                                        max_samples=min(RSF_SETTINGS["max_samples"], len(df)), bootstrap=True,
                                        random_state=self.seed, n_jobs=ENV["threads"]).fit(self.prep.transform(df), y)
        self.times_ = np.asarray(getattr(self.rsf, "unique_times_", getattr(self.rsf, "event_times_", None)))
        return self

    def survival_at(self, df, h=H, chunk=5000):
        k = int(np.searchsorted(self.times_, h, side="right")) - 1
        X = self.prep.transform(df)
        if k < 0:
            return np.ones(len(X))
        key = (hashlib.sha256(X.tobytes()).hexdigest(), k)        # the forest is slow to predict: reuse the last few results
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = np.concatenate([self.rsf.predict_survival_function(X[i:i + chunk], return_array=True)[:, k] for i in range(0, len(X), chunk)])
        return cache[key].copy()

    def risk3(self, df):
        return 1.0 - self.survival_at(df, H)

    def lp(self, df):
        return cloglog(self.risk3(df))


MODELS["RSF"] = lambda cols, seed=0: SurvivalForest(cols, seed)
RSF_STATUS = f"defined with scikit-survival {sksurv.__version__}"

# ---------------------------------------------------------------- tests on toy arrays
_d = toy_survival(600)
_d.loc[::9, "x2"] = np.nan
_tr, _te = _d.iloc[:400], _d.iloc[400:]
_te = _te[_te["time_y"] < _tr["time_y"].max()]
_f = fit_model("RSF", ["x1", "x2"], _tr, seed=0)
_r = _f.risk(_te)
A.check("toy forest: 200 trees, each on all 400 toy rows drawn with replacement (fewer than 20,000), risks between 0 and 1",
        len(_f.model.rsf.estimators_) == 200 and ((_r >= 0) & (_r <= 1)).all())
A.check("survival never increases with time: S(2) >= S(3) >= S(5) for every toy row",
        (_f.model.survival_at(_te, 2.0) >= _f.model.survival_at(_te, 3.0) - 1e-12).all() and (_f.model.survival_at(_te, 3.0) >= _f.model.survival_at(_te, 5.0) - 1e-12).all())
A.check("the forest ranks unseen toy rows better than chance (AUC over 0.55)", auc_at(_tr, _te, _f.model.lp(_te)) > 0.55, round(auc_at(_tr, _te, _f.model.lp(_te)), 3))
A.check("the same seed gives the same forest", np.array_equal(fit_model("RSF", ["x1", "x2"], _tr, seed=0).risk(_te), _r))

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2 (about 2 minutes)
if not skip_unapproved(A, "A1.2", "the random survival forest") and not restoring():
    _cols, _t0 = predictors("RICH", "T2"), time.time()
    _fit = fit_model("RSF", _cols, SMOKE_TRAIN, seed=0)
    _mt = evaluate(_fit, SMOKE_TEST)
    print(f"\n  Random survival forest, RICH tier T2, fold (repeat 0, fold 0); exploratory comparator, one fold only ({time.time() - _t0:.0f} s):")
    print(pd.Series({k: _mt[k] for k in ("n", "events", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate")}).round(4).to_string())
    A.check("RSF: risks between 0 and 1, C-index over 0.7, Brier score beats the uninformative prediction",
            ((_fit.risk(SMOKE_TEST) >= 0) & (_fit.risk(SMOKE_TEST) <= 1)).all() and _mt["cindex"] > 0.7 and _mt["scaled_brier"] > 0, round(_mt["cindex"], 4))
    RSF_STATUS = f"fitted with scikit-survival {sksurv.__version__}"
_ = A.close(outputs={"rsf_status": RSF_STATUS, "settings": RSF_SETTINGS})
keep_for_session(32)

# %%
# CELL 33 - the 3-year classification endpoint, with censoring weights                           [status A: amendment item A1.2; proposal p. 17]
# Why: logistic regression, random forest, boosted classifiers and SVM predict a yes/no label, and have no way of using
# a participant who was censored before 3 years. A valid label is: event by 3 years = 1; followed beyond 3 years = 0;
# censored earlier = unknown, weight 0, with the others weighted up by the inverse probability of remaining uncensored.
# Nobody censored early is ever labelled a non-event. The function is defined and tested on toy arrays here.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
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
# Why: the proposal's comparison list (p. 17): logistic regression and random forest (scikit-learn), XGBoost and
# LightGBM classifiers. Settings are those of amendment A1.2, fixed, not tuned. XGBoost and LightGBM handle missing
# values themselves; the other two use the training-fold fill of cell 22. These are exploratory comparators: they are
# not part of the "best baseline" of claim C1.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 34)
A = Audit(34, "Weighted classifiers on the 3-year endpoint", status="A")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

CLASSIFIER_SETTINGS = {"LR": "L2, C = 1", "RF": "300 trees, minimum leaf 50", "XGB-clf": "XGBoost, 300 trees, depth 3, rate 0.05",
                       "LGBM-clf": "LightGBM, 300 trees, 8 leaves, rate 0.05"}


class WeightedClassifier:
    """A classifier for 'event by 3 years', fitted on the training rows whose status is known, with censoring weights.
    risk3() is the predicted 3-year risk; lp() is its complementary log-log, used only for ranking and violation checks."""

    def __init__(self, cols, kind, seed=0):
        self.cols, self.kind, self.seed = list(cols), kind, seed

    def _X(self, df):
        return self.prep.transform(df) if self.kind in ("LR", "RF") else df[self.cols].astype(float)    # XGBoost, LightGBM: NaN kept

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
        elif self.kind == "XGB-clf":
            self.clf = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, tree_method="hist", random_state=self.seed,
                                         n_jobs=ENV["threads"]).fit(X, y, sample_weight=w)
        elif self.kind == "LGBM-clf":
            self.clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05, random_state=self.seed, n_jobs=ENV["threads"],
                                          deterministic=True, force_row_wise=True, verbose=-1).fit(X, y, sample_weight=w)
        else:
            raise KeyError(self.kind)
        self.n_fit_, self.events_fit_ = int(known.sum()), int(y.sum())
        return self

    def risk3(self, df):
        return self.clf.predict_proba(self._X(df))[:, 1]

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
A.check("the same seed gives the same forest, the same XGBoost and the same LightGBM model",
        all(np.array_equal(fit_model(k, ["x1", "x2"], _tr, seed=3).risk(_te), fit_model(k, ["x1", "x2"], _tr, seed=3).risk(_te)) for k in ("RF", "XGB-clf", "LGBM-clf")))
A.check(f"the libraries in use: XGBoost {xgb.__version__}, LightGBM {lgb.__version__}", True)

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
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
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
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 36)
require_gate("D")
A = Audit("gate_E", "Gate E: baseline models")

_recs = {c: load_record(c) for c in range(30, 36)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else f"{len(r['checks'])} checks, {r['time']}")
LOCKED_BASELINES = ["Cox", "XGB", "XGB-mono"]
EXPLORATORY_COMPARATORS = ["RSF", "LR", "RF", "XGB-clf", "LGBM-clf", "SVM-linear", "SVM-rbf"]
A.check("the three locked baselines are registered", all(m in MODELS for m in LOCKED_BASELINES))
A.check("the seven exploratory comparators are defined; they enter an analysis only if A1.2 is approved", all(m in MODELS for m in EXPLORATORY_COMPARATORS))
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
print(f"  Software: XGBoost {xgb.__version__}, LightGBM {lgb.__version__}, scikit-survival {sksurv.__version__}, scikit-learn {__import__('sklearn').__version__} (deviation D-7).")
_ = A.close(outputs={"stage": "E", "locked_baselines": LOCKED_BASELINES, "active_models": ACTIVE_MODELS, "rsf_status": RSF_STATUS},
            extra_path=DIR["gates"] / "gate_E.json")
print("\n  GATE E PASSED. Stage F (cells 37 to 42, network and CKINN) may start.")
