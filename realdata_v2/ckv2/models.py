"""Every model behind one interface (protocol_v2.md, section 4). fit(train frame, labels, weights) -> predict(frame) -> probabilities.

Pairing (A1): the inner 85/15 split, the row order and the network seed depend only on `seed`, never on the model name, so models
fitted in the same fold with the same seed see identical data, and CKINN with lambda 0 equals M2.
"""
import math
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer

from .config import BINARY, MLP_DEFAULT, NN_MODELS
from .missing import augment_frame
from .nets import Net, predict_logit, train_net
from .prep import Prep

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def inner_split(y: np.ndarray, seed: int, val_frac: float):
    idx = np.arange(len(y))
    itr, iva = train_test_split(idx, test_size=val_frac, stratify=y, random_state=seed % (2 ** 32 - 1))
    return np.sort(itr), np.sort(iva)


def threshold_for_spec(y: np.ndarray, p: np.ndarray, spec: float) -> float:
    """Lowest threshold with specificity >= spec (v1 rule)."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return 0.5
    k = min(int(math.ceil(spec * len(neg))), len(neg)) - 1
    return float(np.nextafter(neg[k], 1.0))


def threshold_for_sens(y: np.ndarray, p: np.ndarray, sens: float) -> float:
    """Highest threshold with sensitivity >= sens: at least ceil(sens * n_pos) positives have p >= threshold."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = min(int(math.ceil(sens * len(pos))), len(pos)) - 1
    return float(pos[k])


class ConstrainedLR:
    """L2 logistic regression with sign bounds on chosen coefficients; same objective as scikit-learn:
    0.5 * ||beta||^2 + C * sum_i w_i * logloss_i (intercept unpenalised). L-BFGS-B."""

    def __init__(self, C: float, bounds_sign: Sequence[int]):
        self.C, self.bounds_sign = C, list(bounds_sign)

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        n, p = X.shape
        w = np.ones(n) if w is None else np.asarray(w, float)
        Xa = np.c_[np.ones(n), X]

        def f(theta):
            z = Xa @ theta
            ll = np.logaddexp(0, z) - y * z
            g = Xa.T @ (w * (expit(z) - y))
            obj = self.C * (w * ll).sum() + 0.5 * (theta[1:] ** 2).sum()
            grad = self.C * g
            grad[1:] += theta[1:]
            return obj, grad

        b = [(None, None)] + [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in self.bounds_sign]
        x0 = np.zeros(p + 1)
        x0[0] = logit(np.clip((w * y).sum() / w.sum(), 1e-4, 1 - 1e-4))
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": 2000, "gtol": 1e-8})
        self.intercept_, self.coef_ = res.x[0], res.x[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = expit(self.intercept_ + X @ self.coef_)
        return np.c_[1 - p, p]


class Fitted:
    """name in LR, cLR, GAM, RF, XGB, XGB_mono, LGBM, LGBM_mono, MI_LR, LR_aug, XGB_aug, M1, M2, CKINN."""

    def __init__(self, name: str, features: Sequence[str], hp: dict, seed: int, cfg, signs: Optional[Dict[str, int]] = None,
                 lam: float = 0.0):
        self.name, self.features, self.hp, self.seed, self.cfg = name, list(features), dict(hp or {}), int(seed), cfg
        self.signs = dict(signs or {})
        self.lam = float(lam)
        self.thr_spec, self.thr_sens, self.hist = 0.5, 0.5, {}

    # ------------------------------------------------------------------ fit
    def fit(self, dtr: pd.DataFrame, ytr: np.ndarray, sw: Optional[np.ndarray] = None) -> "Fitted":
        ytr = np.asarray(ytr).astype(int)
        itr, iva = inner_split(ytr, self.seed, self.cfg.val_frac)
        a, b = dtr.iloc[itr].reset_index(drop=True), dtr.iloc[iva].reset_index(drop=True)
        ya, yb = ytr[itr], ytr[iva]
        wa = None if sw is None else np.asarray(sw, float)[itr]
        wb = None if sw is None else np.asarray(sw, float)[iva]
        if self.name in NN_MODELS:
            self._fit_net(a, ya, wa, b, yb, wb)
        else:
            self._fit_other(a, ya, wa, b, yb)
        pb = self.predict(b)
        self.thr_spec = threshold_for_spec(yb, pb, self.cfg.spec_target)
        self.thr_sens = threshold_for_sens(yb, pb, self.cfg.sens_target)
        return self

    def _fit_net(self, a, ya, wa, b, yb, wb):
        cfg = self.cfg
        self.prep = Prep(self.features, cfg.winsor).fit(a)
        Xa, Ma = self.prep.transform(a)
        Xb, Mb = self.prep.transform(b)
        hp = {**MLP_DEFAULT, **self.hp}
        torch.manual_seed(self.seed)
        self.net = Net(self.features, tuple(hp["hidden"]), hp["enc_dim"], hp["dropout"],
                       self.signs if self.name == "CKINN" else None, use_mask=(self.name != "M1"))
        prev = float(np.average(ya, weights=wa)) if wa is not None else float(ya.mean())
        with torch.no_grad():
            self.net.head.bias.fill_(float(logit(np.clip(prev, 1e-4, 1 - 1e-4))))
        wa_ = np.ones(len(ya), np.float32) if wa is None else wa.astype(np.float32)
        wb_ = None if wb is None else wb
        self.hist = train_net(self.net, (Xa, Ma, ya, wa_), (Xb, Mb, yb, wb_), hp, cfg, self.seed,
                              lam=self.lam if self.name == "CKINN" else 0.0, aug=(self.name != "M1"))

    def _design_lr(self, df: pd.DataFrame) -> np.ndarray:
        return self.prep.transform(df)[0].astype(np.float64)

    def _design_gam(self, df: pd.DataFrame) -> np.ndarray:
        X, M = self.prep.transform(df)
        cols = []
        for j, f in enumerate(self.features):
            if f in BINARY:
                cols.append(X[:, [j]])
            else:
                cols.append(self.splines[j].transform(X[:, [j]].astype(np.float64)))
            if self.miss_ind[j]:
                cols.append(1.0 - M[:, [j]])
        return np.hstack(cols).astype(np.float64)

    def _fit_other(self, a, ya, wa, b, yb):
        name, hp, seed, cfg = self.name, self.hp, self.seed, self.cfg
        mono = tuple(int(self.signs.get(f, 0)) for f in self.features)
        if name in ("LR_aug", "XGB_aug"):
            a, ya = augment_frame(a, ya, self.features, seed, cfg.aug_prob, cfg.p_panel_drop, cfg.p_feat_drop)
            wa = None
        if name in ("LR", "LR_aug", "cLR", "GAM", "MI_LR"):
            self.prep = Prep(self.features, cfg.winsor).fit(a)
        if name in ("LR", "LR_aug"):
            self.model = LogisticRegression(C=hp["C"], max_iter=3000).fit(self._design_lr(a), ya, sample_weight=wa)
        elif name == "cLR":
            self.model = ConstrainedLR(hp["C"], mono).fit(self._design_lr(a), ya, wa)
        elif name == "GAM":
            X, M = self.prep.transform(a)
            self.splines, self.miss_ind = {}, []
            for j, f in enumerate(self.features):
                self.miss_ind.append(bool((M[:, j] == 0).any()))
                if f not in BINARY:
                    obs = M[:, j] == 1
                    self.splines[j] = SplineTransformer(n_knots=cfg.spline_knots, degree=3, extrapolation="constant",
                                                        knots="quantile").fit(X[obs, j:j + 1].astype(np.float64))
            self.model = LogisticRegression(C=hp["C"], max_iter=3000).fit(self._design_gam(a), ya, sample_weight=wa)
        elif name == "MI_LR":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            Xa, Ma = self.prep.transform(a)
            Za = np.where(Ma == 1, Xa, np.nan).astype(np.float64)
            keep = ~np.all(np.isnan(Za), axis=0)
            self.mi_keep = keep
            self.imputers, self.mi_models = [], []
            for m in range(cfg.mi_m):
                imp = IterativeImputer(sample_posterior=True, max_iter=cfg.mi_iter, random_state=(seed + 1009 * m) % (2 ** 32 - 1))
                Zi = imp.fit_transform(Za[:, keep])
                self.imputers.append(imp)
                self.mi_models.append(LogisticRegression(C=hp["C"], max_iter=3000).fit(Zi, ya, sample_weight=wa))
        elif name == "RF":
            R = a[self.features].to_numpy(float)
            self.med = np.nan_to_num(np.nanmedian(R, axis=0), nan=0.0)
            self.model = RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                                max_features="sqrt", n_jobs=1, random_state=seed % (2 ** 32 - 1)).fit(
                np.where(np.isnan(R), self.med, R), ya, sample_weight=wa)
        elif name in ("XGB", "XGB_mono", "XGB_aug"):
            import xgboost as xgb
            kw = {"monotone_constraints": "(" + ",".join(map(str, mono)) + ")"} if name == "XGB_mono" else {}
            self.model = xgb.XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
                                           subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=seed % (2 ** 31 - 1),
                                           eval_metric="logloss", verbosity=0, **kw)
            self.model.fit(a[self.features].to_numpy(float), ya, sample_weight=wa)
        elif name in ("LGBM", "LGBM_mono"):
            import lightgbm as lgb
            kw = {"monotone_constraints": list(mono)} if name == "LGBM_mono" else {}
            self.model = lgb.LGBMClassifier(n_estimators=hp["n_estimators"], num_leaves=hp["num_leaves"], learning_rate=hp["learning_rate"],
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=seed % (2 ** 31 - 1),
                                            verbose=-1, **kw)
            self.model.fit(a[self.features].to_numpy(float), ya, sample_weight=wa)
        else:
            raise ValueError(name)

    # -------------------------------------------------------------- predict
    def predict_arrays(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        return expit(predict_logit(self.net, X, M))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        name = self.name
        if name in NN_MODELS:
            X, M = self.prep.transform(df)
            return self.predict_arrays(X, M)
        if name in ("LR", "LR_aug", "cLR"):
            return self.model.predict_proba(self._design_lr(df))[:, 1]
        if name == "GAM":
            return self.model.predict_proba(self._design_gam(df))[:, 1]
        if name == "MI_LR":
            X, M = self.prep.transform(df)
            Z = np.where(M == 1, X, np.nan).astype(np.float64)[:, self.mi_keep]
            return np.mean([m.predict_proba(imp.transform(Z))[:, 1] for imp, m in zip(self.imputers, self.mi_models)], axis=0)
        R = df[self.features].to_numpy(float)
        if name == "RF":
            return self.model.predict_proba(np.where(np.isnan(R), self.med, R))[:, 1]
        return self.model.predict_proba(R)[:, 1]

    def info(self) -> dict:
        h = self.hist or {}
        return {"epochs": h.get("epochs"), "best_val": h.get("best"), "final_pen": (h.get("pen") or [None])[-1],
                "thr_spec": self.thr_spec, "thr_sens": self.thr_sens, "lam": self.lam}
