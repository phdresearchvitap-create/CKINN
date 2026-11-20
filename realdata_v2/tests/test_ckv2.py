"""Unit tests for the v2 library. Toy arrays only (CLAUDE.md); no cohort data are read here.
Run: python -m pytest -q realdata_v2/tests  (the notebook runs the same checks in its test cells)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ckv2 import metrics as mt  # noqa: E402
from ckv2.config import make_config, SIGNS_T2  # noqa: E402
from ckv2.models import ConstrainedLR, Fitted, threshold_for_sens, threshold_for_spec  # noqa: E402
from ckv2.nets import Net, clinical_penalty  # noqa: E402
from ckv2.data import repeated_folds  # noqa: E402

CFG = make_config("smoke")


def toy_binary(n=4000, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.6, n)
    if ties:
        p = np.round(p, 2)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def toy_frame(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"age_years": rng.uniform(20, 80, n), "sex_female": rng.integers(0, 2, n).astype(float),
                      "bmi_kgm2": rng.normal(24, 3, n), "sbp_mmHg": rng.normal(125, 15, n), "tg_mmol": np.exp(rng.normal(0.2, 0.5, n)),
                      "hdl_mmol": rng.normal(1.4, 0.3, n)})
    z = -5 + 0.05 * (d.age_years - 50) + 0.15 * (d.bmi_kgm2 - 24) + 0.8 * np.log(d.tg_mmol) - 1.0 * (d.hdl_mmol - 1.4)
    d["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-z - 2))).astype(int)
    d.loc[rng.random(n) < 0.2, "hdl_mmol"] = np.nan
    return d


# ------------------------------------------------------------------ metrics
def test_auroc_matches_sklearn_with_ties_and_weights():
    from sklearn.metrics import roc_auc_score
    y, p = toy_binary(ties=True)
    r = mt.Ranked(p, y)
    assert abs(mt.auroc_w(r) - roc_auc_score(y, p)) < 1e-12
    w = np.random.default_rng(2).integers(0, 3, len(y)).astype(float)
    assert abs(mt.auroc_w(r, w) - roc_auc_score(y, p, sample_weight=w)) < 1e-12


def test_bootstrap_weights_equal_explicit_resample():
    from sklearn.metrics import roc_auc_score
    y, p = toy_binary(2000, 3)
    w = next(mt.boot_weight_iter(len(y), 1, 7))
    idx = np.repeat(np.arange(len(y)), w.astype(int))
    assert abs(mt.auroc_w(mt.Ranked(p, y), w) - roc_auc_score(y[idx], p[idx])) < 1e-12
    assert abs(mt.brier_w(y, p, w) - np.mean((p[idx] - y[idx]) ** 2)) < 1e-12


def test_calibration_known_answers():
    y, p = toy_binary(20000, 4)
    a, b = mt.calib_w(y, p)
    assert abs(b - 1) < 0.08 and abs(mt.citl_w(y, p)) < 0.08
    assert mt.citl_w(y, np.clip(p * 1.8, 0, 0.99)) < -0.3
    from sklearn.linear_model import LogisticRegression
    z = mt.logit(mt.clip(p))[:, None]
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(z, y)
    assert abs(lr.coef_[0, 0] - b) < 1e-4 and abs(lr.intercept_[0] - a) < 1e-4
    assert mt.ece_w(y, p) < 0.02 and mt.ece_w(y, np.clip(p * 1.5, 0, 1)) > 0.05


def test_tpr_at_fpr():
    from sklearn.metrics import roc_curve
    y, p = toy_binary(5000, 5)
    f, t, _ = roc_curve(y, p)
    assert abs(mt.tpr_at_fpr_w(mt.Ranked(p, y), 0.1) - np.interp(0.1, f, t)) < 2e-3
    yy = np.r_[np.zeros(50), np.ones(50)]
    pp = np.r_[np.linspace(0, .4, 50), np.linspace(.6, 1, 50)]
    assert mt.tpr_at_fpr_w(mt.Ranked(pp, yy), 0.1) == 1.0


def test_delong_against_bootstrap_and_identity():
    y, p = toy_binary(3000, 6)
    rng = np.random.default_rng(1)
    q = np.clip(p + rng.normal(0, 0.1, len(p)), 0, 1)
    d = mt.delong_paired(y, p, q)
    from sklearn.metrics import roc_auc_score
    assert abs(d["auc1"] - roc_auc_score(y, p)) < 1e-12 and abs(d["auc2"] - roc_auc_score(y, q)) < 1e-12
    rp, rq = mt.Ranked(p, y), mt.Ranked(q, y)
    bs = [mt.auroc_w(rp, w) - mt.auroc_w(rq, w) for w in mt.boot_weight_iter(len(y), 400, 3)]
    assert 0.7 < d["se"] / np.std(bs) < 1.3
    assert mt.delong_paired(y, p, p)["diff"] == 0.0


def test_net_benefit():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    # t = 0.5: TP 2, FP 1 -> (2 - 1 * 1) / 10
    assert abs(mt.net_benefit(y, p, 0.5) - 0.1) < 1e-12
    assert abs(mt.net_benefit_all(y, 0.5) - (0.3 - 0.7)) < 1e-12


def test_censoring_metrics_against_reference():
    rng = np.random.default_rng(8)
    n = 600
    risk = rng.normal(size=n)
    tt = rng.exponential(1 / np.exp(0.7 * risk))
    cc = rng.exponential(1.5, n)
    time, event = np.minimum(tt, cc), (tt <= cc).astype(int)
    lifelines = pytest.importorskip("lifelines")
    from lifelines.utils import concordance_index
    assert abs(mt.concordance(risk, time, event) - concordance_index(time, -risk, event)) < 1e-10
    sksurv = pytest.importorskip("sksurv")
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score
    from sksurv.util import Surv
    s = Surv.from_arrays(event.astype(bool), time)
    tau = float(np.quantile(time, 0.6))
    ut, G = mt.km_censoring(time, event)
    auc_ref = cumulative_dynamic_auc(s, s, risk, [tau])[0][0]
    assert abs(mt.td_auc(risk, time, event, tau, ut, G) - auc_ref) < 1e-6
    w = np.where(event == 1, 1 / np.maximum(mt.G_at(ut, G, time), 1e-12) ** 2, 0)
    c_uno = mt.concordance(risk, time, event, event_weight=w, tau=tau)
    assert abs(c_uno - concordance_index_ipcw(s, s, risk, tau=tau)[0]) < 5e-3
    prob = 1 / (1 + np.exp(-risk))
    bs_ref = brier_score(s, s, 1 - prob[:, None], [tau])[1][0]
    assert abs(mt.ipcw_brier(prob, time, event, tau, ut, G) - bs_ref) < 5e-3


def test_holm_tost_corrected_t():
    assert abs(mt.holm([0.01, 0.04, 0.03])[0] - 0.03) < 1e-12 and mt.holm([0.01, 0.04, 0.03])[1] == 0.06
    d = np.r_[np.full(10, 0.0), np.full(10, 1.0)]
    assert mt.p_one_sided(d, 0.5, +1) == (1 + 10) / 21
    r = mt.corrected_t(np.array([0.01, 0.02, 0.0, 0.015, 0.012]), 100, 400)
    assert r["lo"] < r["mean"] < r["hi"] and r["df"] == 4


def test_cluster_bootstrap_weights():
    cl = np.repeat(np.arange(5), [3, 4, 5, 6, 7])
    w = next(mt.cluster_weight_iter(cl, 1, 0))
    for c in range(5):
        assert len(set(w[cl == c])) == 1
    assert w.sum() > 0


def test_spearman_rows():
    A = np.array([[1, 2, 3, 4.0], [4, 3, 2, 1.0]])
    B = np.array([[10, 20, 30, 40.0], [1, 2, 3, 4.0]])
    assert np.allclose(mt.spearman_rows(A, B), [1.0, -1.0])


# ------------------------------------------------------------------ thresholds, folds
def test_thresholds():
    y, p = toy_binary(20000, 9)
    t = threshold_for_spec(y, p, 0.9)
    assert abs((p[y == 0] < t).mean() - 0.9) < 0.01
    t2 = threshold_for_sens(y, p, 0.8)
    assert (p[y == 1] >= t2).mean() >= 0.8 and (p[y == 1] >= t2).mean() < 0.81


def test_repeated_folds_partition_and_stratify():
    y = (np.random.default_rng(0).random(1000) < 0.1).astype(int)
    rf = repeated_folds(y, "toy", 5, 3)
    for rep in rf:
        test = np.concatenate([j for _, j in rep])
        assert np.array_equal(np.sort(test), np.arange(1000))
        for i, j in rep:
            assert len(np.intersect1d(i, j)) == 0 and abs(y[j].mean() - y.mean()) < 0.02
    assert not np.array_equal(rf[0][0][1], rf[1][0][1])


# ------------------------------------------------------------------ constrained models
def test_constrained_lr_matches_sklearn_when_unconstrained_and_respects_bounds():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(X @ [1.0, -0.5, 0.3]) + 1))).astype(int)
    a = ConstrainedLR(1.0, [0, 0, 0]).fit(X, y)
    b = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    assert np.allclose(a.coef_, b.coef_[0], atol=1e-4) and abs(a.intercept_ - b.intercept_[0]) < 1e-4
    c = ConstrainedLR(1.0, [-1, 1, 0]).fit(X, y)            # both signs contradict the data -> held at 0
    assert c.coef_[0] <= 1e-12 and c.coef_[1] >= -1e-12


def _is_monotone(model, df, feat, sign):
    grid = np.quantile(df[feat].dropna(), np.linspace(0.05, 0.95, 9))
    base = df.iloc[:200].copy()
    prev = None
    for v in grid:
        base[feat] = v
        cur = model.predict(base)
        if prev is not None and np.any(sign * (cur - prev) < -1e-9):
            return False
        prev = cur
    return True


@pytest.mark.parametrize("name", ["XGB_mono", "LGBM_mono", "cLR"])
def test_monotone_baselines_are_monotone(name):
    d = toy_frame()
    feats = ["age_years", "sex_female", "bmi_kgm2", "sbp_mmHg", "tg_mmol", "hdl_mmol"]
    hp = {"C": 1.0, "max_depth": 3, "learning_rate": 0.1, "n_estimators": 60, "num_leaves": 7}
    m = Fitted(name, feats, hp, 3, CFG, signs=SIGNS_T2).fit(d, d.y.to_numpy())
    for f in ("age_years", "bmi_kgm2", "sbp_mmHg", "tg_mmol", "hdl_mmol"):
        assert _is_monotone(m, d, f, SIGNS_T2[f]), (name, f)


@pytest.mark.parametrize("name", ["LR", "GAM", "RF", "XGB", "LGBM", "MI_LR", "LR_aug", "XGB_aug", "M1", "M2"])
def test_every_model_returns_valid_probabilities(name):
    d = toy_frame()
    feats = ["age_years", "sex_female", "bmi_kgm2", "sbp_mmHg", "tg_mmol", "hdl_mmol"]
    hp = {"C": 1.0, "max_depth": 4, "min_samples_leaf": 20, "learning_rate": 0.1, "n_estimators": 60, "num_leaves": 7,
          "hidden": (32, 16), "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "enc_dim": 8}
    m = Fitted(name, feats, hp, 3, CFG).fit(d.iloc[:2400], d.y.to_numpy()[:2400])
    p = m.predict(d.iloc[2400:])
    assert p.shape == (600,) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1


# ------------------------------------------------------------------ CKINN
def test_penalty_analytic_and_zero_for_monotone():
    class Toy(torch.nn.Module):
        def __init__(self, coef):
            super().__init__()
            self.coef = torch.nn.Parameter(torch.tensor(coef, dtype=torch.float32))

        def forward(self, X, M):
            return (X * M) @ self.coef
    g = torch.Generator().manual_seed(1)
    X, M = torch.randn(200, 4, generator=g), torch.ones(200, 4)
    M[:50, 0] = 0.0
    X[:, 3] = (torch.rand(200, generator=g) < 0.5).float()
    signs, isb = torch.tensor([1.0, -1.0, 0.0, 1.0]), torch.tensor([False, False, False, True])
    pen, rate = clinical_penalty(Toy([0.8, -0.3, -5.0, 0.4]), X, M, signs, isb)
    assert pen.item() == 0.0 and rate == 0.0
    pen, rate = clinical_penalty(Toy([-0.5, 0.5, 2.0, -0.7]), X, M, signs, isb)
    assert abs(pen.item() - (150 * 0.25 + 200 * 0.25 + 200 * 0.49) / 550) < 1e-5 and rate == 1.0


def test_ckinn_lambda0_equals_m2_and_penalty_changes_fit():
    d = toy_frame(2000)
    feats = ["age_years", "sex_female", "bmi_kgm2", "sbp_mmHg", "tg_mmol", "hdl_mmol"]
    hp = {"hidden": (32, 16), "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "enc_dim": 8}
    y = d.y.to_numpy()
    m2 = Fitted("M2", feats, hp, 5, CFG).fit(d, y)
    c0 = Fitted("CKINN", feats, hp, 5, CFG, signs=SIGNS_T2, lam=0.0).fit(d, y)
    c1 = Fitted("CKINN", feats, hp, 5, CFG, signs=SIGNS_T2, lam=10.0).fit(d, y)
    assert np.array_equal(m2.predict(d), c0.predict(d))
    assert not np.array_equal(m2.predict(d), c1.predict(d)) and max(c1.hist["pen"]) > 0
