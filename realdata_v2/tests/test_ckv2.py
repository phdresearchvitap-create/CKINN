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


@pytest.mark.parametrize("name", ["LR", "GAM", "RF", "XGB", "LGBM", "MI_LR", "LR_aug", "XGB_aug", "M1", "M2", "SVM"])
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


def test_classify_interval_both_signs():
    from ckv2.analysis import classify_interval
    assert classify_interval(1, 0.1, 0.3) == "supported" and classify_interval(1, -0.3, -0.1) == "contradicted"
    assert classify_interval(-1, -0.3, -0.1) == "supported" and classify_interval(-1, 0.1, 0.3) == "contradicted"
    assert classify_interval(-1, -0.07, 0.03) == "uncertain" and classify_interval(1, -0.07, 0.03) == "uncertain"


def test_select_lambda_one_se_rule():
    from ckv2.plan import select_lambda
    rows = []
    for lam, base in ((0.0, 0.100), (0.1, 0.1001), (1.0, 0.1004), (10.0, 0.120)):
        rows += [{"lam": lam, "score": base + e} for e in (-0.001, 0.0, 0.001)]
    lam, tab = select_lambda(pd.DataFrame(rows), (0.1, 1.0, 10.0))
    assert lam == 1.0 and tab.chosen.sum() == 1          # 1.0 is within one SE of the best (lambda 0); 10 is not
    rows = [{"lam": l, "score": 0.1 + (0.05 if l > 0 else 0) + e} for l in (0.0, 0.1, 1.0) for e in (-1e-4, 0, 1e-4)]
    assert select_lambda(pd.DataFrame(rows), (0.1, 1.0))[0] == 0.1   # none qualifies: the smallest grid value


def test_verdict_rules():
    from ckv2.analysis import verdicts
    def row(h, unit, p_pred, p_opp):
        return {"family": "A", "hypothesis": h, "unit": unit, "endpoint": "e", "primary": True, "p_pred": p_pred, "p_opp": p_opp}
    ep = pd.DataFrame([row("H1", "u1", 1e-4, 1.0), row("H1", "u2", 1e-4, 1.0), row("H1", "u3", 0.5, 0.5),
                       row("H2", "u1", 1e-4, 1.0), row("H2", "u2", 1e-4, 1.0), row("H2", "u3", 0.5, 0.5),
                       row("H3", "u1", 1.0, 1e-4), row("H3", "u2", 1.0, 1e-4)])
    _, v = verdicts(ep, 0.05)
    v = v.set_index("hypothesis").verdict
    assert v["H1"] == "NULL" and v["H2"] == "PASS" and v["H3"] == "FAIL"


def test_groups_for_minimum_events():
    from ckv2.analysis import groups_for
    n = 600
    fr = pd.DataFrame({"sex_female": np.r_[np.ones(300), np.zeros(300)],
                       "age_band": np.tile(["20-39", "40-59", "60+"], 200)})
    y = np.zeros(n)
    y[:25] = 1                                            # 25 events in women, 5 in men
    y[300:305] = 1
    g, nb = groups_for(fr, y, 20)
    assert g["sex"] == [] and nb == 0
    y[300:330] = 1
    y[np.arange(0, 600, 3)[:30]] = 1
    g, nb = groups_for(fr, y, 20)
    assert len(g["sex"]) == 2 and nb >= 1


def test_survey_replicate_weights():
    st = np.repeat([1, 2], 40)
    psu = np.tile(np.repeat([1, 2], 20), 2)
    base = np.ones(80) * 2.0
    reps = list(mt.strata_psu_weight_iter(st, psu, base, 200, 3))
    assert all((r >= 0).all() for r in reps)
    assert abs(np.mean([r.sum() for r in reps]) / base.sum() - 1) < 0.05
    r0 = reps[0]
    for s in (1, 2):
        for p in (1, 2):
            assert len(set(r0[(st == s) & (psu == p)])) == 1


# ------------------------------------------------------------------ post hoc additions (PH)
def _linear_predict(beta, feats):
    def predict(df):
        z = sum(b * np.nan_to_num(df[f].to_numpy(float)) for b, f in zip(beta, feats))
        return 1 / (1 + np.exp(-z))
    return predict


def test_permutation_shap_is_exact_for_a_linear_logit_and_additive():
    from ckv2.posthoc import permutation_shap
    rng = np.random.default_rng(0)
    feats = ["a", "b", "c"]
    beta = [0.8, -0.5, 0.0]
    rows = pd.DataFrame(rng.normal(size=(5, 3)), columns=feats)
    bg = pd.DataFrame(rng.normal(size=(40, 3)), columns=feats)
    phi, base, fx = permutation_shap(_linear_predict(beta, feats), rows, bg, feats, 4, np.random.default_rng(1))
    expected = (rows.to_numpy() - bg.to_numpy().mean(0)) * np.array(beta)
    assert np.allclose(phi, expected, atol=1e-8)
    assert np.allclose(phi.sum(1), fx - base, atol=1e-8)


def test_perturbation_violations_monotone_and_anti_monotone():
    from ckv2.posthoc import perturbation_violations
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"bmi_kgm2": rng.normal(25, 3, 300), "hdl_mmol": rng.normal(1.3, 0.3, 300), "exercise_regular": rng.integers(0, 2, 300).astype(float)})
    df.loc[:9, "bmi_kgm2"] = np.nan
    signs = {"bmi_kgm2": 1, "hdl_mmol": -1, "exercise_regular": -1}
    good = perturbation_violations(_linear_predict([0.2, -1.0, -0.5], list(df)), df, df, signs, list(df), 0.25, 1e-6)
    bad = perturbation_violations(_linear_predict([-0.2, 1.0, 0.5], list(df)), df, df, signs, list(df), 0.25, 1e-6)
    assert all(v["viol"] == 0 for v in good.values())
    assert bad["bmi_kgm2"]["viol"] == bad["bmi_kgm2"]["n"] == 290
    assert bad["exercise_regular"]["n"] == int((df.exercise_regular == 0).sum()) == bad["exercise_regular"]["viol"]


def test_reasoning_text_wording_and_flags():
    from ckv2.posthoc import reasoning_inputs, reasoning_text
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"bmi_kgm2": rng.normal(25, 3, 50), "tg_mmol": np.exp(rng.normal(0.2, 0.4, 50)), "hdl_mmol": rng.normal(1.3, 0.3, 50)})
    feats = list(df)
    for beta, consistent in (([0.2, 0.8, -1.0], True), ([0.2, -0.8, -1.0], False)):
        pred = _linear_predict(beta, feats)
        rec = reasoning_inputs(pred, df.iloc[:5], df, {"bmi_kgm2": 1, "tg_mmol": 1, "hdl_mmol": -1}, feats, 0.25, 1e-6, ["tg_mmol", "hdl_mmol"])
        rec["phi"] = np.tile(np.array(beta), (5, 1))
        txt = reasoning_text(rec, 0, "B")
        low = txt.lower()
        for word in ("should", "lose", "prevent", "you will", "advise"):
            assert word not in low, word
        assert ("agrees with every clinical prior" in txt) == consistent
        assert "not a diagnosis" in txt


def test_weights_and_wilson_and_progression_label():
    from ckv2.posthoc import balanced_weights, group_weights, progression_frame, wilson
    y = np.r_[np.ones(10), np.zeros(90)]
    w = balanced_weights(y)
    assert abs(w.mean() - 1) < 1e-12 and abs(w[y == 1].sum() - w[y == 0].sum()) < 1e-9
    fr = pd.DataFrame({"sex": ["f"] * 30 + ["m"] * 70, "age_band": ["20-39"] * 50 + ["60+"] * 50})
    g = group_weights(fr)
    assert abs(g.mean() - 1) < 1e-12
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi
    ev = pd.DataFrame({"fpg_mmol": [5.0, 5.5, 5.7, 5.2], "fpg_final": [5.7, 5.0, 7.5, 5.0], "label": [0.0, 1.0, 1.0, 0.0]})
    pf = progression_frame(ev)
    assert len(pf) == 3 and pf.y.tolist() == [1, 1, 0]
