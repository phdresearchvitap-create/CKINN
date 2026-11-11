"""Unit tests on hand-made toy arrays only. They produce no results and use no study data."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ckinn import lib as L


def toy(n=400, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    t = r.exponential(8 / np.exp(0.8 * x))
    c = r.uniform(1, 6, n)
    return pd.DataFrame({'age': x, 'bmi': r.normal(size=n), 'time_y': np.minimum(t, c), 'event': (t <= c).astype(int),
                         'sex_male': r.integers(0, 2, n), 'sbp': r.normal(size=n), 'dbp': r.normal(size=n), 'site': r.integers(1, 5, n)})


def test_breslow_monotone():
    d = toy()
    b = L.Breslow(d.time_y.values, d.event.values, np.zeros(len(d)))
    assert b(1) <= b(3) <= b(6)


def test_cox_risk_bounds_and_order():
    d = toy()
    f = L.fit_with_baseline(L.CoxLinear(['age', 'bmi']), d)
    r = f.risk(d)
    assert ((r >= 0) & (r <= 1)).all()
    assert np.corrcoef(r, d.age)[0, 1] > 0.5


def test_violation_rate_zero_for_monotone_model():
    d = toy()
    class M:
        cols = ['age', 'bmi']
        def lp(self, df): return 1.0 * df.age.values + 0.5 * df.bmi.values
    v, per = L.violation_rate(M(), d)
    assert v == 0.0
    class N(M):
        def lp(self, df): return -df.age.values
    assert L.violation_rate(N(), d)[1]['age'] == 1.0


def test_cv_folds_disjoint():
    d = toy()
    for tr, te in L.cv_folds(d, 5, 0):
        assert not set(tr.index) & set(te.index)


def test_nadeau_bengio_wider_than_naive():
    d = np.array([0.01, 0.02, 0.015, 0.0, 0.02, 0.01, 0.012, 0.018, 0.005, 0.02, 0.011, 0.014, 0.016, 0.009, 0.013])
    out = L.nadeau_bengio(d, 1000, 250)
    naive = 2 * 2.145 * d.std(ddof=1) / np.sqrt(len(d))
    assert out['hi95'] - out['lo95'] > naive


def test_loso_skips_small_sites():
    d = toy(); d.loc[:5, 'site'] = 0
    assert 0 not in [s for s, _, _ in L.loso_folds(d)]
