"""Shared code for the CKINN notebooks. Real data only. CPU-friendly.

Models expose .fit(df) and .lp(df) (log relative hazard). `fit_with_baseline` adds a Breslow
baseline so every model yields absolute risk at horizon h = 3 years.
"""
import os
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.util import Surv

warnings.filterwarnings("ignore")
THREADS = int(os.environ.get("CKINN_THREADS", "1"))

H = 3.0                                   # primary horizon, years
BASE = ['age', 'sex_male', 'bmi', 'sbp', 'dbp']
TIERS = {'T1': BASE,
         'T2': BASE + ['tc', 'tg', 'alt'],
         'T3': BASE + ['tc', 'tg', 'alt', 'fpg'],
         'T4': BASE + ['tc', 'tg', 'alt', 'fpg', 'hdl', 'ast']}
# Expected direction of risk (protocol section 7). Variables not listed are unconstrained.
PRIORS = {'age': +1, 'bmi': +1, 'sbp': +1, 'tg': +1, 'fpg': +1, 'hdl': -1,
          'famhx': +1, 'hba1c': +1, 'exercise': -1}
SMALL_SITES = (0, 16)                     # pooled, never a test fold (protocol section 8)


# ------------------------------------------------------------------ data
def load_data(data_dir):
    rich = pd.read_csv(f"{data_dir}/rich_harmonized.csv")
    nag = pd.read_csv(f"{data_dir}/nagala_harmonized.csv")
    for d in (rich, nag):
        d['age_band'] = pd.cut(d.age, [0, 40, 55, 200], right=False, labels=['<40', '40-54', '55+'])
    return rich, nag


class Prep:
    """Median fill, missingness indicators (for columns with NaN in training), z-scoring."""
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, df):
        X = df[self.cols]
        self.med = X.median()
        self.ind = [c for c in self.cols if X[c].isna().any()]
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        self.names = self.cols + [c + '_missing' for c in self.ind]
        return self

    def transform(self, df):
        X = df[self.cols]
        out = ((X.fillna(self.med) - self.mu) / self.sd).to_numpy(np.float32)
        if self.ind:
            out = np.hstack([out, X[self.ind].isna().to_numpy(np.float32)])
        return out


# ------------------------------------------------------------------ models
class CoxLinear:
    """Ridge-penalised Cox regression."""
    def __init__(self, cols, penalizer=0.01):
        self.cols, self.penalizer = list(cols), penalizer

    def fit(self, df):
        self.prep = Prep(self.cols).fit(df)
        d = pd.DataFrame(self.prep.transform(df), columns=self.prep.names)
        d['T'], d['E'] = df.time_y.values, df.event.values
        m = CoxPHFitter(penalizer=self.penalizer).fit(d, 'T', 'E')
        self.beta = m.params_.reindex(self.prep.names).to_numpy()
        return self

    def lp(self, df):
        return self.prep.transform(df) @ self.beta


class XGBCox:
    """Gradient-boosted Cox model; monotone=True applies the clinical priors as hard monotone constraints."""
    def __init__(self, cols, monotone=False, n_estimators=300, max_depth=3, eta=0.05, seed=0):
        self.cols, self.monotone = list(cols), monotone
        self.n, self.depth, self.eta, self.seed = n_estimators, max_depth, eta, seed

    def _X(self, df):
        return df[self.cols].to_numpy(np.float32)       # NaN handled natively

    def fit(self, df):
        import xgboost as xgb
        y = np.where(df.event.values == 1, df.time_y.values, -df.time_y.values)
        p = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik', 'eta': self.eta,
             'max_depth': self.depth, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 20, 'tree_method': 'hist', 'seed': self.seed, 'nthread': THREADS}
        if self.monotone:
            p['monotone_constraints'] = '(' + ','.join(str(PRIORS.get(c, 0)) for c in self.cols) + ')'
        self.bst = xgb.train(p, xgb.DMatrix(self._X(df), label=y), num_boost_round=self.n)
        return self

    def lp(self, df):
        import xgboost as xgb
        return self.bst.predict(xgb.DMatrix(self._X(df)), output_margin=True)


class DeepCox:
    """MLP with Cox partial-likelihood loss. lam > 0 adds the clinical-constraint penalty = CKINN.

    penalty = sum_j mean( relu(-s_j * d lp / d x_j)^2 ), derivatives w.r.t. standardised inputs.
    """
    def __init__(self, cols, hidden=(64, 64), lam=0.0, priors=None, epochs=20, batch=8192, lr=2e-3,
                 wd=1e-4, dropout=0.1, seed=0, patience=4, val_frac=0.1, pen_rows=2048, verbose=False):
        self.cols, self.hidden, self.lam = list(cols), hidden, lam
        self.priors = PRIORS if priors is None else priors
        self.epochs, self.batch, self.lr, self.wd, self.dropout = epochs, batch, lr, wd, dropout
        self.seed, self.patience, self.val_frac, self.pen_rows, self.verbose = seed, patience, val_frac, pen_rows, verbose

    @staticmethod
    def _nll(lp, t, e):
        o = torch.argsort(t, descending=True)
        lp, e = lp[o], e[o]
        return -((lp - torch.logcumsumexp(lp, 0)) * e).sum() / e.sum().clamp(min=1)

    def _penalty(self, x):
        x = x.clone().requires_grad_(True)
        g = torch.autograd.grad(self.net(x).squeeze(-1).sum(), x, create_graph=True)[0]
        return sum(torch.relu(-s * g[:, i]).pow(2).mean() for i, s in self.cons)

    def fit(self, df):
        global torch
        import torch as _t
        torch = _t
        torch.set_num_threads(THREADS)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.prep = Prep(self.cols).fit(df)
        X = torch.tensor(self.prep.transform(df))
        T = torch.tensor(df.time_y.values, dtype=torch.float32)
        E = torch.tensor(df.event.values, dtype=torch.float32)
        self.cons = [(self.prep.names.index(c), s) for c, s in self.priors.items() if c in self.prep.names]
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        perm = rng.permutation(len(X))
        nv = int(len(X) * self.val_frac)
        va, tr = torch.tensor(perm[:nv]), torch.tensor(perm[nv:])
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            order = tr[torch.randperm(len(tr))]
            for b in order.split(self.batch):
                if E[b].sum() < 1:
                    continue
                opt.zero_grad()
                loss = self._nll(self.net(X[b]).squeeze(-1), T[b], E[b])
                if self.lam > 0 and self.cons:
                    loss = loss + self.lam * self._penalty(X[b][:self.pen_rows])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = self._nll(self.net(X[va]).squeeze(-1), T[va], E[va]).item()
            if self.verbose:
                print(f"epoch {ep:02d} val nll {v:.4f}")
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def lp(self, df):
        with torch.no_grad():
            return self.net(torch.tensor(self.prep.transform(df))).squeeze(-1).numpy()


class CKINNAuto:
    """CKINN with the constraint strength chosen INSIDE the training data (protocol 6).

    Fit DeepCox on an 80% inner split for lambda in {0} + grid, score scaled Brier on the inner 20%.
    Rule: choose the largest positive lambda whose inner scaled Brier is within `tol` of the best
    (best over lambda = 0 and the grid); if none qualifies, the smallest positive lambda."""
    def __init__(self, cols, grid=(0.1, 1.0, 10.0), tol=0.002, epochs=40, seed=0, priors=None):
        self.cols, self.grid, self.tol, self.epochs, self.seed, self.priors = list(cols), grid, tol, epochs, seed, priors

    def fit(self, df):
        fit_part, val = next(cv_folds(df, 5, self.seed + 1000))
        sb = {}
        for lam in (0.0,) + tuple(self.grid):
            f = fit_with_baseline(DeepCox(self.cols, lam=lam, priors=self.priors, epochs=self.epochs, seed=self.seed), fit_part)
            sb[lam] = evaluate(f, val)['scaled_brier']
        best = max(sb.values())
        ok = [l for l in self.grid if sb[l] >= best - self.tol]
        self.lam_ = max(ok) if ok else min(self.grid)
        self.inner_scaled_brier_ = sb
        self.model = DeepCox(self.cols, lam=self.lam_, priors=self.priors, epochs=self.epochs, seed=self.seed).fit(df)
        return self

    def lp(self, df):
        return self.model.lp(df)


class ScaledLP:
    """Wraps a model and multiplies its lp by a slope (used for recalibration)."""
    def __init__(self, model, slope):
        self.model, self.slope, self.cols = model, slope, model.cols

    def lp(self, df):
        return self.slope * self.model.lp(df)


# ------------------------------------------------------------------ baseline and risk
class Breslow:
    def __init__(self, time, event, lp):
        o = np.argsort(time)
        t, e, r = np.asarray(time)[o], np.asarray(event)[o], np.exp(np.clip(np.asarray(lp)[o], -20, 20))
        S = np.cumsum(r[::-1])[::-1]
        ut, cnt = np.unique(t[e == 1], return_counts=True)
        self.t, self.H = ut, np.cumsum(cnt / S[np.searchsorted(t, ut, side='left')])

    def __call__(self, h):
        i = np.searchsorted(self.t, h, side='right') - 1
        return 0.0 if i < 0 else float(self.H[i])


class Fitted:
    def __init__(self, model, train):
        self.model, self.train = model, train
        self.H0 = Breslow(train.time_y.values, train.event.values, model.lp(train))

    def risk(self, df, h=H):
        lp = np.asarray(self.model.lp(df), dtype=np.float64)
        return 1 - np.exp(-self.H0(h) * np.exp(np.clip(lp, -20, 20)))


def fit_with_baseline(model, train):
    return Fitted(model.fit(train), train)


# ------------------------------------------------------------------ metrics
def km_risk(df, h=H):
    if len(df) < 20 or df.time_y.max() < h or df.event.sum() == 0:
        return np.nan
    return 1 - float(KaplanMeierFitter().fit(df.time_y, df.event).predict(h))


def cal_slope(risk, df):
    z = np.log(-np.log1p(-np.clip(np.asarray(risk, dtype=np.float64), 1e-9, 1 - 1e-9)))
    d = pd.DataFrame({'z': z, 'T': df.time_y.values, 'E': df.event.values})
    return float(CoxPHFitter().fit(d, 'T', 'E').params_['z'])


def violation_rate(model, df, delta=0.25, n=5000, seed=0):
    """Share of patients (per continuous constrained variable) where raising the variable by delta SD
    moves the model's log-hazard against the clinical prior. Finite differences, so it works for any model."""
    s = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    base, per = model.lp(s), {}
    for c in model.cols:
        if c in PRIORS and s[c].nunique() > 2:
            p = s.copy()
            p[c] = p[c] + delta * df[c].std()
            per[c] = float(np.mean(PRIORS[c] * (model.lp(p) - base) < -1e-6))
    return (float(np.mean(list(per.values()))) if per else np.nan), per


def metrics_from_arrays(train, test, lp, risk, h=H, viol=np.nan):
    """Primary metrics from precomputed log-hazard `lp` and absolute risk at h for `test`."""
    keep = (test.time_y < train.time_y.max()).to_numpy()
    test, lp, risk = test[keep], np.asarray(lp)[keep], np.asarray(risk, dtype=np.float64)[keep]
    ytr = Surv.from_arrays(train.event.astype(bool), train.time_y.values)
    yte = Surv.from_arrays(test.event.astype(bool), test.time_y.values)
    bs = brier_score(ytr, yte, (1 - risk)[:, None], [h])[1][0]
    s0 = float(KaplanMeierFitter().fit(train.time_y, train.event).predict(h))
    bs0 = brier_score(ytr, yte, np.full((len(test), 1), s0), [h])[1][0]
    obs = km_risk(test, h)
    slope = cal_slope(risk, test)
    return {'n': len(test), 'events': int(test.event.sum()),
            'cindex': float(concordance_index(test.time_y, -lp, test.event)),
            'auc_3y': float(cumulative_dynamic_auc(ytr, yte, lp, [h])[0][0]),
            'brier': float(bs), 'scaled_brier': float(1 - bs / bs0),
            'obs_risk': obs, 'mean_pred': float(risk.mean()), 'oe': float(obs / risk.mean()),
            'cal_slope': slope, 'cal_slope_dist': abs(slope - 1.0), 'viol_rate': viol}


def evaluate(fit, test, h=H):
    """All primary metrics at horizon h. `fit` is a Fitted; `test` a DataFrame."""
    test = test[test.time_y < fit.train.time_y.max()]
    return metrics_from_arrays(fit.train, test, fit.model.lp(test), fit.risk(test, h), h,
                               viol=violation_rate(fit.model, test)[0])


def boot_diff_arrays(train, test, arrs, ref, n_boot=200, seed=0, h=H, max_n=None,
                     metrics=('cindex', 'scaled_brier', 'cal_slope_dist')):
    """Paired bootstrap of metric(model) - metric(ref) on the same resampled patients.
    arrs: {name: (lp, risk)} aligned with `test`. max_n: optional fixed random subsample of test first."""
    rng = np.random.default_rng(seed)
    if max_n and len(test) > max_n:
        sel = rng.choice(len(test), max_n, replace=False)
        test, arrs = test.iloc[sel], {k: (v[0][sel], v[1][sel]) for k, v in arrs.items()}
    out = {k: {m: [] for m in metrics} for k in arrs if k != ref}
    for _ in range(n_boot):
        idx = rng.integers(0, len(test), len(test))
        b = test.iloc[idx]
        r_ref = metrics_from_arrays(train, b, arrs[ref][0][idx], arrs[ref][1][idx], h)
        for k in out:
            r_k = metrics_from_arrays(train, b, arrs[k][0][idx], arrs[k][1][idx], h)
            for m in metrics:
                out[k][m].append(r_k[m] - r_ref[m])
    rows = []
    for k, d in out.items():
        for m, v in d.items():
            rows.append({'model': k, 'ref': ref, 'metric': m, 'mean': float(np.mean(v)),
                         'lo95': float(np.percentile(v, 2.5)), 'hi95': float(np.percentile(v, 97.5)), 'n_boot': n_boot})
    return pd.DataFrame(rows)


def calibration_table_risk(risk, df, h=H, q=10):
    risk = np.asarray(risk, dtype=np.float64)
    g = pd.qcut(risk, q, duplicates='drop')
    rows = []
    for i, d in df.reset_index(drop=True).groupby(g, observed=True):
        rows.append({'group': str(i), 'n': len(d), 'pred': risk[d.index].mean(), 'obs': km_risk(d, h)})
    return pd.DataFrame(rows)


def calibration_table(fit, test, h=H, q=10):
    return calibration_table_risk(fit.risk(test, h), test, h, q)


def decision_curve_risk(risk, df, thresholds=np.arange(0.01, 0.101, 0.01), h=H):
    risk = np.asarray(risk, dtype=np.float64)
    prev = km_risk(df, h)
    out = []
    for t in thresholds:
        m = risk >= t
        p = m.mean()
        obs = km_risk(df[m], h) if m.sum() >= 20 else np.nan
        nb = p * obs - p * (1 - obs) * t / (1 - t) if not np.isnan(obs) else np.nan
        out.append({'threshold': float(t), 'net_benefit': nb,
                    'treat_all': prev - (1 - prev) * t / (1 - t), 'treat_none': 0.0})
    return pd.DataFrame(out)


def decision_curve(fit, test, thresholds=np.arange(0.01, 0.101, 0.01), h=H):
    return decision_curve_risk(fit.risk(test, h), test, thresholds, h)


def subgroup_metrics(fit, test, by, h=H):
    rows = []
    for g, d in test.groupby(by, observed=True):
        if d.event.sum() >= 20 and d.time_y.max() >= h:
            r = evaluate(fit, d, h)
            rows.append({by: g, **{k: r[k] for k in ['n', 'events', 'cindex', 'auc_3y', 'scaled_brier', 'oe', 'cal_slope']}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ splits and transport
def cv_folds(df, k=5, seed=0):
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(df, df.event):
        yield df.iloc[tr], df.iloc[te]


def loso_folds(rich):
    for s in sorted(rich.site.unique()):
        if s in SMALL_SITES:
            continue
        yield s, rich[rich.site != s], rich[rich.site == s]


def shift_split(df, var, cut):
    return df[df[var] < cut], df[df[var] >= cut]


def recalibrate(fit, target_sample):
    """Intercept-and-slope recalibration on a small target sample (new slope, new Breslow baseline)."""
    lp = fit.model.lp(target_sample)
    d = pd.DataFrame({'lp': lp, 'T': target_sample.time_y.values, 'E': target_sample.event.values})
    slope = float(CoxPHFitter().fit(d, 'T', 'E').params_['lp'])
    return Fitted(ScaledLP(fit.model, slope), target_sample), slope


def run_models(specs, train, test, h=H, recal_n=None, seed=0):
    """specs: {name: callable returning an unfitted model}. Returns metrics DataFrame (plus recalibrated rows)."""
    rows, fits = [], {}
    for name, make in specs.items():
        f = fit_with_baseline(make(), train)
        fits[name] = f
        lam = getattr(f.model, 'lam_', np.nan)
        rows.append({'model': name, 'lam': lam, **evaluate(f, test, h)})
        if recal_n:
            tgt = test.sample(recal_n, random_state=seed)
            rest = test.drop(tgt.index)
            rf, _ = recalibrate(f, tgt)
            rows.append({'model': name + ' +recal', 'lam': lam, **evaluate(rf, rest, h)})
    return pd.DataFrame(rows), fits


def boot_diff(fit_a, fit_b, test, h=H, n_boot=200, seed=0, metrics=('cindex', 'scaled_brier')):
    """Paired bootstrap of metric(a) - metric(b) on the same resampled test patients."""
    rng = np.random.default_rng(seed)
    out = {m: [] for m in metrics}
    for _ in range(n_boot):
        b = test.iloc[rng.integers(0, len(test), len(test))]
        ra, rb = evaluate(fit_a, b, h), evaluate(fit_b, b, h)
        for m in metrics:
            out[m].append(ra[m] - rb[m])
    return pd.DataFrame({m: {'mean': np.mean(v), 'lo95': np.percentile(v, 2.5), 'hi95': np.percentile(v, 97.5)}
                         for m, v in out.items()}).T


def nadeau_bengio(diffs, n_train, n_test, alpha=0.05):
    """Corrected resampled t interval for the mean of paired fold differences (Nadeau & Bengio 2003)."""
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    var = (1.0 / J + n_test / n_train) * d.var(ddof=1)
    se = np.sqrt(var)
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    m = d.mean()
    return {'mean': m, 'lo95': m - t * se, 'hi95': m + t * se, 'J': J}
