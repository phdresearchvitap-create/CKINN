"""Build every table, figure and number macro used by the manuscript from results/parts. No numbers are typed by hand."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ckinn import lib as L

P, PRED, TAB, FIG = ROOT / 'results/parts', ROOT / 'results/preds', ROOT / 'paper/tables', ROOT / 'paper/figs'
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
MODELS = ['Cox', 'XGB', 'XGB-mono', 'MLP', 'CKINN']
BASELINES = ['Cox', 'XGB', 'XGB-mono', 'MLP']
MET = ['cindex', 'scaled_brier', 'cal_slope', 'viol_rate']
NUM = {}


def load(pattern):
    fs = sorted(glob.glob(str(P / pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def f3(x): return '--' if pd.isna(x) else f'{x:.3f}'


def ci(m, lo, hi): return f'{m:.3f} ({lo:.3f}, {hi:.3f})'


W = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four'}


def macro(name, val):
    NUM[''.join(W.get(ch, ch) for ch in name)] = val


def write_tab(name, df, caption, label):
    body = df.to_latex(index=False, escape=True, na_rep='--')
    (TAB / f'{name}.tex').write_text(
        '\\begin{table}[t]\\centering\\small\n\\caption{' + caption + '}\\label{' + label + '}\n' + body + '\\end{table}\n')


def prim(d):
    return d[d.get('subgroup', pd.Series(index=d.index, dtype=object)).isna() & ~d.model.str.contains('recal')]


# ---------------------------------------------------------------- R1 CV
cv = load('cv_T*_r*.csv')
if len(cv):
    cv = prim(cv)
    rows = []
    for tier, g in cv.groupby('tier'):
        for m in MODELS:
            d = g[g.model == m]
            rows.append({'Tier': tier, 'Model': m, **{k: f'{d[k].mean():.3f} ({d[k].std():.3f})' for k in MET}})
    write_tab('cv_summary', pd.DataFrame(rows), 'Repeated 5-fold cross-validation in RICH, mean (SD) over folds. Violation rate is the share of patients whose log-hazard moves against a clinical prior.', 'tab:cv')
    for tier in sorted(cv.tier.unique()):
        g = cv[cv.tier == tier]
        piv = g.pivot_table(index=['repeat', 'fold'], columns='model', values=['cindex', 'viol_rate'])
        ntr, nte = g.n_train.mean(), g.n_test.mean()
        best = piv['cindex'][BASELINES].mean().idxmax()
        nb = L.nadeau_bengio((piv['cindex']['CKINN'] - piv['cindex'][best]).values, ntr, nte)
        macro(f'cvBest{tier}', best); macro(f'cvCdiff{tier}', ci(nb['mean'], nb['lo95'], nb['hi95']))
        nbv = L.nadeau_bengio((piv['viol_rate']['CKINN'] - piv['viol_rate']['MLP']).values, ntr, nte)
        macro(f'cvViolDiff{tier}', ci(nbv['mean'], nbv['lo95'], nbv['hi95']))
        if tier == 'T2':
            macro('C1supported', 'supported' if nb['lo95'] > -0.01 else 'not supported')
            macro('C4supported', 'supported' if nbv['hi95'] < 0 else 'not supported')
        macro(f'lamMedian{tier}', f"{g[g.model == 'CKINN'].lam.median():g}")

# ---------------------------------------------------------------- R2 LOSO
lo = load('loso_T*_s*.csv')
if len(lo):
    lp = prim(lo[lo.subgroup.isna()]) if 'subgroup' in lo else prim(lo)
    rows = []
    for tier, g in lp.groupby('tier'):
        for m in MODELS:
            d = g[g.model == m]
            rows.append({'Tier': tier, 'Model': m, 'Sites': len(d),
                         **{k: f'{d[k].median():.3f} [{d[k].quantile(.25):.3f}, {d[k].quantile(.75):.3f}]' for k in MET[:3]},
                         'Viol.': f3(d.viol_rate.median())})
    write_tab('loso_summary', pd.DataFrame(rows), 'Leave-one-site-out in RICH: median [IQR] over test sites.', 'tab:loso')
    t2 = lp[lp.tier == 'T2']
    if len(t2):
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        for a, k, t in zip(ax, ['cindex', 'cal_slope'], ['C-index', 'Calibration slope']):
            a.boxplot([t2[t2.model == m][k] for m in MODELS], tick_labels=MODELS); a.set_title(t); a.tick_params(axis='x', rotation=45)
        if True: ax[1].axhline(1, ls=':', c='k')
        fig.tight_layout(); fig.savefig(FIG / 'loso_T2.pdf'); plt.close(fig)

# ---------------------------------------------------------------- R4 shift
sh = load('shift_*.csv')
if len(sh):
    s = sh[sh.subgroup.isna()] if 'subgroup' in sh else sh
    rows = [{'Split': r.split, 'Tier': r.tier, 'Model': r.model, 'C': f3(r.cindex), 'sBrier': f3(r.scaled_brier),
             'Slope': f3(r.cal_slope), 'O/E': f3(r.oe), 'Viol.': f3(r.viol_rate)} for r in s.itertuples()]
    write_tab('shift_summary', pd.DataFrame(rows), 'Covariate-shift splits inside RICH; "+recal" rows recalibrate on 1,000 target patients.', 'tab:shift')

# ---------------------------------------------------------------- R3 cross-cohort
cr = load('cross_*_T?.csv')
if len(cr):
    c = cr[cr.subgroup.isna()] if 'subgroup' in cr else cr
    c = c[~c.nosex.astype(bool)]
    rows = [{'Train': r.train, 'Test': r.test, 'Tier': r.tier, 'Model': r.model, 'C': f3(r.cindex), 'sBrier': f3(r.scaled_brier),
             'Slope': f3(r.cal_slope), 'O/E': f3(r.oe), 'Viol.': f3(r.viol_rate)} for r in c.itertuples()]
    write_tab('cross_summary', pd.DataFrame(rows), 'Cross-cohort transport. "+recal" rows recalibrate on 1,000 target patients and are scored on the rest.', 'tab:cross')
    ns = cr[cr.nosex.astype(bool) & ~cr.model.str.contains('recal')]
    if len(ns):
        write_tab('cross_nosex', pd.DataFrame([{'Train': r.train, 'Test': r.test, 'Tier': r.tier, 'Model': r.model, 'C': f3(r.cindex), 'Slope': f3(r.cal_slope)} for r in ns.itertuples()]),
                  'Cross-cohort transport without sex (sensitivity S3).', 'tab:nosex')
    sg = cr[cr.subgroup.notna()]
    if len(sg):
        write_tab('cross_subgroups', pd.DataFrame([{'Train': r.train, 'Test': r.test, 'Model': r.model, 'By': r.subgroup, 'Level': r.level, 'C': f3(r.cindex), 'Slope': f3(r.cal_slope)} for r in sg.itertuples() if r.model in ('MLP', 'CKINN')]),
                  'Subgroup performance, tier T2, MLP versus CKINN.', 'tab:sub')

# bootstrap decision rules C2/C3 (T2, both directions, vs MLP and XGB-mono)
bf = {d: P / f'boot_cross_{d}_T2.csv' for d in ('RICH_to_NAGALA', 'NAGALA_to_RICH')}
if all(f.exists() for f in bf.values()):
    ok, rows = True, []
    for d, f in bf.items():
        b = pd.read_csv(f)
        for comp in ('MLP', 'XGB-mono'):
            sb = b[(b.model == comp) & (b.metric == 'scaled_brier')].iloc[0]      # comp - CKINN
            cd = b[(b.model == comp) & (b.metric == 'cal_slope_dist')].iloc[0]
            better_sb, better_cd = sb.hi95 < 0, cd.lo95 > 0                         # CKINN better
            ok &= bool(better_sb or better_cd)
            rows.append({'Direction': d.replace('_', ' '), 'Comparator': comp,
                         'dScaledBrier (comp-CKINN)': ci(sb['mean'], sb.lo95, sb.hi95),
                         'dCalDist (comp-CKINN)': ci(cd['mean'], cd.lo95, cd.hi95),
                         'CKINN better': 'yes' if (better_sb or better_cd) else 'no'})
    write_tab('boot_rules', pd.DataFrame(rows), 'Paired bootstrap (200 resamples) of comparator minus CKINN, tier T2. CKINN is better when the scaled-Brier interval lies below zero or the calibration-distance interval lies above zero.', 'tab:boot')
    macro('C2C3supported', 'supported' if ok else 'not supported')

# ---------------------------------------------------------------- sensitivity
sens = load('sens*_r*.csv')
if len(sens):
    s = prim(sens)
    s['analysis'] = s.groupby(['tier', 'n_train']).ngroup()
    rows = [{'Tier': r.tier, 'n': int(r.n_train), 'Model': r.model, 'C': f3(r.cindex), 'sBrier': f3(r.scaled_brier)}
            for r in s.groupby(['tier', 'n_train', 'model'], as_index=False)[['cindex', 'scaled_brier']].mean().itertuples()]
    write_tab('sensitivity', pd.DataFrame(rows), 'Sensitivity analyses (S1 diagnosis-only outcome, S4 cohort-specific extras), mean over folds.', 'tab:sens')

# ---------------------------------------------------------------- calibration figure (cross T2)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.3)); any_ = False
for a, d in zip(axs, ('RICH_to_NAGALA', 'NAGALA_to_RICH')):
    f = PRED / f'cross_{d}_T2.npz'
    if not f.exists(): continue
    z = np.load(f); df = pd.DataFrame({'time_y': z['time_y'], 'event': z['event']}); any_ = True
    for m in ('MLP', 'XGB-mono', 'CKINN'):
        t = L.calibration_table_risk(z[m + '__risk'], df)
        a.plot(t.pred, t.obs, 'o-', ms=3, label=m)
    mx = a.get_xlim()[1]; a.plot([0, mx], [0, mx], 'k:'); a.set_title(d.replace('_', ' ')); a.set_xlabel('Predicted 3-year risk'); a.set_ylabel('Observed (Kaplan--Meier)')
if any_:
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / 'calibration_cross_T2.pdf')
plt.close(fig)

# ---------------------------------------------------------------- data facts
for coh in ('rich', 'nagala'):
    f = ROOT / f'data/processed/{coh}_harmonized.csv'
    if f.exists():
        d = pd.read_csv(f, usecols=['time_y', 'event'])
        macro(f'n{coh}', f'{len(d):,}'.replace(',', '{,}')); macro(f'ev{coh}', f'{int(d.event.sum()):,}'.replace(',', '{,}'))
        macro(f'fu{coh}', f'{d.time_y.mean():.2f}')

lines = ['% generated by scripts/make_report.py']
for k, v in NUM.items():
    lines.append('\\newcommand{\\' + k.replace('_', '') + '}{' + str(v).replace('%', '\\%') + '}')
(ROOT / 'paper/numbers.tex').write_text('\n'.join(lines) + '\n')
print('report built:', sorted(p.name for p in TAB.glob('*.tex')), len(NUM), 'macros')
