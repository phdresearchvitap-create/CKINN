"""Run the locked CKINN protocol v1.0 on real data. Resumable: every unit writes a part file and is skipped if present.

Usage: python scripts/run_experiments.py {cv,sens,loso,shift,cross} [--force]
Cross-cohort and shift analyses refuse to run unless protocol/LOCK.sha256 matches protocol_v1.0_LOCKED.md.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ckinn import lib as L

CFG = json.load(open(ROOT / 'configs/protocol_v1.json'))
RES, PARTS, PREDS = ROOT / 'results', ROOT / 'results/parts', ROOT / 'results/preds'
for p in (RES, PARTS, PREDS):
    p.mkdir(parents=True, exist_ok=True)
EP, XN = CFG['models']['epochs'], CFG['models']['xgb_n']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_ok():
    return sha(ROOT / 'protocol/protocol_v1.0_LOCKED.md') == (ROOT / 'protocol/LOCK.sha256').read_text().strip()


def log_manifest(unit):
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    rec = {'unit': unit, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'commit': commit,
           'protocol_sha': sha(ROOT / 'protocol/protocol_v1.0_LOCKED.md')}
    with open(RES / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')


def specs(cols, seed, with_viol_models=True):
    return {'Cox': lambda: L.CoxLinear(cols),
            'XGB': lambda: L.XGBCox(cols, False, XN, seed=seed),
            'XGB-mono': lambda: L.XGBCox(cols, True, XN, seed=seed),
            'MLP': lambda: L.DeepCox(cols, lam=0.0, epochs=EP, seed=seed),
            'CKINN': lambda: L.CKINNAuto(cols, tuple(CFG['models']['ckinn_grid']), CFG['models']['tol'], EP, seed)}


def tier_df(df, cols):
    return df.dropna(subset=cols)


def part(name):
    return PARTS / f'{name}.csv'


def unit(name, force, fn):
    if part(name).exists() and not force:
        print('skip', name, flush=True)
        return
    t0 = time.time()
    out = fn()
    out.to_csv(part(name), index=False)
    log_manifest(name)
    print(f'done {name} in {time.time() - t0:.0f}s', flush=True)


def analyse(train, test, cols, seed, recal_n=None, tag=None, subgroup=False, save_preds=None, boot=None):
    rows, fits, arrs = [], {}, {}
    for name, make in specs(cols, seed).items():
        f = L.fit_with_baseline(make(), train)
        fits[name] = f
        lam = getattr(f.model, 'lam_', np.nan)
        t = test[test.time_y < train.time_y.max()]
        lp, risk = f.model.lp(t), f.risk(t)
        arrs[name] = (lp, risk)
        rows.append({'model': name, 'lam': lam, **L.metrics_from_arrays(train, t, lp, risk, viol=L.violation_rate(f.model, t)[0])})
        if recal_n:
            tgt = t.sample(recal_n, random_state=seed)
            rest = t.drop(tgt.index)
            rf, sl = L.recalibrate(f, tgt)
            rows.append({'model': name + ' +recal', 'lam': lam, 'recal_slope': sl, **L.evaluate(rf, rest)})
        if subgroup:
            for by in ('sex_male', 'age_band'):
                sg = L.subgroup_metrics(f, t, by)
                for _, r in sg.iterrows():
                    rows.append({'model': name, 'subgroup': by, 'level': str(r[by]), **{k: r[k] for k in sg.columns if k != by}})
    res = pd.DataFrame(rows)
    if save_preds:
        t = test[test.time_y < train.time_y.max()]
        np.savez_compressed(PREDS / f'{save_preds}.npz', time_y=t.time_y.values, event=t.event.values,
                            **{k + '__risk': v[1] for k, v in arrs.items()}, **{k + '__lp': v[0] for k, v in arrs.items()})
    if boot:
        t = test[test.time_y < train.time_y.max()]
        bd = L.boot_diff_arrays(train, t, arrs, 'CKINN', n_boot=boot['n'], seed=seed, max_n=boot['max_n'])
        bd.to_csv(PARTS / f'boot_{tag}.csv', index=False)
    return res


# ----------------------------------------------------------------- analyses
def run_cv(rich, force, tiers=None, repeats=None, folds=None, prefix='cv', df_override=None, colmap=None):
    tiers = tiers or CFG['cv']['tiers']
    repeats = repeats if repeats is not None else CFG['cv']['repeats']
    folds = folds or CFG['cv']['folds']
    for tier in tiers:
        cols = colmap[tier] if colmap else L.TIERS[tier]
        d = tier_df(df_override if df_override is not None else rich, cols)
        for rep in range(repeats):
            def job():
                rows = []
                for k, (tr, te) in enumerate(L.cv_folds(d, folds, rep)):
                    r = analyse(tr, te, cols, 1000 * rep + k)
                    r['repeat'], r['fold'], r['tier'], r['n_train'], r['n_test'] = rep, k, tier, len(tr), len(te)
                    rows.append(r)
                    print(f'  {prefix} {tier} r{rep} f{k}', flush=True)
                return pd.concat(rows)
            unit(f'{prefix}_{tier}_r{rep}', force, job)


def run_sens(rich, nag, force):
    ex = CFG['extras']
    s = CFG['sens']
    # S1 diagnosis-only outcome
    d1 = rich.assign(event=rich.event_dx_only.fillna(rich.event).astype(int)) if 'event_dx_only' in rich else rich
    run_cv(d1, force, ['T2'], s['repeats'], s['folds'], 'sensS1')
    # S2 T4 complete cases is the primary T4 definition; add full-data T3-vs-T4 contrast via XGB only is not pre-specified
    # S4 cohort-specific extras
    for coh, df in (('RICH', rich), ('NAGALA', nag)):
        x = ex[coh]
        cm = {'E2': L.TIERS['T2'] + x, 'E3': L.TIERS['T3'] + x + (ex['NAGALA_E3_extra'] if coh == 'NAGALA' else [])}
        if coh == 'NAGALA':
            cm['E2'] = L.TIERS['T2'] + x
        run_cv(df, force, ['E2', 'E3'], s['repeats'], s['folds'], f'sensS4{coh}', colmap=cm)


def run_loso(rich, force):
    for tier in CFG['loso']['tiers']:
        cols = L.TIERS[tier]
        d = tier_df(rich, cols)
        for site, tr, te in L.loso_folds(d):
            if te.time_y.max() <= L.H:      # 3-year metrics undefined (see protocol/deviations.md D-1)
                print('skip site', site, 'max follow-up', round(te.time_y.max(), 2), flush=True)
                continue
            sp = f'loso_{tier}_s{site}' if tier in CFG['loso']['save_preds_tiers'] else None
            sg = tier == CFG['loso']['subgroup_tier']
            def job():
                r = analyse(tr, te, cols, int(site), subgroup=sg, save_preds=sp)
                r['site'], r['tier'], r['n_train'], r['n_test'] = site, tier, len(tr), len(te)
                return r
            unit(f'loso_{tier}_s{site}', force, job)


def run_shift(rich, force):
    for var, cut in CFG['shift']['splits']:
        for tier in CFG['shift']['tiers']:
            cols = L.TIERS[tier]
            d = tier_df(rich, cols)
            tr, te = L.shift_split(d, var, cut)
            def job():
                r = analyse(tr, te, cols, 0, recal_n=CFG['cross']['recal_n'])
                r['split'], r['tier'], r['n_train'], r['n_test'] = f'{var}<{cut}->{var}>={cut}', tier, len(tr), len(te)
                return r
            unit(f'shift_{var}{cut}_{tier}', force, job)


def run_cross(rich, nag, force):
    c = CFG['cross']
    jobs = [('RICH', 'NAGALA', t) for t in c['tiers']] + [('NAGALA', 'RICH', t) for t in c['tiers']]
    if c['t4_rich_to_nagala']:
        jobs.append(('RICH', 'NAGALA', 'T4'))
    D = {'RICH': rich, 'NAGALA': nag}
    for src, dst, tier in jobs:
        for nosex in ([False, True] if tier in ('T2', 'T3') else [False]):
            cols = [x for x in L.TIERS[tier] if not (nosex and x == 'sex_male')]
            tr, te = tier_df(D[src], cols), tier_df(D[dst], cols)
            tag = f'cross_{src}_to_{dst}_{tier}{"_nosex" if nosex else ""}'
            b = c['boot'] if (tier in c['boot']['tiers'] and not nosex) else None
            sg = (tier == c['subgroup_tier']) and not nosex
            def job():
                r = analyse(tr, te, cols, 0, recal_n=c['recal_n'], tag=tag, subgroup=sg,
                            save_preds=None if nosex else tag, boot=b)
                r['train'], r['test'], r['tier'], r['nosex'], r['n_train'], r['n_test'] = src, dst, tier, nosex, len(tr), len(te)
                return r
            unit(tag, force, job)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['cv', 'sens', 'loso', 'shift', 'cross'])
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.what in ('shift', 'cross') and not lock_ok():
        sys.exit('LOCK mismatch: protocol_v1.0_LOCKED.md changed after locking. Record a deviation first.')
    rich, nag = L.load_data(ROOT / 'data/processed')
    {'cv': lambda: run_cv(rich, a.force), 'sens': lambda: run_sens(rich, nag, a.force),
     'loso': lambda: run_loso(rich, a.force), 'shift': lambda: run_shift(rich, a.force),
     'cross': lambda: run_cross(rich, nag, a.force)}[a.what]()
