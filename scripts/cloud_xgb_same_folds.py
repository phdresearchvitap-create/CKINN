"""Cloud-only: gradient-boosted Cox baselines (XGB, XGB-mono) on the SAME folds as the iPad notebook (N.cv_folds)."""
import sys, time, pathlib, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/CKINN/src')
from ckinn import np_backend as N
from ckinn import lib as L
D = pathlib.Path(sys.argv[1]); PARTS = D / 'ckinn_cache/parts'
rich = pd.read_csv(D / 'ckinn_cache/rich_harmonized.csv')
for tier in ['T2', 'T1', 'T3', 'T4']:
    cols = N.TIERS[tier]
    for rep in range(3):
        out = PARTS / f'cv_XGB_{tier}_r{rep}.csv'
        if out.exists(): continue
        t0, rows = time.time(), []
        for k, (tr, te) in enumerate(N.cv_folds(rich, 5, rep)):
            seed = 1000 * rep + k
            for name, mono in [('XGB', False), ('XGB-mono', True)]:
                fit = N.fit_with_baseline(L.XGBCox(cols, mono, 300, seed=seed), tr)
                m = N.evaluate(fit, te)
                rows.append({'model': name, 'tier': tier, 'repeat': rep, 'fold': k, 'n_train': len(tr), 'n_test': len(te), **m})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'done {tier} r{rep} in {time.time() - t0:.0f} s', flush=True)
