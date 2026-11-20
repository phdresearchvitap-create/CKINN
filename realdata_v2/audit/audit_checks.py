"""Audit checks for the v1 CKINN vs MLP real-data study (run before any v2 analysis).

Usage: python audit_checks.py <v1_outputs_dir> <v1_clean.ipynb> <v1_executed.ipynb> <nagala.xlsx> <rich.xlsx> <nhanes.csv>
Every number quoted in AUDIT.md is printed by this script. Reads data; writes nothing except stdout.
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

out_dir, nb_clean, nb_exec, f_nag, f_rich, f_nh = map(Path, sys.argv[1:7])
P = lambda *a: print(*a, flush=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

P("== A. provenance of the v1 record ==")
h = json.loads((out_dir / "report/output_hashes.json").read_text())
bad = [k for k, v in h.items() if not (out_dir / k).exists() or sha(out_dir / k) != v]
P(f"A1 output files listed in output_hashes.json: {len(h)}; hash mismatches: {len(bad)}")
caches = [k for k in h if k.startswith("_cache_")]
P(f"A2 raw-data copies inside the v1 outputs archive: {caches}")
man = json.loads((out_dir / "report/manifest.json").read_text())
for k, path in (("NAGALA", f_nag), ("RICH", f_rich), ("NHANES", f_nh)):
    P(f"A3 {k} input sha256 matches v1 manifest: {sha(path) == man['inputs'][k]['sha256']}")
a, b = json.loads(nb_clean.read_text()), json.loads(nb_exec.read_text())
P(f"A4 clean and executed notebooks have identical cell sources: {[''.join(c['source']) for c in a['cells']] == [''.join(c['source']) for c in b['cells']]}")
P(f"A5 error outputs in executed notebook: {sum(o.get('output_type') == 'error' for c in b['cells'] for o in c.get('outputs', []))}")
txt = "\n".join("".join(o.get("text", "")) for c in b["cells"] for o in c.get("outputs", []) if o.get("output_type") == "stream")
e1 = pd.read_csv(out_dir / "tables/table09_main_results.csv")
P(f"A6 table09 AUROC strings printed in executed notebook: {sum(s in txt for s in e1.auroc_ci)} of {len(e1)}")
t10 = pd.read_csv(out_dir / "tables/table10_hypothesis_endpoints.csv")
P(f"A7 table10 estimates printed in executed notebook: {sum(f'{v:.4f}' in txt for v in t10.ckinn_minus_m2)} of {len(t10)}")
v = pd.read_csv(out_dir / "tables/table11_verdicts.csv")
P(f"A8 table11 'NULL' verdicts read back by pandas defaults as missing: {int(v.verdict.isna().sum())} of {len(v)}")
prim = t10[t10.primary]
P(f"A9 primary endpoints pooled in one Holm family: {len(prim)}")
src = "".join("".join(c["source"]) for c in a["cells"])
P(f"A10 'or True' in v1 code: {src.count('or True')}; '... or None' no-op lines: {len(re.findall(r'\) or None', src))}; bare 'except Exception: pass': {len(re.findall(r'except Exception:\s*\n\s*pass', src))}")
P(f"A11 per-model fit seed in engine includes the model name (CKINN and M2 not paired): {'seed_for(\"fit\", coh, tier, m, f)' in src}")
P(f"A12 training-row cap max_train_rows in frozen config: {json.loads((out_dir / 'report/frozen_protocol.json').read_text())['config']['max_train_rows']}")
lam = pd.read_csv(out_dir / "tables/table05_lambda_selection.csv")
ch = lam[lam.chosen]
P(f"A13 lambda chosen at the top of the grid (10): {int((ch['lambda'] == 10).sum())} of {len(ch)} cohort-tiers; "
  f"largest within-cohort spread of dev scores across lambda: {lam.groupby(['cohort', 'tier']).dev_score.agg(lambda s: s.max() - s.min()).max():.4f}")
P(f"A14 M1 described as median-imputed in markdown: {'median-imputed' in src}; Prep fills missing with 0 on the z-scale (training mean): {'np.where(M, Z, 0.0)' in src}")
P(f"A15 M1 trained without augmentation while M2/CKINN use it: {'use_aug = (name != \"M1\")' in src}; augmentation panel-drop 0.3 equals primary test MCAR rate 0.3")
P(f"A16 pooled ablation uses per-frame own statistics at prediction time (transductive): {'transform_aligned' in src and 'own_stats=True' in src}")

P("\n== B. NAGALA ==")
n = pd.read_excel(f_nag)
c = n.columns
fu = n[c[10]] / 365.25
P(f"B1 rows {len(n)}, events {int(n[c[-1]].sum())}, follow-up years min {fu.min():.2f} median {fu.median():.2f} max {fu.max():.2f}; participants followed > 11 years: {int((fu > 11).sum())}")
P(f"B2 sex codes in file: {n[c[1]].value_counts().sort_index().to_dict()} (header says 1 men, 2 women); mean ALT by code: {n.groupby(c[1])[c[11]].mean().round(2).to_dict()}")
P(f"B3 HDL mmol/L stored as 0: {int((n[c[17]] == 0).sum())}; of these with HDL mg/dL blank: {int(((n[c[17]] == 0) & n[c[16]].isna()).sum())}")
for mg, mm in ((16, 17), (18, 19), (20, 21), (26, 27)):
    r = n[c[mg]] / n[c[mm]].replace(0, np.nan)
    P(f"B4 {c[mg][:34]} / mmol column: ratio median {r.median():.3f}, 1st-99th pct {r.quantile(.01):.3f}-{r.quantile(.99):.3f}")
P(f"B5 baseline FPG >= 7.0 mmol/L: {int((n[c[27]] >= 7).sum())}; baseline HbA1c >= 6.5%: {int((n[c[22]] >= 6.5).sum())}")

P("\n== C. RICH ==")
r = pd.read_excel(f_rich)
r.columns = ["id", "age", "sex", "site", "ht", "wt", "bmi", "sbp", "dbp", "fpg", "tc", "tg", "hdl", "ldl", "alt", "ast", "bun", "ccr",
             "fpg_final", "dx_followup", "event", "fu_years", "smoking", "drinking", "family_history"]
P(f"C1 rows {len(r)}, events {int(r.event.sum())}; follow-up years min {r.fu_years.min():.2f} median {r.fu_years.median():.2f} max {r.fu_years.max():.2f}; event times min {r.fu_years[r.event == 1].min():.2f}")
P(f"C2 events by 3 years {int(((r.event == 1) & (r.fu_years <= 3)).sum())}; censored before 3 years {int(((r.event == 0) & (r.fu_years < 3)).sum())}")
fh = r.groupby("site").family_history.agg(["size", "mean"])
P(f"C3 family history 'yes' overall {r.family_history.mean():.4f}; by site min {fh['mean'].min():.4f} max {fh['mean'].max():.4f}; sites with yes-rate below 0.005: {int((fh['mean'] < 0.005).sum())} of {len(fh)} ({int(fh.loc[fh['mean'] < 0.005, 'size'].sum())} participants)")
P(f"C4 smoking missing {r.smoking.isna().mean():.3f}; drinking missing {r.drinking.isna().mean():.3f} (drinking not used by v1)")
P(f"C5 smoking missing by site ranges {r.smoking.isna().groupby(r.site).mean().min():.3f}-{r.smoking.isna().groupby(r.site).mean().max():.3f}; HDL missing by site {r.hdl.isna().groupby(r.site).mean().min():.3f}-{r.hdl.isna().groupby(r.site).mean().max():.3f}")
P(f"C6 exact zeros: TG {int((r.tg == 0).sum())}, HDL {int((r.hdl == 0).sum())}, LDL {int((r.ldl == 0).sum())}, ALT {int((r.alt == 0).sum())}, AST {int((r.ast == 0).sum())} (v1 set TG, HDL and LDL zeros to missing, kept ALT and AST zeros)")
P(f"C7 baseline FPG >= 7.0: {int((r.fpg >= 7).sum())}; final-visit FPG >= 7.0 without an event: {int(((r.fpg_final >= 7) & (r.event == 0)).sum())}")

P("\n== D. NHANES extract ==")
h = pd.read_csv(f_nh, low_memory=False)
h.columns = [x.split("__")[-1] for x in h.columns]
h = h.loc[:, ~h.columns.duplicated(keep="first")]
P(f"D1 rows {len(h)}; rows per SDDSRVYR code: {h.SDDSRVYR.value_counts().sort_index().to_dict()}")
ad = h[(h.SDDSRVYR >= 1) & (h.RIDAGEYR >= 20)]
P(f"D2 adults 20+ in cycles 1-10 (v1 frame): {len(ad)}; adults with code -1 excluded by v1: {int(((h.SDDSRVYR == -1) & (h.RIDAGEYR >= 20)).sum())}")
known, med = ad.DIQ010 == 1, (ad.DIQ050 == 1) | (ad.DIQ070 == 1)
unk = ad.DIQ010.isin([7, 9]) | ad.DIQ010.isna()
fpg_v1 = ad.LBXGLU / 18.016
lab_v1 = (fpg_v1.fillna(-1) >= 7.0) | (ad.LBXGH.fillna(-1) >= 6.5)
keep_v1 = ~known & (fpg_v1.notna() | ad.LBXGH.notna())
P(f"D3 v1 labelled {int(keep_v1.sum())}, positives {int((lab_v1 & keep_v1).sum())}")
P(f"D4 v1-labelled adults taking insulin or diabetes pills without a reported diagnosis: {int((keep_v1 & med).sum())}, of whom positive {int((keep_v1 & med & lab_v1).sum())}")
P(f"D5 v1-labelled adults with DIQ010 refused/don't know/missing: {int((keep_v1 & unk).sum())}")
g = ad.LBXGLU.notna()
P(f"D6 fasting glucose values {int(g.sum())}; with fasting time < 8 h: {int((g & (ad.PHAFSTHR < 8)).sum())}; v1 positives resting only on such a glucose: {int((keep_v1 & (fpg_v1 >= 7) & (ad.PHAFSTHR < 8) & ~(ad.LBXGH >= 6.5)).sum())}")
fpg2 = fpg_v1.where(ad.PHAFSTHR >= 8)
lab2 = (fpg2.fillna(-1) >= 7.0) | (ad.LBXGH.fillna(-1) >= 6.5)
keep2 = ~known & ~med & ~unk & (fpg2.notna() | ad.LBXGH.notna())
P(f"D7 corrected definition (v2): labelled {int(keep2.sum())}, positives {int((lab2 & keep2).sum())}, prevalence {100 * (lab2 & keep2).sum() / keep2.sum():.2f}%")
P(f"D8 survey design variables present: {[x for x in ('SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTMEC4YR', 'WTSAF2YR') if x in h.columns]}; adults with WTMEC2YR = 0: {int((ad.WTMEC2YR == 0).sum())}")
P(f"D9 exercise (PAQ650) observed share by cycle: {ad.groupby('SDDSRVYR').PAQ650.apply(lambda s: round(s.notna().mean(), 2)).to_dict()}")
P(f"D10 alcohol (ALQ101) observed share by cycle: {ad.groupby('SDDSRVYR').ALQ101.apply(lambda s: round(s.notna().mean(), 2)).to_dict()}")
