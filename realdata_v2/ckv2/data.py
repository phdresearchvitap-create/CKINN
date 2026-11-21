"""Loading, harmonisation and splitting of the three cohorts (protocol_v2.md, section 3).

Differences from v1 (each one is an amendment in protocol_v2.md section 2): RICH ALT and AST zeros set to missing (A12);
NHANES label corrected (A11) with the v1 label kept in `label_v1` for the sensitivity analysis.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ALL_FEATURES, GLYC, LOW_COST, PANEL, seed_for


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_file(roots: Sequence[Path], patterns: Sequence[str]) -> Path:
    hits = []
    for root in roots:
        if root.exists():
            for pat in patterns:
                hits += sorted(root.glob(pat))
    if not hits:
        raise FileNotFoundError(f"none of {patterns} found in {[str(r) for r in roots]}")
    if len({sha256_file(h) for h in hits}) > 1:
        raise RuntimeError(f"several different files match {patterns}: {[str(h) for h in hits]}; keep one")
    return hits[0]


def read_cached(path: Path, cache_dir: Path) -> pd.DataFrame:
    """xlsx parsing is slow; a csv copy is cached in the (never archived) cache folder."""
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    cache = cache_dir / f"_raw_{sha256_file(path)[:16]}.csv"
    if not cache.exists():
        pd.read_excel(path).to_csv(cache, index=False)
    return pd.read_csv(cache, low_memory=False)


def empty_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({c: np.full(n, np.nan) for c in ALL_FEATURES})


def load_nagala(raw: pd.DataFrame) -> pd.DataFrame:
    n = raw.copy()
    n.columns = ["id", "sex", "age", "etoh", "fatty_liver", "bmi", "waist", "wc_hi", "bmi25", "phenotype", "fu_days", "alt", "ast", "wt",
                 "exercise", "ggt", "hdl_mg", "hdl", "tc_mg", "tc", "tg_mg", "tg", "a1c", "a1c_mmol", "alcohol", "smoking",
                 "fpg_mg", "fpg", "sbp", "dbp", "incident"]
    assert len(n) == 15464 and n["incident"].sum() == 373, "unexpected NAGALA file"
    d = empty_frame(len(n))
    d["age_years"], d["sex_female"] = n.age, 1 - n.sex                       # file codes 1 = men, 0 = women (AUDIT B2)
    d["bmi_kgm2"], d["waist_cm"], d["sbp_mmHg"], d["dbp_mmHg"] = n.bmi, n.waist, n.sbp, n.dbp
    d["smoker"] = (n.smoking == 3).astype(float)                            # 1 never, 2 past, 3 current
    d["alcohol_level"] = n.alcohol - 1                                       # 1 none ... 4 heavy -> 0..3
    d["exercise_regular"] = n.exercise
    d["tg_mmol"], d["hdl_mmol"], d["tc_mmol"] = n.tg.where(n.tg > 0), n.hdl.where(n.hdl > 0), n.tc
    d["alt_u_l"], d["ast_u_l"], d["ggt_u_l"] = n.alt.where(n.alt > 0), n.ast.where(n.ast > 0), n.ggt
    d["fpg_mmol"], d["hba1c_pct"] = n.fpg, n.a1c
    d["tg_hdl_ratio"] = d.tg_mmol / d.hdl_mmol                          # post hoc feature (PH)
    d["cohort"], d["site"], d["task"] = "NAGALA", "0", "B"
    d["label"] = n.incident.astype(float)
    d["follow_years"] = n.fu_days / 365.25
    d["group_extra"] = "all"
    return d


def load_rich(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.copy()
    r.columns = ["id", "age", "sex", "site", "ht", "wt", "bmi", "sbp", "dbp", "fpg", "tc", "tg", "hdl", "ldl", "alt", "ast", "bun", "ccr",
                 "fpg_final", "dx_followup", "event", "fu_years", "smoking", "drinking", "family_history"]
    assert len(r) == 211833 and r["event"].sum() == 4174, "unexpected Rich file"
    d = empty_frame(len(r))
    d["age_years"], d["sex_female"] = r.age, (r.sex == 2).astype(float)
    d["bmi_kgm2"], d["sbp_mmHg"], d["dbp_mmHg"] = r.bmi, r.sbp, r.dbp
    d["smoker"] = np.where(r.smoking.isna(), np.nan, (r.smoking == 1).astype(float))   # 1 current, 2 ever, 3 never
    d["family_history_dm"] = r.family_history
    for c_out, c_in in (("tg_mmol", "tg"), ("hdl_mmol", "hdl"), ("ldl_mmol", "ldl"), ("alt_u_l", "alt"), ("ast_u_l", "ast")):
        d[c_out] = r[c_in].where(r[c_in] > 0)                                # exact zeros = not measured (A12)
    d["tc_mmol"], d["fpg_mmol"] = r.tc, r.fpg
    d["tg_hdl_ratio"] = d.tg_mmol / d.hdl_mmol                          # post hoc feature (PH)
    d["fpg_final"] = r.fpg_final                                        # final-visit FPG: post hoc progression analysis only, never a predictor
    d["cohort"], d["site"], d["task"] = "RICH", r.site.astype(int).astype(str), "B"
    d["label"] = r.event.astype(float)
    d["follow_years"] = r.fu_years
    d["group_extra"] = d["site"]
    return d


def load_nhanes(raw: pd.DataFrame) -> pd.DataFrame:
    h = raw.copy()
    h.columns = [c.split("__")[-1] if c != "SEQN" else c for c in h.columns]
    h = h.loc[:, ~h.columns.duplicated(keep="first")]
    h = h[(h.SDDSRVYR >= 1) & (h.RIDAGEYR >= 20)].reset_index(drop=True)       # adults, cycles 1999-2018 (code -1 excluded)
    d = empty_frame(len(h))
    d["age_years"], d["sex_female"] = h.RIDAGEYR, (h.RIAGENDR == 2).astype(float)
    d["bmi_kgm2"], d["waist_cm"] = h.BMXBMI, h.BMXWAIST
    d["sbp_mmHg"] = h.VNAVEBPXSY.fillna(h.BPXSY1)
    d["dbp_mmHg"] = h[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1)
    drinks = h.ALQ130.where(h.ALQ130 < 100)
    d["alcohol_level"] = np.where(h.ALQ101 == 2, 0.0, np.where(drinks.isna(), np.nan, np.where(drinks <= 1, 1.0, np.where(drinks <= 2, 2.0, 3.0))))
    d["exercise_regular"] = np.where(h.PAQ650 == 1, 1.0, np.where(h.PAQ650 == 2, 0.0, np.nan))
    d["family_history_dm"] = np.where(h.MCQ300C == 1, 1.0, np.where(h.MCQ300C == 2, 0.0, np.nan))
    d["tg_mmol"], d["hdl_mmol"], d["ldl_mmol"], d["tc_mmol"] = h.LBXTR / 88.57, h.LBDHDD / 38.67, h.LBDLDL / 38.67, h.LBXTC / 38.67
    d["alt_u_l"], d["ast_u_l"], d["ggt_u_l"] = h.LBXSATSI, h.LBXSASSI, h.LBXSGTSI
    fpg_all = h.LBXGLU / 18.016
    d["fpg_mmol"], d["hba1c_pct"] = fpg_all, h.LBXGH
    d["tg_hdl_ratio"] = d.tg_mmol / d.hdl_mmol                          # post hoc features (PH): ratio and history questions
    d["prediabetes_told"] = np.where(h.DIQ160 == 1, 1.0, np.where(h.DIQ160 == 2, 0.0, np.nan))
    d["hypertension_told"] = np.where(h.BPQ020 == 1, 1.0, np.where(h.BPQ020 == 2, 0.0, np.nan))
    # v1 label (kept for the sensitivity analysis)
    known = (h.DIQ010 == 1)
    lab_v1 = np.where(fpg_all.notna() | h.LBXGH.notna(), ((fpg_all.fillna(-1) >= 7.0) | (h.LBXGH.fillna(-1) >= 6.5)).astype(float), np.nan)
    d["label_v1"] = np.where(known, np.nan, lab_v1)
    # v2 label (A11): no diagnosis, no glucose-lowering medication, diagnosis question answered, FPG only after >= 8 h fasting
    med = (h.DIQ050 == 1) | (h.DIQ070 == 1)
    unknown = h.DIQ010.isin([7, 9]) | h.DIQ010.isna()
    fpg_fast = fpg_all.where(h.PHAFSTHR >= 8)
    lab = np.where(fpg_fast.notna() | h.LBXGH.notna(), ((fpg_fast.fillna(-1) >= 7.0) | (h.LBXGH.fillna(-1) >= 6.5)).astype(float), np.nan)
    d["label"] = np.where(known | med | unknown, np.nan, lab)
    d["cohort"], d["site"], d["task"] = "NHANES", h.SDDSRVYR.astype(int).astype(str), "A"
    d["follow_years"] = np.nan
    d["group_extra"] = h.RIDRETH1.map({1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
                                       5: "Other/multiracial"}).fillna("unknown")
    # survey design (A19): combined MEC weight for 10 cycles (WTMEC4YR for 1999-2002), strata and PSU
    w = np.where(h.SDDSRVYR <= 2, h.WTMEC4YR * 2 / 10, h.WTMEC2YR / 10)
    d["svy_w"], d["svy_strata"], d["svy_psu"] = w, h.SDMVSTRA.astype(float), h.SDMVPSU.astype(float)
    return d


def task_frame(d: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    t = d[d[label_col].notna()].copy()
    t["raw_row"] = t.index.to_numpy()                       # row identity in the harmonised frame (used to keep splits disjoint)
    t = t.reset_index(drop=True)
    t["age_band"] = np.where(t.age_years < 40, "20-39", np.where(t.age_years < 60, "40-59", "60+"))
    t["sex"] = np.where(t.sex_female == 1, "female", "male")
    t["y"] = t[label_col].astype(int)
    return t


def dev_eval_split(t: pd.DataFrame, cohort_key: str, dev_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The v1 split: stratified, seed_for('dev', cohort). The development rows are never evaluated."""
    idx_dev, idx_eval = train_test_split(np.arange(len(t)), train_size=dev_frac, stratify=t.y, random_state=seed_for("dev", cohort_key))
    return t.iloc[np.sort(idx_dev)].reset_index(drop=True), t.iloc[np.sort(idx_eval)].reset_index(drop=True)


def subsample(t: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    if not n or len(t) <= n:
        return t
    keep = train_test_split(np.arange(len(t)), train_size=n, stratify=t.y, random_state=seed_for("smoke", key))[0]
    return t.iloc[np.sort(keep)].reset_index(drop=True)


def cap_rows(d: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    if len(d) <= n:
        return d
    keep = train_test_split(np.arange(len(d)), train_size=n, stratify=d.y, random_state=seed_for("devcap", key))[0]
    return d.iloc[np.sort(keep)].reset_index(drop=True)


def features_for(t: pd.DataFrame, tier: str) -> List[str]:
    cols = LOW_COST + (PANEL if tier in ("T2", "T3") else []) + (GLYC if tier == "T3" else [])
    if tier == "T3" and t.task.iloc[0] == "A":
        raise ValueError("glycaemic predictors define the Task A label: tier T3 is not allowed for detection")
    return [c for c in cols if t[c].notna().mean() > 0.0]


FORBIDDEN = {"A": set(GLYC) | {"known_dm", "label", "label_v1", "y"}, "B": {"label", "y", "follow_years", "event"}}


def assert_no_leakage(features: Sequence[str], task: str, tier: str) -> None:
    bad = set(features) & (FORBIDDEN[task] | (set(GLYC) if tier != "T3" else set()))
    if bad:
        raise AssertionError(f"leakage: forbidden predictor(s) {sorted(bad)} for task {task}, tier {tier}")
    if len(set(features)) != len(features):
        raise AssertionError("duplicated predictors")


def repeated_folds(y: np.ndarray, cohort_key: str, n_splits: int, n_repeats: int) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """Repeated stratified K-fold. Repeat 0 uses the v1 seed so that its folds equal v1's for NAGALA and RICH."""
    out = []
    for r in range(n_repeats):
        rs = seed_for("cv", cohort_key) if r == 0 else seed_for("cv", cohort_key, r)
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=rs)
        out.append([(np.sort(i), np.sort(j)) for i, j in skf.split(np.zeros(len(y)), y)])
    return out
