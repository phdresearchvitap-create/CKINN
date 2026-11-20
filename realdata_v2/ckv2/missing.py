"""Test-time missingness injection (v1 mechanisms at more rates, A16) and the augmentation used by LR_aug and XGB_aug."""
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from .config import GROUP_OF


def panel_cols(feats: Sequence[str]) -> List[str]:
    return [c for c in feats if GROUP_OF[c] == "panel"]


def _intercept_for_rate(score: np.ndarray, rate: float) -> float:
    return float(brentq(lambda a: expit(a + score).mean() - rate, -30, 30))


def corrupt(df: pd.DataFrame, feats: Sequence[str], mech: str, rate: float, seed: int) -> pd.DataFrame:
    """Copy of df with laboratory-panel predictors removed.
    mcar: each row loses the whole panel with probability `rate`. mar: same, more often for younger and leaner people.
    mnar: each lab value removed on its own, more often when the value is low."""
    rng = np.random.default_rng(seed)
    out, cols, n = df.copy(), panel_cols(feats), len(df)
    if not cols or rate <= 0:
        return out
    if mech in ("mcar", "mar"):
        if mech == "mcar":
            drop = rng.random(n) < rate
        else:
            za = ((df.age_years - df.age_years.mean()) / df.age_years.std()).fillna(0).to_numpy()
            zb = ((df.bmi_kgm2 - df.bmi_kgm2.mean()) / df.bmi_kgm2.std()).fillna(0).to_numpy()
            s = -0.8 * (za + zb)
            drop = rng.random(n) < expit(_intercept_for_rate(s, rate) + s)
        out.loc[drop, cols] = np.nan
    elif mech == "mnar":
        for c in cols:
            v = df[c].to_numpy(float)
            z = np.nan_to_num((v - np.nanmean(v)) / (np.nanstd(v) + 1e-9))
            s = -1.0 * z
            out.loc[rng.random(n) < expit(_intercept_for_rate(s, rate) + s), c] = np.nan
    else:
        raise ValueError(mech)
    return out


def augment_frame(df: pd.DataFrame, y: np.ndarray, feats: Sequence[str], seed: int, aug_prob: float, p_panel: float,
                  p_feat: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """v1 augmentation for tree and linear baselines: append a corrupted copy of a share of the training rows."""
    rng = np.random.default_rng(seed)
    pick = np.flatnonzero(rng.random(len(df)) < aug_prob)
    cp = corrupt(df.iloc[pick], feats, "mcar", p_panel, seed + 1)
    for c in feats:
        if GROUP_OF[c] == "clin":
            cp.loc[rng.random(len(cp)) < p_feat, c] = np.nan
    return pd.concat([df, cp], ignore_index=True), np.r_[y, y[pick]]


def scenarios(cfg) -> List[Tuple[str, float]]:
    return [(m, r) for m in cfg.miss_mechs for r in cfg.miss_rates]


def scen_tag(mech: str, rate: float) -> str:
    return f"{mech}{int(round(rate * 100))}"
