"""Preprocessing fitted on training rows only (v1 behaviour, protocol_v2.md section 4)."""
import warnings
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BINARY, LOG_COLS


class Prep:
    """Winsorise (0.5/99.5 percentiles), log1p for skewed labs, z-score; X has 0 where missing (= training mean), M is 1 where observed."""

    def __init__(self, features: Sequence[str], winsor: Tuple[float, float] = (0.5, 99.5)):
        self.features = list(features)
        self.winsor = winsor
        self.is_bin = np.array([f in BINARY for f in self.features])
        self.is_log = np.array([f in LOG_COLS for f in self.features])

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(float)

    def _warp(self, R, lo, hi):
        T = np.clip(R, lo, hi)
        T[:, self.is_log] = np.log1p(np.maximum(T[:, self.is_log], 0))
        return T

    def _stats(self, R, w=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(R, self.winsor[0], axis=0), np.nanpercentile(R, self.winsor[1], axis=0)
            lo = np.where(self.is_bin | np.isnan(lo), -np.inf, lo)
            hi = np.where(self.is_bin | np.isnan(hi), np.inf, hi)
            T = self._warp(R, lo, hi)
            mu, sd = np.nanmean(T, axis=0), np.nanstd(T, axis=0)
        mu = np.where(self.is_bin | ~np.isfinite(mu), 0.0, mu)
        sd = np.where(self.is_bin | ~np.isfinite(sd) | (sd < 1e-8), 1.0, sd)
        return lo, hi, mu, sd

    def fit(self, df: pd.DataFrame) -> "Prep":
        self.lo, self.hi, self.mu, self.sd = self._stats(self._raw(df))
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        R = self._raw(df)
        M = ~np.isnan(R)
        Z = (self._warp(R, self.lo, self.hi) - self.mu) / self.sd
        return np.where(M, Z, 0.0).astype(np.float32), M.astype(np.float32)

    def z_of_raw(self, feature: str, values: np.ndarray) -> np.ndarray:
        """z-scale value of raw values of one feature (used for partial-dependence grids)."""
        j = self.features.index(feature)
        v = np.clip(np.asarray(values, float), self.lo[j], self.hi[j])
        if self.is_log[j]:
            v = np.log1p(np.maximum(v, 0))
        return ((v - self.mu[j]) / self.sd[j]).astype(np.float32)
