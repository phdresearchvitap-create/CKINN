"""Task engine: every model fit is one task with a deterministic id, run in a pool of single-threaded worker processes and
checkpointed to disk, so an interrupted run resumes where it stopped (protocol_v2.md, section 6)."""
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_FRAMES: Dict[str, pd.DataFrame] = {}
_CACHE_DIR: Optional[Path] = None


def _init_worker(cache_dir: str) -> None:
    global _CACHE_DIR
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    _CACHE_DIR = Path(cache_dir)


def save_frames(cache_dir: Path, frames: Dict[str, pd.DataFrame]) -> None:
    (cache_dir / "frames").mkdir(parents=True, exist_ok=True)
    for k, f in frames.items():
        p = cache_dir / "frames" / f"{k}.pkl"
        tmp = p.with_suffix(".tmp")
        f.to_pickle(tmp)
        os.replace(tmp, p)
        _FRAMES[k] = f


def frame(key: str) -> pd.DataFrame:
    if key not in _FRAMES:
        _FRAMES[key] = pd.read_pickle(_CACHE_DIR / "frames" / f"{key}.pkl")
    return _FRAMES[key]


def task_path(cache_dir: Path, tid: str) -> Path:
    return cache_dir / "tasks" / f"{tid}.pkl"


def _mahalanobis(X: np.ndarray, mu: np.ndarray, icov: np.ndarray) -> np.ndarray:
    D = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, icov, D), 0))


def run_task(spec: dict, cfg) -> dict:
    """Fit one model and return predictions and requested extras. Pure function of spec and cfg."""
    from .config import BINARY
    from .missing import corrupt
    from .models import Fitted
    from .nets import integrated_gradients, violation_rows
    t0 = time.perf_counter()
    tr = frame(spec["train_frame"]).iloc[spec["train_idx"]].reset_index(drop=True)
    ytr = np.asarray(spec["train_y"]) if spec.get("train_y") is not None else tr.y.to_numpy()
    sw = spec.get("train_w")
    if sw is not None:                                     # IPCW models: rows with weight 0 do not enter the fit
        keep = np.asarray(sw) > 0
        tr, ytr, sw = tr[keep].reset_index(drop=True), ytr[keep], np.asarray(sw, float)[keep]
    fm = Fitted(spec["model"], spec["feats"], spec["hp"], spec["seed"], cfg, signs=spec.get("signs"), lam=spec.get("lam", 0.0))
    fm.fit(tr, ytr, sw)
    out = {"id": spec["id"], "info": fm.info(), "preds": {}, "fit_seconds": time.perf_counter() - t0}
    for t in spec.get("tests", []):
        df = frame(t["frame"]).iloc[t["idx"]].reset_index(drop=True)
        if t.get("corrupt"):
            mech, rate, cseed = t["corrupt"]
            df = corrupt(df, spec["feats"], mech, rate, cseed)
        for f, v in (t.get("set") or {}).items():                      # post hoc: set a predictor to a fixed value (BMI 24 / 35)
            df[f] = float(v)
        out["preds"][t["tag"]] = fm.predict(df).astype(np.float32)
    nn = spec["model"] in ("M1", "M2", "CKINN")
    if nn and spec.get("explain"):
        e = spec["explain"]
        df = frame(e["frame"]).iloc[e["idx"]].reset_index(drop=True)
        X, M = fm.prep.transform(df)
        A = integrated_gradients(fm.net, X, M, cfg.ig_steps)
        out["attr"] = A.astype(np.float32)
        out["viol"] = violation_rows(fm.net, X, M, e["signs"], spec["feats"], cfg.viol_tol, attributions=A)
        if e.get("sparse"):
            Xtr, _ = fm.prep.transform(tr)
            mu, cov = Xtr.mean(0), np.cov(Xtr, rowvar=False) + 1e-6 * np.eye(Xtr.shape[1])
            icov = np.linalg.pinv(cov)
            thr = float(np.quantile(_mahalanobis(Xtr, mu, icov), cfg.sparse_quantile))
            dte = frame(e["frame"]).iloc[e["sparse_pool"]].reset_index(drop=True)
            Xs, Ms = fm.prep.transform(dte)
            dist = _mahalanobis(Xs, mu, icov)
            sel = np.flatnonzero(dist > thr)[: cfg.n_explain]
            out["sparse"] = {"n_pool": int(len(dte)), "n_sparse": int((dist > thr).sum()), "thr": thr,
                             **violation_rows(fm.net, Xs[sel], Ms[sel], e["signs"], spec["feats"], cfg.viol_tol)}
    if nn and spec.get("extra_viol"):
        out["extra_viol"] = {}
        for tag, fkey, idx, signs in spec["extra_viol"]:
            df = frame(fkey).iloc[idx].reset_index(drop=True)
            X, M = fm.prep.transform(df)
            out["extra_viol"][tag] = violation_rows(fm.net, X, M, signs, spec["feats"], cfg.viol_tol)
    if spec.get("pd"):
        p = spec["pd"]
        df = frame(p["frame"]).iloc[p["idx"]].reset_index(drop=True)
        res = {}
        for f in p["features"]:
            if f not in spec["feats"] or f in BINARY:
                continue
            grid = np.nanpercentile(tr[f].to_numpy(float), np.arange(10, 91, 10))
            curve = []
            for v in grid:
                d2 = df.copy()
                d2[f] = v
                curve.append(float(fm.predict(d2).mean()))
            res[f] = {"grid": grid, "pd": np.array(curve)}
        out["pd"] = res
    if spec.get("perturb"):
        from .posthoc import perturbation_violations
        pz = spec["perturb"]
        df = frame(pz["frame"]).iloc[pz["idx"]].reset_index(drop=True)
        out["perturb"] = perturbation_violations(fm.predict, df, tr, pz["signs"], spec["feats"], pz["delta_sd"], cfg.viol_tol * pz["delta_sd"])
    if spec.get("shap"):
        from .posthoc import permutation_shap
        sh = spec["shap"]
        df = frame(sh["frame"]).iloc[sh["idx"]].reset_index(drop=True)
        rng = np.random.default_rng(sh["seed"])
        bg = tr.iloc[rng.choice(len(tr), min(sh["n_bg"], len(tr)), replace=False)].reset_index(drop=True)
        phi, base, fx = permutation_shap(fm.predict, df, bg, spec["feats"], sh["n_perm"], rng)
        out["shap"] = {"phi": phi.astype(np.float32), "base": base, "fx": fx.astype(np.float32),
                       "x": df[spec["feats"]].to_numpy(float).astype(np.float32), "feats": list(spec["feats"])}
        if spec.get("reason"):
            from .posthoc import reasoning_inputs
            rz = spec["reason"]
            k = min(rz["n"], len(df))
            out["reason"] = reasoning_inputs(fm.predict, df.iloc[:k].reset_index(drop=True), tr, rz["signs"], spec["feats"],
                                             rz["delta_sd"], cfg.viol_tol * rz["delta_sd"], rz["panel"])
            out["reason"]["phi"] = phi[:k].astype(np.float32)
    out["seconds"] = time.perf_counter() - t0
    return out


def _worker(spec: dict, cfg, cache_dir: str) -> str:
    path = task_path(Path(cache_dir), spec["id"])
    if path.exists():
        return spec["id"]
    try:
        res = run_task(spec, cfg)
    except Exception:
        raise RuntimeError(f"task {spec['id']} failed:\n{traceback.format_exc()}")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh, protocol=4)
    os.replace(tmp, path)
    return spec["id"]


def run_tasks(specs: Sequence[dict], cfg, cache_dir: Path, n_workers: int, label: str = "") -> None:
    """Run every spec whose checkpoint is missing. Raises if any task fails (after the others finish)."""
    (cache_dir / "tasks").mkdir(parents=True, exist_ok=True)
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    todo = [s for s in specs if not task_path(cache_dir, s["id"]).exists()]
    t0 = time.perf_counter()
    print(f"[{label}] {len(specs)} tasks, {len(specs) - len(todo)} already done, {len(todo)} to run on {n_workers} workers", flush=True)
    if not todo:
        return
    errors = []
    if n_workers <= 1:
        _init_worker(str(cache_dir))
        for i, s in enumerate(todo):
            try:
                _worker(s, cfg, str(cache_dir))
            except Exception as e:  # noqa: BLE001  (collected and raised below)
                errors.append(str(e))
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker, initargs=(str(cache_dir),)) as ex:
            futs = {ex.submit(_worker, s, cfg, str(cache_dir)): s["id"] for s in todo}
            done = 0
            step = max(1, len(todo) // 10)
            for f in as_completed(futs):
                done += 1
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                if done % step == 0 or done == len(todo):
                    print(f"[{label}] {done}/{len(todo)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} task(s) failed; first:\n{errors[0]}")


def load(cache_dir: Path, tid: str) -> dict:
    with open(task_path(cache_dir, tid), "rb") as fh:
        return pickle.load(fh)
