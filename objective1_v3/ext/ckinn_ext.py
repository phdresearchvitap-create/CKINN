"""Post hoc extension analyses (label EXT) for Objective 1, task A (RICH development, NAGALA frozen evaluation).

Rules: docs/EXT_PRESPECIFICATION.md (fixed before any EXT result). The module imports the v2 package (`study/`) and
never edits it. It lives outside `study/`, so the run identity (which hashes every file in `study/`) and all v2
checkpoints stay valid. New fits go through `Study.fit` / `Study.tune`, so they are checkpointed and resumable under
the run's private folder. Outputs: aggregate CSV files in <results>/tables/<profile>/<run>/A/EXT/. Nothing
participant-level is written outside the private checkpoint folder.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data import CORE, csv, dump, restore, status
from stats import weights, metrics, brier, calibration
from models import DIRECTIONS
from analysis import grad_raw, probability
from runner import CLASSIC

EXP_DIRS = ["fpg", "bmi", "age", "sbp", "tg", "hdl"]
LAMBDA_PATH = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
SIZES = [1000, 3000, 10000, 30000]
DRAWS = [1, 2, 3, 4, 5]
SIZE_LAMBDAS = [10.0, 1e3, 1e5]
THRESHOLDS_NB = [0.005, 0.01, 0.02, 0.03, 0.05]
STEPS = {"fpg": .1, "bmi": 1., "age": 1., "sbp": 5., "tg": .1, "hdl": .1}
NEURAL = ["M1", "M3", "fixed_zero", "fixed_positive", "expanded_prior"]


def out(st):
    p = st.table / "A" / "EXT"
    p.mkdir(parents=True, exist_ok=True)
    return p


def n_boot(st):
    """1,000 draws in STANDARD and FULL (pre-specified); the profile's own small number in SMOKE."""
    return st.b["bootstrap"] if st.mode == "SMOKE" else 1000


# ---------------------------------------------------------------- E1: plans from the stored selection table
def plans_from_selection(selection_csv):
    """Rebuild the frozen final plans from source_selection.csv (used only to reconstruct a run whose private
    checkpoints are not available, e.g. outside the original Colab Drive)."""
    sel = pd.read_csv(selection_csv)
    plans = {}
    for r in sel.itertuples():
        h = json.loads(r.settings)
        neural = r.model not in CLASSIC
        plans[r.model] = dict(h=h, epochs=int(r.epochs) if neural else 1, score=float(r.source_cv_brier),
                              threshold=float(r.threshold), trial=int(r.candidate_index))
    return plans


def compare_external(st, reference_csv):
    """E1 check: this run's external_metrics.csv against a reference run's file."""
    mine = pd.read_csv(st.table / "A" / "external_metrics.csv").set_index("model")
    ref = pd.read_csv(reference_csv).set_index("model")
    rows = []
    for m in ref.index:
        if m not in mine.index:
            rows.append(dict(model=m, status="missing in this run"))
            continue
        db, da = abs(mine.loc[m, "brier"] - ref.loc[m, "brier"]), abs(mine.loc[m, "auroc"] - ref.loc[m, "auroc"])
        rows.append(dict(model=m, brier_this=mine.loc[m, "brier"], brier_reference=ref.loc[m, "brier"], abs_diff_brier=db,
                         auroc_this=mine.loc[m, "auroc"], auroc_reference=ref.loc[m, "auroc"], abs_diff_auroc=da,
                         status="match" if db < 1e-6 and da < 1e-6 else "MISMATCH"))
    csv(out(st) / "E1_reconstruction_check.csv", rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers
def frozen_models(st):
    folder, m = st.frozen("A")
    models = {n: [restore(folder / f) for f in m["files"] if f.rsplit("_", 1)[0] == n] for n in m["plans"]}
    return models, m


def prob_gradients(models, d):
    """Gradient of the seed-averaged probability with respect to the standardised inputs, plus the support mask."""
    f0 = models[0]
    X, inside = f0.prep.transform(d)
    g = np.zeros((len(d), len(f0.prep.names)))
    for f in models:
        f.obj.eval()
        for idx in np.array_split(np.arange(len(d)), max(1, len(d) // 20000)):
            x = torch.from_numpy(X[idx]).requires_grad_(True)
            g[idx] += torch.autograd.grad(f.obj(x).sigmoid().sum(), x)[0][:, :len(f0.prep.names)].detach().numpy()
    return g / len(models), inside, f0.prep.names


def weighted_bce(d, p):
    y, w, _ = weights(d)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))


# ---------------------------------------------------------------- E2: penalty diagnostics
def penalty_diagnostics(st):
    models, m = frozen_models(st)
    rows = []
    for cohort in ["RICH", "NAGALA"]:
        d = st.data(cohort)
        for n in [x for x in NEURAL if x in models]:
            g, inside, names = prob_gradients(models[n], d)
            p = probability(models[n], d)
            h = m["plans"][n]["h"]
            penalised = h.get("directions", []) if h["lambda"] else []
            pen = []
            for f in EXP_DIRS:
                j = names.index(f)
                ok = inside[:, j]
                s = DIRECTIONS[f]
                term = np.maximum(0, -s * g[ok, j]) ** 2
                if f in penalised:
                    pen.append(term)
                rows.append(dict(cohort=cohort, model=n, lambda_value=h["lambda"], feature=f, penalised=f in penalised,
                                 eligible_pairs=int(ok.sum()), wrong_way_share=float(np.mean(s * g[ok, j] < 0)),
                                 median_abs_dp_dz=float(np.median(abs(g[ok, j]))), mean_squared_wrong_way=float(term.mean())))
            data_loss = weighted_bce(d, p)
            pen_value = float(np.concatenate(pen).mean()) if pen else 0.0
            rows.append(dict(cohort=cohort, model=n, lambda_value=h["lambda"], feature="ALL_PENALISED", penalised=True,
                             weighted_data_loss=data_loss, clinical_penalty=pen_value,
                             lambda_times_penalty=h["lambda"] * pen_value,
                             ratio_to_data_loss=h["lambda"] * pen_value / data_loss))
    csv(out(st) / "E2_penalty_diagnostics.csv", rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E3, E4, E5: paired bootstrap of many statistics
def _nb(y, w, p, t, n):
    s = p >= t
    return (np.sum(w * y * s) - t / (1 - t) * np.sum(w * (1 - y) * s)) / n


def external_comparisons(st):
    d, p, m = st.results("A")
    names = [n for n in m["plans"] if n in p]
    ref = "reference"
    y, w, _ = weights(d)
    N = len(d)

    def stats_of(y, w, idx):
        b = {n: float(np.mean(w * (y - p[n][idx]) ** 2)) for n in names}
        nb = {(n, t): _nb(y, w, p[n][idx], t, len(idx)) for n in ["M1", "M3", "LR"] for t in THRESHOLDS_NB}
        for t in THRESHOLDS_NB:
            nb[("treat_all", t)] = (np.sum(w * y) - t / (1 - t) * np.sum(w * (1 - y))) / len(idx)
        return b, nb

    b0, nb0 = stats_of(y, w, np.arange(N))
    rng = np.random.default_rng(st.seed + 40)
    draws_b, draws_nb, failed = [], [], 0
    B = n_boot(st)
    for _ in range(B):
        i = rng.integers(N, size=N)
        dd = d.iloc[i].reset_index(drop=True)
        try:
            yy, ww, _ = weights(dd)
            if np.unique(yy[ww > 0]).size < 2:
                raise ValueError("single class")
            bb, nn = stats_of(yy, ww, i)
            draws_b.append(bb)
            draws_nb.append(nn)
        except (ValueError, FloatingPointError):
            failed += 1
    DB = pd.DataFrame(draws_b)
    ok = len(DB) >= max(2, .8 * B)

    def ci(v):
        return tuple(np.quantile(v, [.025, .975])) if ok else (np.nan, np.nan)

    rows = []
    for a in ["M1", "M3"]:
        for x in [n for n in ["reference", "LR", "RF", "XGB", "LGB", "SVM", "Cox"] if n in names]:
            lo, hi = ci(DB[a] - DB[x])
            rows.append(dict(comparison=f"{a} minus {x}", delta_brier=b0[a] - b0[x], lo=lo, hi=hi, valid_boot=len(DB),
                             requested_boot=B, failed_boot=failed))
    csv(out(st) / "E3_paired_brier_vs_baselines.csv", rows)
    rows = []
    for n in names:
        ipa = 1 - DB[n] / DB[ref]
        lo, hi = ci(ipa)
        rows.append(dict(model=n, brier=b0[n], brier_reference=b0[ref], ipa=1 - b0[n] / b0[ref], lo=lo, hi=hi, valid_boot=len(DB)))
    csv(out(st) / "E4_index_of_prediction_accuracy.csv", rows)
    NB = pd.DataFrame([{f"{k[0]}|{k[1]}": v for k, v in x.items()} for x in draws_nb])
    rows = []
    for (n, t), v in nb0.items():
        lo, hi = ci(NB[f"{n}|{t}"])
        rows.append(dict(model=n, threshold=t, net_benefit=v, lo=lo, hi=hi))
    for t in THRESHOLDS_NB:
        rows.append(dict(model="treat_none", threshold=t, net_benefit=0.0, lo=0.0, hi=0.0))
    csv(out(st) / "E5_net_benefit.csv", rows)
    return True


# ---------------------------------------------------------------- E6, E7: estimand-aligned subset and HDL removed
def _score_all(st, d, label):
    models, m = frozen_models(st)
    rows = []
    for n, fs in models.items():
        pr = probability(fs, d)
        rows.append(dict(analysis=label, model=n, **metrics(d, pr, m["plans"][n]["threshold"])))
    return rows, models


def aligned_and_hdl(st):
    nag = st.data("NAGALA")
    aligned = nag[nag.time >= 2].reset_index(drop=True)
    rows, models = _score_all(st, aligned, "NAGALA follow-up >= 2 years (RICH-like)")
    csv(out(st) / "E6_estimand_aligned_nagala.csv", rows)
    nohdl = nag.assign(hdl=np.nan)
    rows, _ = _score_all(st, nohdl, "NAGALA with HDL set to missing")
    base = {n: probability(fs, nag) for n, fs in models.items()}
    for r in rows:
        q = probability(models[r["model"]], nohdl)
        r["mean_risk_change_vs_observed_hdl"] = float(np.mean(q - base[r["model"]]))
        r["mean_abs_risk_change"] = float(np.mean(abs(q - base[r["model"]])))
    csv(out(st) / "E7_nagala_hdl_missing.csv", rows)
    return True


# ---------------------------------------------------------------- E8: finite-step direction checks, six directions
def directions_all(st):
    models, m = frozen_models(st)
    target = st.data("NAGALA")
    d = target.sample(min(512, len(target)), random_state=st.seed + 19).reset_index(drop=True)
    rows = []
    for n in [x for x in ["M1", "M3", "fixed_positive", "expanded_prior"] if x in models]:
        fs = models[n]
        pp = fs[0].prep
        for f in EXP_DIRS:
            j = pp.names.index(f)
            step, eps = STEPS[f], .005 * pp.scale[j]
            dd = d.loc[d[f].between(pp.lo[j] + eps, pp.hi[j] - max(step, eps))].copy()
            if not len(dd):
                continue
            center = probability(fs, dd)
            change = probability(fs, dd.assign(**{f: dd[f] + step})) - center
            s = DIRECTIONS[f]
            rows.append(dict(model=n, lambda_value=m["plans"][n]["h"]["lambda"], feature=f, penalised=f in m["plans"][n]["h"].get("directions", []) and m["plans"][n]["h"]["lambda"] > 0,
                             raw_step=step, n_eligible=len(dd), violation_rate=float(np.mean(s * change < -1e-6)),
                             mean_wrong_way_change=float(np.maximum(0, -s * change).mean()),
                             mean_change=float(change.mean()), meaningful_fraction=float(np.mean(abs(change) >= .001)),
                             flat_fraction=float(np.mean(abs(change) <= 1e-6))))
    csv(out(st) / "E8_directions_six.csv", rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E9: penalty-strength path, six directions
def lambda_path(st):
    _, m = frozen_models(st)
    base, ep = m["plans"]["M3"]["h"], m["plans"]["M3"]["epochs"]
    d, nag = st.data("RICH"), st.data("NAGALA")
    rows = []
    for lam in LAMBDA_PATH:
        h = dict(base, directions=EXP_DIRS, **{"lambda": float(lam)})
        cfg = st.tune("A", "ext_lambda_path", d, "A_final", grid=[h], epochs=ep)
        model, _ = st.fit("A", "ext_lambda_path", d, None, h, st.seed, ep, "A_ext_lambda_path_refit")
        pr = model.predict(nag)
        g, inside, names = prob_gradients([model], nag)
        viol = {f"wrong_way_share_{f}": float(np.mean(DIRECTIONS[f] * g[inside[:, names.index(f)], names.index(f)] < 0)) for f in EXP_DIRS}
        rows.append(dict(lambda_value=lam, rich_inner_cv_brier=cfg["score"], epochs=ep, **metrics(nag, pr, cfg["threshold"]), **viol))
    df = pd.DataFrame(rows)
    best = df.rich_inner_cv_brier.min()
    chosen = df[df.rich_inner_cv_brier <= best + 1e-9].lambda_value.min()
    df["selected_by_rich_inner_cv"] = df.lambda_value == chosen
    csv(out(st) / "E9_lambda_path_expanded.csv", df)
    return df


def lambda_path_contrasts(st):
    """E9b (addendum written after E9 was seen): paired NAGALA Brier difference of every lambda on the E9 path
    against lambda 0 of the same path; package bootstrap (G re-estimated per draw). Reuses the E9 checkpoints."""
    from stats import bootstrap
    _, m = frozen_models(st)
    base, ep = m["plans"]["M3"]["h"], m["plans"]["M3"]["epochs"]
    d, nag = st.data("RICH"), st.data("NAGALA")
    preds = {}
    for lam in LAMBDA_PATH:
        h = dict(base, directions=EXP_DIRS, **{"lambda": float(lam)})
        model, _ = st.fit("A", "ext_lambda_path", d, None, h, st.seed, ep, "A_ext_lambda_path_refit")
        preds[lam] = model.predict(nag)
    y, w, _ = weights(nag)
    rows = []
    for lam in LAMBDA_PATH[1:]:
        res = bootstrap(nag, preds[lam], preds[0.0], n_boot(st), st.seed + 30)
        rows.append(dict(comparison=f"lambda {lam:g} minus lambda 0 (six directions)", lambda_value=lam,
                         delta_brier=brier(y, w, preds[lam]) - brier(y, w, preds[0.0]), **res))
    csv(out(st) / "E9b_lambda_path_paired_brier.csv", rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E10: training-size curve
def size_curve(st, sizes=None, draws=None):
    _, m = frozen_models(st)
    base = {k: v for k, v in m["plans"]["M3"]["h"].items()}
    rich, nag = st.data("RICH"), st.data("NAGALA")
    yn, wn, _ = weights(nag)
    smoke = st.mode == "SMOKE"  # SMOKE samples 6,000 rows per cohort: software check only, tiny sizes
    sizes = sizes or ([1000, 2000] if smoke else SIZES)
    draws = draws or ([1] if smoke else DRAWS)
    rows = []
    for size in sizes:
        for draw in draws:
            path = st.private / "ext" / f"size_n{size}_d{draw}.pkl"
            if path.exists():
                rows += restore(path)
                continue
            idx, _ = train_test_split(np.arange(len(rich)), train_size=size, stratify=status(rich), random_state=draw)
            sub = rich.iloc[np.sort(idx)].reset_index(drop=True)
            label = f"A_ext_n{size}_d{draw}"
            arms = {"M1": [dict(base, **{"lambda": 0.0})],
                    "penalised_primary": [dict(base, directions=["fpg", "bmi"], **{"lambda": v}) for v in SIZE_LAMBDAS],
                    "penalised_expanded": [dict(base, directions=EXP_DIRS, **{"lambda": v}) for v in SIZE_LAMBDAS]}
            res = []
            for arm, grid in arms.items():
                cfg = st.tune("A", "ext_size_" + arm, sub, label, grid=grid)
                model, _ = st.fit("A", "ext_size_" + arm, sub, None, cfg["h"], st.seed, cfg["epochs"], label + "_refit")
                pr = model.predict(nag)
                res.append(dict(size=size, draw=draw, arm=arm, lambda_value=cfg["h"]["lambda"], epochs=cfg["epochs"],
                                cases_in_subsample=int((status(sub) == 1).sum()), nagala_brier=brier(yn, wn, pr),
                                nagala_auroc=metrics(nag, pr, cfg["threshold"])["auroc"],
                                nagala_cal_intercept=calibration(yn, wn, pr)[0]))
            for r in res:
                ref = next(x for x in res if x["arm"] == "M1")
                r["brier_minus_M1"] = r["nagala_brier"] - ref["nagala_brier"]
                r["auroc_minus_M1"] = r["nagala_auroc"] - ref["nagala_auroc"]
            path.parent.mkdir(parents=True, exist_ok=True)
            dump(path, res)
            rows += res
    df = pd.DataFrame(rows)
    csv(out(st) / "E10_training_size_draws.csv", df)
    summ = df.groupby(["size", "arm"]).agg(draws=("draw", "size"), mean_brier=("nagala_brier", "mean"),
                                           mean_brier_minus_M1=("brier_minus_M1", "mean"), sd_brier_minus_M1=("brier_minus_M1", "std"),
                                           draws_penalised_better=("brier_minus_M1", lambda v: int((v < 0).sum())),
                                           mean_auroc=("nagala_auroc", "mean"), mean_auroc_minus_M1=("auroc_minus_M1", "mean")).reset_index()
    csv(out(st) / "E10_training_size_summary.csv", summ)
    return df


# ---------------------------------------------------------------- E11: without sites lacking three-year support
def without_unsupported_sites(st):
    _, m = frozen_models(st)
    rich, nag = st.data("RICH"), st.data("NAGALA")
    sub = rich[~rich.site.isin([14, 16])].reset_index(drop=True)
    rows = []
    for n in ["M1", "M3"]:
        plan = m["plans"][n]
        model, _ = st.fit("A", "ext_nosite_" + n, sub, None, plan["h"], st.seed, plan["epochs"], "A_ext_nosite1416_refit")
        rows.append(dict(analysis="RICH without sites 14 and 16", model=n, rich_n=len(sub),
                         **metrics(nag, model.predict(nag), plan["threshold"])))
    csv(out(st) / "E11_without_sites_14_16.csv", rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- external subgroups (v2 stage 31 rules, external part)
def external_subgroups(st):
    d, p, m = st.results("A")
    rows = []
    for model in ["M1", "M3", "LR"]:
        cuts = {"all": np.ones(len(d), bool), "male": d.sex == 1, "female": d.sex == 0, "age_20_39": d.age < 40,
                "age_40_59": (d.age >= 40) & (d.age < 60), "age_60_plus": d.age >= 60, "FPG_below_5.6": d.fpg < 5.6,
                "FPG_5.6_to_6.0": d.fpg >= 5.6}
        for g, mask in cuts.items():
            part = d.loc[np.asarray(mask)].reset_index(drop=True)
            stt = status(part)
            row = dict(model=model, group=g, n=len(part), cases=int((stt == 1).sum()), controls=int((stt == 0).sum()), unknown=int((stt == -1).sum()))
            if row["cases"] < st.cfg["min_cases"] or row["controls"] < st.cfg["min_controls"]:
                row["status"] = "insufficient known cases or controls"
            else:
                row.update(status="estimated", **metrics(part, p[model][np.asarray(mask)], m["plans"][model]["threshold"]))
            rows.append(row)
    csv(out(st) / "EXT_external_subgroups.csv", rows)
    return pd.DataFrame(rows)


def run_all(st, include_new_fits=True):
    t = {}
    for name, fn in [("E2", penalty_diagnostics), ("E3-E5", external_comparisons), ("E6-E7", aligned_and_hdl),
                     ("E8", directions_all), ("subgroups", external_subgroups)] + (
                     [("E9", lambda_path), ("E9b", lambda_path_contrasts), ("E11", without_unsupported_sites),
                      ("E10", size_curve)] if include_new_fits else []):
        s = time.perf_counter()
        fn(st)
        t[name] = round(time.perf_counter() - s, 1)
        print(f"EXT {name} done in {t[name]} s", flush=True)
    csv(out(st) / "EXT_runtime.csv", [dict(analysis=k, seconds=v) for k, v in t.items()])
    return t
