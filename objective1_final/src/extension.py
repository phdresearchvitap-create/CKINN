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
LAMBDA_PATH = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
SIZES = [1000, 3000, 10000, 30000]
DRAWS = [1, 2, 3, 4, 5]
SIZE_LAMBDAS = [10.0, 1000.0, 100000.0]
THRESHOLDS_NB = [0.005, 0.01, 0.02, 0.03, 0.05]
STEPS = {"fpg": 0.1, "bmi": 1.0, "age": 1.0, "sbp": 5.0, "tg": 0.1, "hdl": 0.1}
NEURAL = ["M1", "M3", "fixed_zero", "fixed_positive", "expanded_prior"]


def out(st):
    p = st.table / "A" / "EXT"
    p.mkdir(parents=True, exist_ok=True)
    return p


def n_boot(st):
    return st.b["bootstrap"] if st.mode == "SMOKE" else 1000


def plans_from_selection(selection_csv):
    sel = pd.read_csv(selection_csv)
    plans = {}
    for r in sel.itertuples():
        h = json.loads(r.settings)
        neural = r.model not in CLASSIC
        plans[r.model] = dict(
            h=h,
            epochs=int(r.epochs) if neural else 1,
            score=float(r.source_cv_brier),
            threshold=float(r.threshold),
            trial=int(r.candidate_index),
        )
    return plans


def compare_external(st, reference_csv):
    mine = pd.read_csv(st.table / "A" / "external_metrics.csv").set_index("model")
    ref = pd.read_csv(reference_csv).set_index("model")
    rows = []
    for m in ref.index:
        if m not in mine.index:
            rows.append(dict(model=m, status="missing in this run"))
            continue
        db, da = (
            abs(mine.loc[m, "brier"] - ref.loc[m, "brier"]),
            abs(mine.loc[m, "auroc"] - ref.loc[m, "auroc"]),
        )
        rows.append(
            dict(
                model=m,
                brier_this=mine.loc[m, "brier"],
                brier_reference=ref.loc[m, "brier"],
                abs_diff_brier=db,
                auroc_this=mine.loc[m, "auroc"],
                auroc_reference=ref.loc[m, "auroc"],
                abs_diff_auroc=da,
                status="match" if db < 1e-06 and da < 1e-06 else "MISMATCH",
            )
        )
    csv(out(st) / "E1_reconstruction_check.csv", rows)
    return pd.DataFrame(rows)


def frozen_models(st):
    folder, m = st.frozen("A")
    models = {n: [restore(folder / f) for f in m["files"] if f.rsplit("_", 1)[0] == n] for n in m["plans"]}
    return (models, m)


def prob_gradients(models, d):
    f0 = models[0]
    X, inside = f0.prep.transform(d)
    g = np.zeros((len(d), len(f0.prep.names)))
    for f in models:
        f.obj.eval()
        for idx in np.array_split(np.arange(len(d)), max(1, len(d) // 20000)):
            x = torch.from_numpy(X[idx]).requires_grad_(True)
            g[idx] += (
                torch.autograd.grad(f.obj(x).sigmoid().sum(), x)[0][:, : len(f0.prep.names)].detach().numpy()
            )
    return (g / len(models), inside, f0.prep.names)


def weighted_bce(d, p):
    y, w, _ = weights(d)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))


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
                rows.append(
                    dict(
                        cohort=cohort,
                        model=n,
                        lambda_value=h["lambda"],
                        feature=f,
                        penalised=f in penalised,
                        eligible_pairs=int(ok.sum()),
                        wrong_way_share=float(np.mean(s * g[ok, j] < 0)),
                        median_abs_dp_dz=float(np.median(abs(g[ok, j]))),
                        mean_squared_wrong_way=float(term.mean()),
                    )
                )
            data_loss = weighted_bce(d, p)
            pen_value = float(np.concatenate(pen).mean()) if pen else 0.0
            rows.append(
                dict(
                    cohort=cohort,
                    model=n,
                    lambda_value=h["lambda"],
                    feature="ALL_PENALISED",
                    penalised=True,
                    weighted_data_loss=data_loss,
                    clinical_penalty=pen_value,
                    lambda_times_penalty=h["lambda"] * pen_value,
                    ratio_to_data_loss=h["lambda"] * pen_value / data_loss,
                )
            )
    csv(out(st) / "E2_penalty_diagnostics.csv", rows)
    return pd.DataFrame(rows)


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
            nb["treat_all", t] = (np.sum(w * y) - t / (1 - t) * np.sum(w * (1 - y))) / len(idx)
        return (b, nb)

    b0, nb0 = stats_of(y, w, np.arange(N))
    rng = np.random.default_rng(st.seed + 40)
    draws_b, draws_nb, failed = ([], [], 0)
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
    ok = len(DB) >= max(2, 0.8 * B)

    def ci(v):
        return tuple(np.quantile(v, [0.025, 0.975])) if ok else (np.nan, np.nan)

    rows = []
    for a in ["M1", "M3"]:
        for x in [n for n in ["reference", "LR", "RF", "XGB", "LGB", "SVM", "Cox"] if n in names]:
            lo, hi = ci(DB[a] - DB[x])
            rows.append(
                dict(
                    comparison=f"{a} minus {x}",
                    delta_brier=b0[a] - b0[x],
                    lo=lo,
                    hi=hi,
                    valid_boot=len(DB),
                    requested_boot=B,
                    failed_boot=failed,
                )
            )
    csv(out(st) / "E3_paired_brier_vs_baselines.csv", rows)
    rows = []
    for n in names:
        ipa = 1 - DB[n] / DB[ref]
        lo, hi = ci(ipa)
        rows.append(
            dict(
                model=n,
                brier=b0[n],
                brier_reference=b0[ref],
                ipa=1 - b0[n] / b0[ref],
                lo=lo,
                hi=hi,
                valid_boot=len(DB),
            )
        )
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


def _score_all(st, d, label):
    models, m = frozen_models(st)
    rows = []
    for n, fs in models.items():
        pr = probability(fs, d)
        rows.append(dict(analysis=label, model=n, **metrics(d, pr, m["plans"][n]["threshold"])))
    return (rows, models)


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
            step, eps = (STEPS[f], 0.005 * pp.scale[j])
            dd = d.loc[d[f].between(pp.lo[j] + eps, pp.hi[j] - max(step, eps))].copy()
            if not len(dd):
                continue
            center = probability(fs, dd)
            change = probability(fs, dd.assign(**{f: dd[f] + step})) - center
            s = DIRECTIONS[f]
            rows.append(
                dict(
                    model=n,
                    lambda_value=m["plans"][n]["h"]["lambda"],
                    feature=f,
                    penalised=f in m["plans"][n]["h"].get("directions", [])
                    and m["plans"][n]["h"]["lambda"] > 0,
                    raw_step=step,
                    n_eligible=len(dd),
                    violation_rate=float(np.mean(s * change < -1e-06)),
                    mean_wrong_way_change=float(np.maximum(0, -s * change).mean()),
                    mean_change=float(change.mean()),
                    meaningful_fraction=float(np.mean(abs(change) >= 0.001)),
                    flat_fraction=float(np.mean(abs(change) <= 1e-06)),
                )
            )
    csv(out(st) / "E8_directions_six.csv", rows)
    return pd.DataFrame(rows)


def lambda_path(st):
    _, m = frozen_models(st)
    base, ep = (m["plans"]["M3"]["h"], m["plans"]["M3"]["epochs"])
    d, nag = (st.data("RICH"), st.data("NAGALA"))
    rows = []
    for lam in LAMBDA_PATH:
        h = dict(base, directions=EXP_DIRS, **{"lambda": float(lam)})
        cfg = st.tune("A", "ext_lambda_path", d, "A_final", grid=[h], epochs=ep)
        model, _ = st.fit("A", "ext_lambda_path", d, None, h, st.seed, ep, "A_ext_lambda_path_refit")
        pr = model.predict(nag)
        g, inside, names = prob_gradients([model], nag)
        viol = {
            f"wrong_way_share_{f}": float(
                np.mean(DIRECTIONS[f] * g[inside[:, names.index(f)], names.index(f)] < 0)
            )
            for f in EXP_DIRS
        }
        rows.append(
            dict(
                lambda_value=lam,
                rich_inner_cv_brier=cfg["score"],
                epochs=ep,
                **metrics(nag, pr, cfg["threshold"]),
                **viol,
            )
        )
    df = pd.DataFrame(rows)
    best = df.rich_inner_cv_brier.min()
    chosen = df[df.rich_inner_cv_brier <= best + 1e-09].lambda_value.min()
    df["selected_by_rich_inner_cv"] = df.lambda_value == chosen
    csv(out(st) / "E9_lambda_path_expanded.csv", df)
    return df


def lambda_path_contrasts(st):
    from stats import bootstrap

    _, m = frozen_models(st)
    base, ep = (m["plans"]["M3"]["h"], m["plans"]["M3"]["epochs"])
    d, nag = (st.data("RICH"), st.data("NAGALA"))
    preds = {}
    for lam in LAMBDA_PATH:
        h = dict(base, directions=EXP_DIRS, **{"lambda": float(lam)})
        model, _ = st.fit("A", "ext_lambda_path", d, None, h, st.seed, ep, "A_ext_lambda_path_refit")
        preds[lam] = model.predict(nag)
    y, w, _ = weights(nag)
    rows = []
    for lam in LAMBDA_PATH[1:]:
        res = bootstrap(nag, preds[lam], preds[0.0], n_boot(st), st.seed + 30)
        rows.append(
            dict(
                comparison=f"lambda {lam:g} minus lambda 0 (six directions)",
                lambda_value=lam,
                delta_brier=brier(y, w, preds[lam]) - brier(y, w, preds[0.0]),
                **res,
            )
        )
    csv(out(st) / "E9b_lambda_path_paired_brier.csv", rows)
    return pd.DataFrame(rows)


def size_curve(st, sizes=None, draws=None):
    _, m = frozen_models(st)
    base = {k: v for k, v in m["plans"]["M3"]["h"].items()}
    rich, nag = (st.data("RICH"), st.data("NAGALA"))
    yn, wn, _ = weights(nag)
    smoke = st.mode == "SMOKE"
    sizes = sizes or ([1000, 2000] if smoke else SIZES)
    draws = draws or ([1] if smoke else DRAWS)
    rows = []
    for size in sizes:
        for draw in draws:
            path = st.private / "ext" / f"size_n{size}_d{draw}.pkl"
            if path.exists():
                rows += restore(path)
                continue
            idx, _ = train_test_split(
                np.arange(len(rich)), train_size=size, stratify=status(rich), random_state=draw
            )
            sub = rich.iloc[np.sort(idx)].reset_index(drop=True)
            label = f"A_ext_n{size}_d{draw}"
            arms = {
                "M1": [dict(base, **{"lambda": 0.0})],
                "penalised_primary": [
                    dict(base, directions=["fpg", "bmi"], **{"lambda": v}) for v in SIZE_LAMBDAS
                ],
                "penalised_expanded": [
                    dict(base, directions=EXP_DIRS, **{"lambda": v}) for v in SIZE_LAMBDAS
                ],
            }
            res = []
            for arm, grid in arms.items():
                cfg = st.tune("A", "ext_size_" + arm, sub, label, grid=grid)
                model, _ = st.fit(
                    "A", "ext_size_" + arm, sub, None, cfg["h"], st.seed, cfg["epochs"], label + "_refit"
                )
                pr = model.predict(nag)
                res.append(
                    dict(
                        size=size,
                        draw=draw,
                        arm=arm,
                        lambda_value=cfg["h"]["lambda"],
                        epochs=cfg["epochs"],
                        cases_in_subsample=int((status(sub) == 1).sum()),
                        nagala_brier=brier(yn, wn, pr),
                        nagala_auroc=metrics(nag, pr, cfg["threshold"])["auroc"],
                        nagala_cal_intercept=calibration(yn, wn, pr)[0],
                    )
                )
            for r in res:
                ref = next((x for x in res if x["arm"] == "M1"))
                r["brier_minus_M1"] = r["nagala_brier"] - ref["nagala_brier"]
                r["auroc_minus_M1"] = r["nagala_auroc"] - ref["nagala_auroc"]
            path.parent.mkdir(parents=True, exist_ok=True)
            dump(path, res)
            rows += res
    df = pd.DataFrame(rows)
    csv(out(st) / "E10_training_size_draws.csv", df)
    summ = (
        df.groupby(["size", "arm"])
        .agg(
            draws=("draw", "size"),
            mean_brier=("nagala_brier", "mean"),
            mean_brier_minus_M1=("brier_minus_M1", "mean"),
            sd_brier_minus_M1=("brier_minus_M1", "std"),
            draws_penalised_better=("brier_minus_M1", lambda v: int((v < 0).sum())),
            mean_auroc=("nagala_auroc", "mean"),
            mean_auroc_minus_M1=("auroc_minus_M1", "mean"),
        )
        .reset_index()
    )
    csv(out(st) / "E10_training_size_summary.csv", summ)
    return df


def without_unsupported_sites(st):
    _, m = frozen_models(st)
    rich, nag = (st.data("RICH"), st.data("NAGALA"))
    sub = rich[~rich.site.isin([14, 16])].reset_index(drop=True)
    rows = []
    for n in ["M1", "M3"]:
        plan = m["plans"][n]
        model, _ = st.fit(
            "A", "ext_nosite_" + n, sub, None, plan["h"], st.seed, plan["epochs"], "A_ext_nosite1416_refit"
        )
        rows.append(
            dict(
                analysis="RICH without sites 14 and 16",
                model=n,
                rich_n=len(sub),
                **metrics(nag, model.predict(nag), plan["threshold"]),
            )
        )
    csv(out(st) / "E11_without_sites_14_16.csv", rows)
    return pd.DataFrame(rows)


def external_subgroups(st):
    d, p, m = st.results("A")
    rows = []
    for model in ["M1", "M3", "LR"]:
        cuts = {
            "all": np.ones(len(d), bool),
            "male": d.sex == 1,
            "female": d.sex == 0,
            "age_20_39": d.age < 40,
            "age_40_59": (d.age >= 40) & (d.age < 60),
            "age_60_plus": d.age >= 60,
            "FPG_below_5.6": d.fpg < 5.6,
            "FPG_5.6_to_6.0": d.fpg >= 5.6,
        }
        for g, mask in cuts.items():
            part = d.loc[np.asarray(mask)].reset_index(drop=True)
            stt = status(part)
            row = dict(
                model=model,
                group=g,
                n=len(part),
                cases=int((stt == 1).sum()),
                controls=int((stt == 0).sum()),
                unknown=int((stt == -1).sum()),
            )
            if row["cases"] < st.cfg["min_cases"] or row["controls"] < st.cfg["min_controls"]:
                row["status"] = "insufficient known cases or controls"
            else:
                row.update(
                    status="estimated",
                    **metrics(part, p[model][np.asarray(mask)], m["plans"][model]["threshold"]),
                )
            rows.append(row)
    csv(out(st) / "EXT_external_subgroups.csv", rows)
    return pd.DataFrame(rows)


def calibration_contrast(st):
    from stats import calbins

    d, p, m = st.results("A")

    def cal(y, w, pr):
        ci, cs = calibration(y, w, pr)
        cb = calbins(y, w, pr)
        return (ci, cs, float(np.average(abs(cb.predicted - cb.observed), weights=cb.n)))

    y, w, _ = weights(d)
    point = {n: cal(y, w, p[n]) for n in ("M1", "M3")}
    rng = np.random.default_rng(st.seed + 41)
    draws, failed, B = ([], 0, n_boot(st))
    for _ in range(B):
        i = rng.integers(len(d), size=len(d))
        dd = d.iloc[i].reset_index(drop=True)
        try:
            yy, ww, _ = weights(dd)
            if np.unique(yy[ww > 0]).size < 2:
                raise ValueError("single class")
            a = {n: cal(yy, ww, p[n][i]) for n in ("M1", "M3")}
            draws.append(
                [
                    abs(a["M3"][0]) - abs(a["M1"][0]),
                    abs(a["M3"][1] - 1) - abs(a["M1"][1] - 1),
                    a["M3"][2] - a["M1"][2],
                ]
            )
        except (ValueError, FloatingPointError):
            failed += 1
    D = np.array(draws, dtype=float)
    labels = ["|calibration intercept|", "|calibration slope - 1|", "ECE"]
    pts = [
        (abs(point["M1"][0]), abs(point["M3"][0])),
        (abs(point["M1"][1] - 1), abs(point["M3"][1] - 1)),
        (point["M1"][2], point["M3"][2]),
    ]
    rows = []
    for k, (lab, (a1, a3)) in enumerate(zip(labels, pts)):
        v = D[:, k][np.isfinite(D[:, k])] if len(D) else np.array([])
        lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) >= max(2, 0.8 * B) else (np.nan, np.nan)
        rows.append(
            dict(
                metric=lab,
                M1=a1,
                M3=a3,
                M3_minus_M1=a3 - a1,
                lo=lo,
                hi=hi,
                valid_boot=len(v),
                requested_boot=B,
                failed_boot=failed,
            )
        )
    csv(out(st) / "E12_calibration_contrast.csv", rows)
    return pd.DataFrame(rows)


def _read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if Path(path).exists() else None


def _f(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3e}" if v != 0 and abs(v) < 0.001 else f"{v:.{d}f}"


def proposal_evidence(st):
    t, e = (st.table / "A", out(st))
    rows = []

    def add(item, question, artifact, estimate, decision="descriptive"):
        rows.append(
            dict(
                item=item, proposal_question=question, artifact=artifact, estimate=estimate, decision=decision
            )
        )

    ex, oof = (_read(t / "external_metrics.csv"), _read(t / "pooled_oof.csv"))
    if ex is not None and oof is not None:
        g = lambda df, n, k: df.set_index("model").loc[n, k]
        add(
            "RQ1",
            "Can a neural network predict future type 2 diabetes from baseline variables?",
            "pooled_oof.csv; external_metrics.csv",
            f"M1 AUROC RICH OOF {_f(g(oof, 'M1', 'auroc'))}, NAGALA {_f(g(ex, 'M1', 'auroc'))}; NAGALA Brier {_f(g(ex, 'M1', 'brier'))} (reference {_f(g(ex, 'reference', 'brier'))}); calibration intercept {_f(g(ex, 'M1', 'cal_intercept'))}, slope {_f(g(ex, 'M1', 'cal_slope'))}",
        )
    pb = _read(t / "paired_brier.csv")
    if pb is not None:
        r = pb.iloc[0]
        dec = "supported" if r.delta_hi < 0 else "contradicted" if r.delta_lo > 0 else "inconclusive"
        add(
            "RQ2 / H1",
            "Does clinical knowledge improve prediction over an unconstrained network (CKINN > MLP)?",
            "paired_brier.csv",
            f"Brier(M3) - Brier(M1) = {_f(r.delta)} (95% {_f(r.delta_lo)} to {_f(r.delta_hi)}); M3 lambda {_f(r.lambda_candidate, 0)}",
            f"{dec} (pre-specified primary contrast)",
        )
    mp = _read(t / "matched_positive_paired_brier.csv")
    if mp is not None:
        r = mp.iloc[0]
        add(
            "RQ2 / H1 (secondary)",
            "Matched positive versus zero penalty, same architecture, seeds and epochs",
            "matched_positive_paired_brier.csv",
            f"Brier difference {_f(r.delta)} (95% {_f(r.delta_lo)} to {_f(r.delta_hi)}); lambda {_f(r.lambda_candidate, 0)}",
        )
    site = _read(t / "A_RICH_site.csv")
    if pb is not None:
        txt = f"NAGALA paired Brier difference {_f(pb.iloc[0].delta)}"
        if site is None:
            txt += "; held-out-site analysis not run"
        else:
            done = site[site.status == "complete"] if "brier" in site else site.iloc[0:0]
            s = (
                done.pivot_table(index="site", columns="model", values="brier")
                if len(done)
                else pd.DataFrame()
            )
            if {"M1", "M3"} <= set(s.columns):
                dd = (s["M3"] - s["M1"]).dropna()
                txt += f"; held-out RICH sites estimable {len(dd)} of {site.site.nunique()}, M3 lower in {int((dd < 0).sum())}, median difference {_f(dd.median())}"
            else:
                txt += f"; no held-out RICH site met the support rule ({site.site.nunique()} sites examined)"
        e9b = _read(e / "E9b_lambda_path_paired_brier.csv")
        if e9b is not None:
            r = e9b.loc[e9b.delta_brier.idxmin()]
            txt += f"; exploratory lambda path (six directions): largest NAGALA reduction {_f(r.delta_brier)} (95% {_f(r.delta_lo)} to {_f(r.delta_hi)}) at lambda {_f(r.lambda_value, 0)}"
        add(
            "RQ3 / H3",
            "Does clinical knowledge improve cross-population generalisation?",
            "paired_brier.csv; A_RICH_site.csv; EXT/E9b",
            txt,
        )
    e12 = _read(e / "E12_calibration_contrast.csv")
    if e12 is not None:
        add(
            "RQ4 / H2",
            "Do clinical constraints improve (maintain) external calibration?",
            "EXT/E12_calibration_contrast.csv",
            "; ".join(
                (
                    f"{r.metric}: M1 {_f(r.M1)}, M3 {_f(r.M3)}, difference {_f(r.M3_minus_M1)} (95% {_f(r.lo)} to {_f(r.hi)})"
                    for r in e12.itertuples()
                )
            ),
        )
    di, e8 = (_read(t / "directions.csv"), _read(e / "E8_directions_six.csv"))
    if di is not None:
        v = di[di.deployment == "frozen_average"].pivot_table(
            index="model", columns="feature", values="violation_rate"
        )
        txt = "; ".join(
            (
                f"{m} wrong-way share FPG {_f(v.loc[m, 'fpg'], 3)}, BMI {_f(v.loc[m, 'bmi'], 3)}"
                for m in ["M1", "M3"]
                if m in v.index
            )
        )
        if e8 is not None:
            w = e8.pivot_table(index="model", columns="feature", values="violation_rate")
            txt += "; " + "; ".join(
                (
                    f"{m} HDL {_f(w.loc[m, 'hdl'], 3)}, SBP {_f(w.loc[m, 'sbp'], 3)}"
                    for m in ["M1", "M3"]
                    if m in w.index
                )
            )
        add(
            "RQ5 / H4",
            "Are CKINN explanations more clinically consistent than an unconstrained network?",
            "directions.csv; attributions.csv; EXT/E8",
            txt,
        )
    sg = _read(t / "sex_gaps.csv")
    if sg is not None:
        parts = []
        for r in sg.itertuples():
            gap = getattr(r, "sensitivity_gap", np.nan)
            parts.append(
                f"{r.model} {r.stage}: {r.status}"
                + (f", sensitivity gap {_f(gap, 3)}" if r.status.startswith("descriptive") else "")
            )
        add(
            "RQ6 / H5",
            "Does the model give more consistent performance across demographic groups?",
            "sex_gaps.csv; subgroups.csv",
            "; ".join(parts),
        )
    sub = _read(t / "subgroups.csv")
    if sub is not None:
        s = sub[sub.group == "FPG_below_5.6"]
        add(
            "RQ7",
            "Can the model identify elevated risk before diagnostic thresholds are reached?",
            "subgroups.csv (FPG < 5.6 mmol/L)",
            "; ".join(
                (
                    f"{r.model} {r.stage}: n {int(r.n)}, cases {int(r.events)}, "
                    + (f"AUROC {_f(r.auroc, 3)}" if r.status == "estimated" else r.status)
                    for r in s.itertuples()
                )
            ),
        )
    for item, q in [
        (
            "Section 3C",
            "Progression from normal through prediabetes to diabetes (repeated dated states required)",
        ),
        ("Section 11", "Latent metabolic state or physiological equations (glucose-insulin data required)"),
        ("Section 24", "Clinical expert agreement with explanations (clinician panel required)"),
        ("Section 26", "Clinical knowledge graph (curated evidence graph required)"),
        ("Section 21", "Clinical decision support and patient benefit (prospective evaluation required)"),
    ]:
        add(item, q, "", "not measurable with these data", "not measurable")
    csv(e / "proposal_hypotheses_evidence.csv", rows)
    return pd.DataFrame(rows)


def run_all(st, include_new_fits=True):
    t = {}
    for name, fn in [
        ("E2", penalty_diagnostics),
        ("E3-E5", external_comparisons),
        ("E6-E7", aligned_and_hdl),
        ("E8", directions_all),
        ("E12", calibration_contrast),
        ("subgroups", external_subgroups),
    ] + (
        [
            ("E9", lambda_path),
            ("E9b", lambda_path_contrasts),
            ("E11", without_unsupported_sites),
            ("E10", size_curve),
        ]
        if include_new_fits
        else []
    ):
        s = time.perf_counter()
        fn(st)
        t[name] = round(time.perf_counter() - s, 1)
        print(f"EXT {name} done in {t[name]} s", flush=True)
    csv(out(st) / "EXT_runtime.csv", [dict(analysis=k, seconds=v) for k, v in t.items()])
    return t
