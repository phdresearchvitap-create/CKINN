from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from data import CORE, csv, dump, restore, status, js, atomic, digest
from stats import weights, metrics, brier, calibration, calbins, point_threshold, bootstrap
from models import DIRECTIONS
from analysis import probability, oof
from runner import CLASSIC

PATH_NAME = "CKINN_path"


def table_dir(st, task):
    p = st.table / st.tag(task) / "v4"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cached(st, task, name):
    folder = st.private / "frozen" / st.tag(task)
    key = digest(folder / "manifest.json") if (folder / "manifest.json").exists() else ""
    marker = table_dir(st, task) / f".{name}.done"
    return (marker, key, marker.exists() and marker.read_text() == key)


def mark(marker, mkey):
    marker.write_text(mkey)


def ck_names(task):
    return ["CKINN", "CKINN_FPG_BMI", "M4", "M5"] if task in ("A", "R") else ["CKINN"]


def directions_for(st, task, name):
    c = st.cfg["ckinn"]
    if task == "B":
        return list(c["directions_B"])
    return list(c["directions_initial"] if name == "CKINN_FPG_BMI" else c["directions_A"])


def one_se_choice(losses):
    lams = sorted(losses)
    means = {l: float(np.mean(losses[l])) for l in lams}
    best = min(lams, key=lambda l: (means[l], l))
    rows = []
    for l in lams:
        diff = np.asarray(losses[l]) - np.asarray(losses[best])
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        rows.append(
            dict(
                lambda_value=l,
                inner_cv_brier=means[l],
                excess_over_best=float(diff.mean()),
                paired_se=se,
                within_one_se=bool(diff.mean() <= se + 1e-15),
            )
        )
    chosen = max((r["lambda_value"] for r in rows if r["within_one_se"]))
    for r in rows:
        r["selected"] = r["lambda_value"] == chosen
    return (chosen, rows)


def path_h(base, directions, lam):
    return dict(base, directions=list(directions) if lam else [], **{"lambda": float(lam)})


def select_lambda(st, task, d, label, base, epochs, directions):
    split = st.split(d, st.b["inner"], label + "_inner", st.seed + 11)
    losses, preds = ({}, {})
    Y, W = (np.zeros(len(d)), np.zeros(len(d)))
    for lam in st.cfg["ckinn"]["lambda"]:
        h = path_h(base, directions, lam)
        p = np.full(len(d), np.nan)
        loss = np.zeros(len(d))
        for i, (a, b) in enumerate(split):
            tr, va = (d.iloc[a], d.iloc[b])
            model, _ = st.fit(task, PATH_NAME, tr, None, h, st.seed, epochs, f"{label}_lam_i{i}")
            p[b] = model.predict(va)
            y, w, _ = weights(va)
            loss[b] = w * (y - p[b]) ** 2
            Y[b], W[b] = (y, w)
        losses[float(lam)] = loss
        preds[float(lam)] = p
    chosen, path = one_se_choice(losses)
    return dict(
        h=path_h(base, directions, chosen),
        epochs=int(epochs),
        score=float(losses[chosen].mean()),
        threshold=point_threshold(Y, W, preds[chosen]),
        trial=-1,
        path=path,
    )


def ck_plans(st, task, d, label, m1):
    base = {k: v for k, v in m1["h"].items() if k not in ("directions", "lambda")}
    ep = int(m1["epochs"])
    plans = {}
    for n in ["CKINN", "CKINN_FPG_BMI"] if task in ("A", "R") else ["CKINN"]:
        plans[n] = select_lambda(st, task, d, label, base, ep, directions_for(st, task, n))
    if task in ("A", "R"):
        for n, extra in [
            ("M4", dict(site_balance=True)),
            ("M5", dict(site_balance=True, group_penalty=st.cfg["ckinn"]["group_penalty"])),
        ]:
            h = dict(plans["CKINN"]["h"], **extra)
            res = st.tune(task, n, d, label, fixed=h, epochs=ep)
            plans[n] = dict(res, epochs=ep)
    return plans


def nested_fold(st, task, i):
    src = st.sources(task)[0]
    d = st.data(src)
    label = st.tag(task) + "_" + src + "_nested"
    splits = st.split(d, st.b["outer"], label, st.seed)
    a, b = splits[i]
    tr, va = (d.iloc[a], d.iloc[b])
    lab = label + f"_o{i}"
    out = st.private / "oof_v4" / f"{lab}.pkl"
    if out.exists():
        return restore(out)
    plans = st.plan(task, st.model_names(task), tr, lab, False)
    ck = ck_plans(st, task, tr, lab, plans["M1"])
    rows, paths = ([], [])
    for n, plan in ck.items():
        model, _ = st.fit(task, n, tr, None, plan["h"], st.seed, plan["epochs"], lab + "_refit")
        p = model.predict(va)
        dump(
            st.private / "predictions" / lab / (n + ".pkl"),
            dict(id=va.id.tolist(), p=p, threshold=plan["threshold"]),
        )
        rows.append(
            dict(
                fold=i,
                model=n,
                lambda_value=plan["h"]["lambda"],
                status="complete",
                **metrics(va, p, plan["threshold"]),
            )
        )
        paths += [dict(fold=i, model=n, **r) for r in plan.get("path", [])]
    res = dict(rows=rows, paths=paths)
    dump(out, res)
    return res


def nested(st, task):
    st.internal(task)
    rows, paths = ([], [])
    for i in range(st.b["outer"]):
        r = nested_fold(st, task, i)
        rows += r["rows"]
        paths += r["paths"]
    t = table_dir(st, task)
    csv(t / "nested_ckinn_folds.csv", rows)
    csv(t / "lambda_paths_nested.csv", paths)
    summary = []
    for n in st.model_names(task) + ck_names(task):
        od = oof(st, task, n)
        summary.append(dict(model=n, **metrics(od, od.p.to_numpy(), od.threshold.to_numpy())))
    csv(t / "internal_pooled_oof.csv", summary)
    return pd.DataFrame(summary)


def final_plans(st, task):
    plans = st.final_plan(task)
    if all((n in plans for n in ck_names(task))):
        return plans
    d = st.data(st.sources(task)[0])
    plans.update(ck_plans(st, task, d, st.tag(task) + "_final", plans["M1"]))
    dump(st.private / "plans" / (st.tag(task) + "_final.pkl"), plans)
    csv(
        table_dir(st, task) / "lambda_path_final.csv",
        [dict(model=n, **r) for n in ck_names(task) for r in plans[n].get("path", [])],
    )
    return plans


def _stats(y, w, p):
    k = w > 0
    two = np.unique(y[k]).size == 2
    return dict(
        brier=float(np.mean(w * (y - p) ** 2)),
        auroc=roc_auc_score(y[k], p[k], sample_weight=w[k]) if two else np.nan,
        auprc=average_precision_score(y[k], p[k], sample_weight=w[k]) if two else np.nan,
    )


def _cal(y, w, p):
    ci, cs = calibration(y, w, p)
    cb = calbins(y, w, p)
    return dict(
        abs_cal_intercept=abs(ci),
        abs_slope_minus_1=abs(cs - 1) if np.isfinite(cs) else np.nan,
        ece=float(np.average(abs(cb.predicted - cb.observed), weights=cb.n)),
    )


def _nb(y, w, p, t):
    s = p >= t
    return (np.sum(w * y * s) - t / (1 - t) * np.sum(w * (1 - y) * s)) / len(y)


def _ci(v, B):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return tuple(np.quantile(v, [0.025, 0.975])) if len(v) >= max(2, 0.8 * B) else (np.nan, np.nan)


def external_report(st, task):
    marker, mkey, done = cached(st, task, "external_report")
    if done:
        return read(table_dir(st, task) / "external_metrics_ci.csv")
    d, p, m = st.results(task)
    plans = m["plans"]
    names = list(plans)
    B = st.b["bootstrap"]
    t = table_dir(st, task)
    rows = []
    for n in names:
        r = dict(model=n, **metrics(d, p[n], plans[n]["threshold"]))
        r.update(bootstrap(d, p[n], None, B, st.seed + 50, plans[n]["threshold"], True))
        rows.append(r)
    csv(t / "external_metrics_ci.csv", rows)
    y, w, _ = weights(d)
    thresholds = st.cfg["validation"]["net_benefit_thresholds"]
    nb_models = [n for n in ["M1", "CKINN", "LR", "XGB"] if n in p]
    cal_models = [n for n in ["M1", "CKINN", "LR", "XGB"] if n in p]
    edges = {n: np.unique(np.quantile(p[n], np.linspace(0, 1, 11)[1:-1])) for n in cal_models}

    def bins(y_, w_, pr, e):
        lab = np.searchsorted(e, pr, side="left")
        return [
            float(np.average(y_[lab == k], weights=w_[lab == k])) if w_[lab == k].sum() > 0 else np.nan
            for k in range(len(e) + 1)
        ]

    def draw_stats(y_, w_, idx):
        s = {n: _stats(y_, w_, p[n][idx]) for n in names}
        c = {n: _cal(y_, w_, p[n][idx]) for n in ["M1", "CKINN"] if n in p}
        nb = {(n, th): _nb(y_, w_, p[n][idx], th) for n in nb_models for th in thresholds}
        for th in thresholds:
            nb["treat_all", th] = _nb(y_, w_, np.ones(len(idx)), th)
        cb = {n: bins(y_, w_, p[n][idx], edges[n]) for n in cal_models}
        return (s, c, nb, cb)

    point = draw_stats(y, w, np.arange(len(d)))
    rng = np.random.default_rng(st.seed + 51)
    draws, failed = ([], 0)
    for _ in range(B):
        i = rng.integers(len(d), size=len(d))
        dd = d.iloc[i].reset_index(drop=True)
        try:
            yy, ww, _ = weights(dd)
            if np.unique(yy[ww > 0]).size < 2:
                raise ValueError("single class")
            draws.append(draw_stats(yy, ww, i))
        except (ValueError, FloatingPointError):
            failed += 1
    out = []
    if "CKINN" in p:
        for key in ["brier", "auroc", "auprc"]:
            a, b0 = (point[0]["CKINN"][key], point[0]["M1"][key])
            lo, hi = _ci([x[0]["CKINN"][key] - x[0]["M1"][key] for x in draws], B)
            out.append(
                dict(
                    metric=key,
                    CKINN=a,
                    M1=b0,
                    CKINN_minus_M1=a - b0,
                    lo=lo,
                    hi=hi,
                    valid_boot=len(draws),
                    failed_boot=failed,
                )
            )
        for key in ["abs_cal_intercept", "abs_slope_minus_1", "ece"]:
            a, b0 = (point[1]["CKINN"][key], point[1]["M1"][key])
            lo, hi = _ci([x[1]["CKINN"][key] - x[1]["M1"][key] for x in draws], B)
            out.append(
                dict(
                    metric=key,
                    CKINN=a,
                    M1=b0,
                    CKINN_minus_M1=a - b0,
                    lo=lo,
                    hi=hi,
                    valid_boot=len(draws),
                    failed_boot=failed,
                )
            )
        r0 = out[0]
        decision = "supported" if r0["hi"] < 0 else "contradicted" if r0["lo"] > 0 else "inconclusive"
        for r in out:
            r["h1_decision"] = decision if r["metric"] == "brier" and task == "A" else ""
        csv(t / "contrast_ckinn_vs_m1.csv", out)
        pairs = []
        for x_name in [n for n in names if n != "CKINN"]:
            for key in ["brier", "auroc"]:
                diff = point[0]["CKINN"][key] - point[0][x_name][key]
                lo, hi = _ci([z[0]["CKINN"][key] - z[0][x_name][key] for z in draws], B)
                pairs.append(
                    dict(comparison=f"CKINN minus {x_name}", metric=key, estimate=diff, lo=lo, hi=hi)
                )
        csv(t / "contrast_ckinn_vs_all.csv", pairs)
    ref = "reference"
    ipa = []
    for n in names:
        val = 1 - point[0][n]["brier"] / point[0][ref]["brier"]
        lo, hi = _ci([1 - x[0][n]["brier"] / x[0][ref]["brier"] for x in draws], B)
        ipa.append(dict(model=n, brier=point[0][n]["brier"], ipa=val, lo=lo, hi=hi))
    csv(t / "ipa.csv", ipa)
    nb = []
    for key, val in point[2].items():
        lo, hi = _ci([x[2][key] for x in draws], B)
        nb.append(dict(model=key[0], threshold=key[1], net_benefit=val, lo=lo, hi=hi))
    for th in thresholds:
        nb.append(dict(model="treat_none", threshold=th, net_benefit=0.0, lo=0.0, hi=0.0))
    csv(t / "net_benefit.csv", nb)
    cal = []
    for n in cal_models:
        e = edges[n]
        lab = np.searchsorted(e, p[n], side="left")
        for k in range(len(e) + 1):
            pick = lab == k
            lo, hi = _ci([x[3][n][k] for x in draws], B)
            cal.append(
                dict(
                    model=n,
                    bin=k,
                    n=int(pick.sum()),
                    predicted=float(p[n][pick].mean()) if pick.any() else np.nan,
                    observed=point[3][n][k],
                    observed_lo=lo,
                    observed_hi=hi,
                )
            )
    csv(t / "calibration_deciles_ci.csv", cal)
    mark(marker, mkey)
    return pd.DataFrame(rows)


def dersimonian_laird(y, v):
    y, v = (np.asarray(y, float), np.asarray(v, float))
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = (y[ok], v[ok])
    k = len(y)
    if k == 0:
        return dict(k=0, pooled=np.nan, lo=np.nan, hi=np.nan, tau2=np.nan, i2=np.nan)
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - fixed) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (v + tau2)
    pooled = float(np.sum(ws * y) / np.sum(ws))
    se = float(np.sqrt(1 / np.sum(ws)))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    return dict(
        k=k, pooled=pooled, lo=pooled - 1.959964 * se, hi=pooled + 1.959964 * se, tau2=tau2, i2=i2, q=q
    )


def loso(st):
    marker, mkey, done = cached(st, "A", "loso")
    if done:
        return read(table_dir(st, "A") / "loso_pooled.csv")
    _, m = st.frozen("A")
    plans = m["plans"]
    d = st.data("RICH")
    models = [n for n in st.cfg["validation"]["loso_models"] if n in plans]
    B = st.b["bootstrap"]
    rows, diffs = ([], [])
    for s in sorted(d.site.unique(), key=str):
        te = d[d.site == s].reset_index(drop=True)
        tr = d[d.site != s].reset_index(drop=True)
        stt = status(te)
        cases, controls = (int((stt == 1).sum()), int((stt == 0).sum()))
        if cases < st.cfg["min_cases"] or controls < st.cfg["min_controls"]:
            rows.append(
                dict(
                    site=str(s),
                    model="all",
                    status="insufficient support",
                    n=len(te),
                    events=cases,
                    controls=controls,
                )
            )
            continue
        pr = {}
        for n in models:
            plan = plans[n]
            try:
                model, _ = st.fit("A", n, tr, None, plan["h"], st.seed, plan["epochs"], f"A_loso_site{s}")
            except ValueError as exc:
                rows.append(
                    dict(
                        site=str(s),
                        model=n,
                        status=f"fit failed: {exc}",
                        n=len(te),
                        events=cases,
                        controls=controls,
                    )
                )
                continue
            pr[n] = model.predict(te)
            rows.append(
                dict(site=str(s), model=n, status="complete", **metrics(te, pr[n], plan["threshold"]))
            )
        if "CKINN" in pr and "M1" in pr:
            y, w, _ = weights(te)
            p0 = {n: _stats(y, w, pr[n]) for n in ("CKINN", "M1")}
            rng = np.random.default_rng(st.seed + 60)
            dr = []
            for _ in range(B):
                i = rng.integers(len(te), size=len(te))
                dd = te.iloc[i].reset_index(drop=True)
                try:
                    yy, ww, _ = weights(dd)
                    if np.unique(yy[ww > 0]).size < 2:
                        raise ValueError("single class")
                    a, b0 = (_stats(yy, ww, pr["CKINN"][i]), _stats(yy, ww, pr["M1"][i]))
                    dr.append((a["brier"] - b0["brier"], a["auroc"] - b0["auroc"]))
                except (ValueError, FloatingPointError):
                    pass
            dr = np.array(dr, float)
            for j, key in enumerate(["brier", "auroc"]):
                est = p0["CKINN"][key] - p0["M1"][key]
                col = dr[:, j] if len(dr) else np.array([])
                lo, hi = _ci(col, B)
                diffs.append(
                    dict(
                        site=str(s),
                        metric=key,
                        CKINN_minus_M1=est,
                        lo=lo,
                        hi=hi,
                        se=float(np.nanstd(col, ddof=1)) if len(col) > 1 else np.nan,
                        n=len(te),
                        events=cases,
                        valid_boot=len(col),
                    )
                )
    t = table_dir(st, "A")
    csv(t / "loso_sites.csv", rows)
    csv(t / "loso_ckinn_vs_m1.csv", diffs)
    pooled = []
    D = pd.DataFrame(diffs)
    for key in ["brier", "auroc"]:
        if len(D):
            g = D[D.metric == key]
            pooled.append(
                dict(
                    metric=key,
                    **dersimonian_laird(g.CKINN_minus_M1, g.se**2),
                    sites_ckinn_better=int(
                        (g.CKINN_minus_M1 < 0 if key == "brier" else g.CKINN_minus_M1 > 0).sum()
                    ),
                )
            )
    csv(t / "loso_pooled.csv", pooled)
    mark(marker, mkey)
    return pd.DataFrame(pooled)


def sensitivity(st):
    marker, mkey, done = cached(st, "A", "sensitivity")
    if done:
        return read(table_dir(st, "A") / "sensitivity.csv")
    _, m = st.frozen("A")
    plans = m["plans"]
    nag = st.data("NAGALA")
    rows = []
    for source in ["RICH_broad", "RICH_alt0"]:
        d = st.data(source)
        pr = {}
        for n in [x for x in st.cfg["validation"]["loso_models"] if x in plans]:
            plan = plans[n]
            model, _ = st.fit("A", n, d, None, plan["h"], st.seed, plan["epochs"], f"A_sens_{source}")
            pr[n] = model.predict(nag)
            rows.append(
                dict(analysis=source, model=n, development_n=len(d), **metrics(nag, pr[n], plan["threshold"]))
            )
        if "CKINN" in pr:
            r = bootstrap(nag, pr["CKINN"], pr["M1"], st.b["bootstrap"], st.seed + 70)
            y, w, _ = weights(nag)
            rows.append(
                dict(
                    analysis=source,
                    model="CKINN minus M1 (Brier)",
                    brier=brier(y, w, pr["CKINN"]) - brier(y, w, pr["M1"]),
                    brier_lo=r["delta_lo"],
                    brier_hi=r["delta_hi"],
                )
            )
    aligned = nag[nag.time >= 2].reset_index(drop=True)
    nohdl = nag.assign(hdl=np.nan)
    folder, _ = st.frozen("A")
    for label, frame in [("NAGALA follow-up >= 2 years", aligned), ("NAGALA HDL treated as missing", nohdl)]:
        for n in [x for x in ["M1", "CKINN", "LR", "XGB"] if x in plans]:
            fs = [restore(folder / f) for f in m["files"] if f.rsplit("_", 1)[0] == n]
            pr = probability(fs, frame)
            rows.append(dict(analysis=label, model=n, **metrics(frame, pr, plans[n]["threshold"])))
    csv(table_dir(st, "A") / "sensitivity.csv", rows)
    mark(marker, mkey)
    return pd.DataFrame(rows)


def _groups(dd, task):
    g = {
        "male": dd.sex == 1,
        "female": dd.sex == 0,
        "age_20_39": dd.age < 40,
        "age_40_59": (dd.age >= 40) & (dd.age < 60),
        "age_60_plus": dd.age >= 60,
    }
    if task in ("A", "R"):
        g.update({"FPG_below_5.6": dd.fpg < 5.6, "FPG_5.6_to_6.0": dd.fpg >= 5.6})
    return g


def fairness(st, task="A"):
    marker, mkey, done = cached(st, task, "fairness")
    if done:
        return read(table_dir(st, task) / "fairness_groups.csv")
    d, p, m = st.results(task)
    plans = m["plans"]
    models = [n for n in ["M1", "CKINN", "M5", "LR", "XGB"] if n in p]
    datasets = [("external", d, p, {n: plans[n]["threshold"] for n in models})]
    if task in ("A", "R"):
        od = {n: oof(st, task, n) for n in models}
        datasets.append(("development_OOF", None, od, None))
    rows = []
    for stage, dd0, pp, th in datasets:
        for n in models:
            if stage == "external":
                dd, pr, thr = (dd0, pp[n], th[n])
            else:
                dd, pr, thr = (pp[n], pp[n].p.to_numpy(), pp[n].threshold.to_numpy())
            for gname, mask in _groups(dd, task).items():
                pick = np.asarray(mask)
                part = dd.loc[pick].reset_index(drop=True)
                stt = status(part) if "time" in part else part.event.to_numpy()
                row = dict(
                    stage=stage,
                    model=n,
                    group=gname,
                    n=len(part),
                    cases=int((stt == 1).sum()),
                    controls=int((stt == 0).sum()),
                )
                if row["cases"] < st.cfg["min_cases"] or row["controls"] < st.cfg["min_controls"]:
                    row["status"] = "insufficient known cases or controls"
                else:
                    tg = thr if np.ndim(thr) == 0 else thr[pick]
                    row.update(status="estimated", **metrics(part, pr[pick], tg))
                    if n in ("M1", "CKINN"):
                        row.update(bootstrap(part, pr[pick], None, st.b["bootstrap"], st.seed + 80, tg, True))
                rows.append(row)
    t = table_dir(st, task)
    csv(t / "fairness_groups.csv", rows)
    F = pd.DataFrame(rows)
    gaps = []
    for (stage, n), g in F[F.status == "estimated"].groupby(["stage", "model"]):
        for gtype, members in [
            ("sex", ["male", "female"]),
            ("age", ["age_20_39", "age_40_59", "age_60_plus"]),
        ]:
            s = g[g.group.isin(members)]
            r = dict(stage=stage, model=n, group_type=gtype, supported_groups=len(s))
            if len(s) >= 2:
                for k in ["sensitivity", "specificity", "fnr", "fpr", "auroc", "average_precision", "brier"]:
                    r[k + "_gap"] = float(s[k].max() - s[k].min())
                r["abs_cal_intercept_gap"] = float(s.cal_intercept.abs().max() - s.cal_intercept.abs().min())
            gaps.append(r)
    csv(t / "fairness_gaps.csv", gaps)
    mark(marker, mkey)
    return F


def size_curve(st):
    _, m = st.frozen("A")
    base = {k: v for k, v in m["plans"]["M1"]["h"].items() if k not in ("directions", "lambda")}
    rich, nag = (st.data("RICH"), st.data("NAGALA"))
    y, w, _ = weights(nag)
    smoke = st.mode == "SMOKE"
    sizes = [1000, 2000] if smoke else st.cfg["validation"]["size_curve"]
    draws = [1] if smoke else st.cfg["validation"]["size_draws"]
    sizes = [n for n in sizes if n < len(rich)] or [len(rich) // 2]
    rows = []
    for size in sizes:
        for draw in draws:
            path = st.private / "v4_size" / f"n{size}_d{draw}.pkl"
            if path.exists():
                rows += restore(path)
                continue
            idx, _ = train_test_split(
                np.arange(len(rich)), train_size=size, stratify=status(rich), random_state=draw
            )
            sub = rich.iloc[np.sort(idx)].reset_index(drop=True)
            label = f"A_v4size_n{size}_d{draw}"
            m1 = st.tune("A", "M1", sub, label, grid=[dict(base, directions=[], **{"lambda": 0.0})])
            ck = select_lambda(st, "A", sub, label, base, m1["epochs"], directions_for(st, "A", "CKINN"))
            res = []
            for n, plan in [("M1", m1), ("CKINN", ck)]:
                model, _ = st.fit("A", n, sub, None, plan["h"], st.seed, plan["epochs"], label + "_refit")
                pr = model.predict(nag)
                res.append(
                    dict(
                        size=size,
                        draw=draw,
                        model=n,
                        lambda_value=plan["h"]["lambda"],
                        epochs=plan["epochs"],
                        cases=int((status(sub) == 1).sum()),
                        nagala_brier=brier(y, w, pr),
                        nagala_auroc=metrics(nag, pr, plan["threshold"])["auroc"],
                    )
                )
            for r in res:
                r["brier_minus_M1"] = r["nagala_brier"] - res[0]["nagala_brier"]
                r["auroc_minus_M1"] = r["nagala_auroc"] - res[0]["nagala_auroc"]
            dump(path, res)
            rows += res
    df = pd.DataFrame(rows)
    t = table_dir(st, "A")
    csv(t / "size_curve_draws.csv", df)
    ck = df[df.model == "CKINN"]
    summ = (
        ck.groupby("size")
        .agg(
            draws=("draw", "size"),
            cases=("cases", "mean"),
            mean_brier_minus_M1=("brier_minus_M1", "mean"),
            sd_brier_minus_M1=("brier_minus_M1", "std"),
            draws_ckinn_lower_brier=("brier_minus_M1", lambda v: int((v < 0).sum())),
            mean_auroc_minus_M1=("auroc_minus_M1", "mean"),
            sd_auroc_minus_M1=("auroc_minus_M1", "std"),
        )
        .reset_index()
    )
    csv(t / "size_curve_summary.csv", summ)
    return summ


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if Path(path).exists() else None


def fmt(v, d=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3e}" if v != 0 and abs(v) < 0.001 else f"{v:.{d}f}"


def proposal_table(st):
    ta, tr = (table_dir(st, "A"), table_dir(st, "R"))
    rows = []

    def add(item, question, artifact, estimate, decision="descriptive"):
        rows.append(
            dict(
                item=item, proposal_question=question, artifact=artifact, estimate=estimate, decision=decision
            )
        )

    ex, oo = (read(ta / "external_metrics_ci.csv"), read(ta / "internal_pooled_oof.csv"))
    if ex is not None and oo is not None:
        e, o = (ex.set_index("model"), oo.set_index("model"))
        add(
            "RQ1",
            "Can a neural network predict future type 2 diabetes from baseline variables?",
            "v4/internal_pooled_oof.csv; v4/external_metrics_ci.csv",
            "; ".join(
                (
                    f"{n}: RICH OOF AUROC {fmt(o.loc[n, 'auroc'], 3)}, NAGALA AUROC {fmt(e.loc[n, 'auroc'], 3)} ({fmt(e.loc[n, 'auroc_lo'], 3)}-{fmt(e.loc[n, 'auroc_hi'], 3)})"
                    for n in ["M1", "CKINN"]
                    if n in e.index and n in o.index
                )
            ),
        )
    c = read(ta / "contrast_ckinn_vs_m1.csv")
    if c is not None:
        c = c.set_index("metric")
        r = c.loc["brier"]
        add(
            "RQ2 / H1",
            "Does clinical knowledge improve prediction over an unconstrained network (CKINN > MLP)?",
            "v4/contrast_ckinn_vs_m1.csv",
            f"Brier CKINN - M1 {fmt(r.CKINN_minus_M1)} (95% {fmt(r.lo)} to {fmt(r.hi)}); AUROC difference {fmt(c.loc['auroc', 'CKINN_minus_M1'])} ({fmt(c.loc['auroc', 'lo'])} to {fmt(c.loc['auroc', 'hi'])}); AUPRC difference {fmt(c.loc['auprc', 'CKINN_minus_M1'])}",
            f"{r.h1_decision} (pre-specified primary contrast)",
        )
        add(
            "RQ4 / H2",
            "Do clinical constraints improve calibration?",
            "v4/contrast_ckinn_vs_m1.csv",
            "; ".join(
                (
                    f"{k}: CKINN {fmt(c.loc[k, 'CKINN'])}, M1 {fmt(c.loc[k, 'M1'])}, difference {fmt(c.loc[k, 'CKINN_minus_M1'])} (95% {fmt(c.loc[k, 'lo'])} to {fmt(c.loc[k, 'hi'])})"
                    for k in ["abs_cal_intercept", "abs_slope_minus_1", "ece"]
                )
            ),
        )
    cr, lp = (read(tr / "contrast_ckinn_vs_m1.csv"), read(ta / "loso_pooled.csv"))
    parts = []
    if cr is not None:
        r = cr.set_index("metric").loc["brier"]
        parts.append(
            f"NAGALA->RICH Brier CKINN - M1 {fmt(r.CKINN_minus_M1)} (95% {fmt(r.lo)} to {fmt(r.hi)})"
        )
    if lp is not None and len(lp):
        for r in lp.itertuples():
            parts.append(
                f"leave-one-site-out pooled {r.metric} difference {fmt(r.pooled)} (95% {fmt(r.lo)} to {fmt(r.hi)}), {int(r.k)} sites, CKINN better in {int(r.sites_ckinn_better)}, I2 {fmt(r.i2, 2)}"
            )
    if parts:
        add(
            "RQ3 / H3",
            "Do clinical constraints improve cross-population generalisation?",
            "R/v4/contrast_ckinn_vs_m1.csv; v4/loso_pooled.csv",
            "; ".join(parts),
        )
    dr, sh, stb = (
        read(ta / "directions_gradient.csv"),
        read(ta / "shap_direction.csv"),
        read(ta / "shap_stability.csv"),
    )
    parts = []
    if dr is not None:
        g = dr[(dr.cohort == "NAGALA") & dr.model.isin(["M1", "CKINN"])].pivot_table(
            index="model", columns="feature", values="wrong_way_share"
        )
        parts.append(
            "; ".join(
                (
                    f"{n} wrong-way gradient share "
                    + ", ".join((f"{f} {fmt(g.loc[n, f], 3)}" for f in g.columns))
                    for n in g.index
                )
            )
        )
    if sh is not None:
        a = sh[sh.model.isin(["M1", "CKINN"])].groupby("model").agrees_with_prior.mean()
        parts.append("; ".join((f"{n} SHAP direction agreement {fmt(v, 3)}" for n, v in a.items())))
    if stb is not None:
        a = stb.groupby("model").spearman.mean()
        parts.append("; ".join((f"{n} seed stability (Spearman) {fmt(v, 3)}" for n, v in a.items())))
    if parts:
        add(
            "RQ5 / H4",
            "Are CKINN explanations more clinically consistent than an unconstrained network?",
            "v4/directions_gradient.csv; v4/shap_direction.csv; v4/shap_stability.csv",
            "; ".join(parts),
        )
    gp = read(ta / "fairness_gaps.csv")
    if gp is not None:
        g = gp[gp.model.isin(["M1", "CKINN"])]
        add(
            "RQ6 / H5",
            "Does the model give more consistent performance across demographic groups?",
            "v4/fairness_gaps.csv; v4/fairness_groups.csv",
            "; ".join(
                (
                    f"{r.model} {r.stage} {r.group_type}: sensitivity gap {fmt(getattr(r, 'sensitivity_gap', np.nan), 3)}, AUROC gap {fmt(getattr(r, 'auroc_gap', np.nan), 3)}"
                    for r in g.itertuples()
                    if r.supported_groups >= 2
                )
            ),
        )
    fg = read(ta / "fairness_groups.csv")
    if fg is not None:
        s = fg[(fg.group == "FPG_below_5.6") & fg.model.isin(["M1", "CKINN"])]
        add(
            "RQ7",
            "Can the model identify elevated risk before diagnostic thresholds are reached?",
            "v4/fairness_groups.csv (FPG < 5.6 mmol/L)",
            "; ".join(
                (
                    f"{r.model} {r.stage}: n {int(r.n)}, cases {int(r.cases)}, "
                    + (f"AUROC {fmt(r.auroc, 3)}" if r.status == "estimated" else r.status)
                    for r in s.itertuples()
                )
            ),
        )
    for item, q in [
        (
            "Section 3C",
            "Progression from normal through prediabetes to diabetes (repeated dated states required)",
        ),
        (
            "Section 11",
            "Physiological glucose-insulin constraints (insulin and dynamic measurements required)",
        ),
        ("Section 18", "Dataset A + Dataset B -> Dataset C (a third cohort with incident outcomes required)"),
        ("Section 24", "Clinical expert agreement with explanations (clinician panel required)"),
        ("Section 26", "Clinical knowledge graph (curated evidence graph required)"),
        ("Section 21", "Clinical decision support and patient benefit (prospective evaluation required)"),
    ]:
        add(item, q, "", "not measurable with these data", "not measurable")
    csv(ta / "proposal_questions_evidence.csv", rows)
    return pd.DataFrame(rows)


def report(st):
    lines = [
        f"# CKINN Objective 1 · {st.mode} · protocol v4",
        "Every value below is read from this run's saved tables.",
    ]
    if st.mode in ("SMOKE", "AUDIT"):
        lines.append("Software check only. These numbers support no scientific conclusion.")
    for r in proposal_table(st).itertuples():
        lines.append(f"**{r.item}.** {r.proposal_question} {r.estimate} [{r.decision}]")
    atomic(st.report / "summary.md", lambda p: p.write_text("\n\n".join(lines), encoding="utf8"))
    js(
        st.report / "run_manifest.json",
        dict(
            mode=st.mode,
            run=st.run,
            identity=st.identity,
            options=st.cfg["options"],
            ckinn=st.cfg["ckinn"],
            validation=st.cfg["validation"],
            code=st.code,
            budget=st.b,
            artifacts=[str(p.relative_to(st.root)) for p in sorted(st.table.rglob("*.csv"))],
            figures=[str(p.relative_to(st.root)) for p in sorted(st.fig.rglob("*.png"))],
            frozen={
                q.name: digest(q / "manifest.json")
                for q in (st.private / "frozen").glob("*")
                if (q / "manifest.json").exists()
            },
        ),
    )


def export(st, package):
    report(st)
    dest = st.report / ("CKINN_" + st.mode + "_aggregate.zip")

    def write(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for folder in [st.table, st.fig, st.report, st.auditpath]:
                for f in folder.rglob("*"):
                    if (
                        f.is_file()
                        and f.suffix in (".csv", ".json", ".md", ".png", ".pdf")
                        and (not f.name.endswith(".zip"))
                    ):
                        z.write(f, str(f.relative_to(st.root)))
            for f in Path(package).rglob("*"):
                if f.is_file() and f.suffix in (".py", ".json", ".md"):
                    z.write(f, "software/" + str(f.relative_to(package)))
            for f in st.log.glob("*.json*"):
                z.write(f, "logs/" + f.name)

    atomic(dest, write)
    return dest
