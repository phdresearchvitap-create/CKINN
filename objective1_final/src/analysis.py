from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from sklearn.metrics import roc_curve, precision_recall_curve
from data import *
from stats import *
from models import DIRECTIONS
from runner import VARIANTS


def external_contrast(st, task="A", matched=False):
    d, p, m = st.results(task)
    if matched:
        candidate, baseline = ("fixed_positive", "fixed_zero")
        filename = "matched_positive_paired_brier.csv"
        left = m["plans"][candidate]
        right = m["plans"][baseline]
        left_base = {k: v for k, v in left["h"].items() if k != "lambda"}
        right_base = {k: v for k, v in right["h"].items() if k != "lambda"}
        if (
            left_base != right_base
            or left["epochs"] != right["epochs"]
            or (not left["h"]["lambda"] > 0)
            or (right["h"]["lambda"] != 0)
        ):
            raise ValueError("Frozen matched pair must differ only in penalty strength")
    else:
        candidate, baseline = ("M3", "M1")
        filename = "paired_brier.csv"
    y, w, _ = weights(d)
    delta = brier(y, w, p[candidate]) - brier(y, w, p[baseline])
    row = dict(
        comparison=f"{candidate} minus {baseline}",
        task=st.tag(task),
        lambda_candidate=m["plans"][candidate]["h"]["lambda"],
        lambda_baseline=m["plans"][baseline]["h"]["lambda"],
        n=len(d),
        known_cases=int(y.sum()),
        candidate_brier=brier(y, w, p[candidate]),
        baseline_brier=brier(y, w, p[baseline]),
        delta=delta,
        **bootstrap(d, p[candidate], p[baseline], st.b["bootstrap"], st.seed + 30),
    )
    csv(st.table / st.tag(task) / filename, [row])
    return row


def figures(st, task):
    d, p, m = st.results(task)
    y, w, _ = weights(d)
    ok = w > 0
    rows = []
    folder = st.fig / st.tag(task)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for n in ["M1", "M3"]:
        pred = p[n]
        cb = calbins(y, w, pred)
        cb["model"] = n
        rows += cb.to_dict("records")
        ax[0].plot(cb.predicted, cb.observed, "o-", label=n)
        if len(np.unique(y[ok])) == 2:
            f, t, _ = roc_curve(y[ok], pred[ok], sample_weight=w[ok])
            ax[1].plot(f, t, label=n)
            pr, re, _ = precision_recall_curve(y[ok], pred[ok], sample_weight=w[ok])
            ax[2].step(re, pr, where="post", label=n)
    lim = max(0.001, max(ax[0].get_xlim()[1], ax[0].get_ylim()[1]))
    ax[0].plot([0, lim], [0, lim], "--", color="gray")
    for a, (x, l) in zip(
        ax,
        [
            ("Mean predicted probability", "Weighted event fraction"),
            ("False-positive rate", "Sensitivity"),
            ("Recall", "Precision"),
        ],
    ):
        a.set(xlabel=x, ylabel=l)
        a.legend()
    fig.suptitle(f"{st.mode} · {st.tag(task)} · {st.sources(task)[1]}")
    fig.savefig(folder / "calibration_roc_pr.png", dpi=150)
    plt.close(fig)
    csv(st.table / st.tag(task) / "calibration_bins.csv", rows)


def oof(st, task, model):
    if model == "M2":
        model = "M1"
    source = st.sources(task)[0]
    d = st.data(source)
    lab = st.tag(task) + "_" + source + "_nested"
    rows = []
    for i in range(st.b["outer"]):
        o = restore(st.private / "predictions" / (lab + f"_o{i}") / (model + ".pkl"))
        rows.append(pd.DataFrame(dict(id=o["id"], p=o["p"], threshold=o["threshold"], fold=i)))
    df = d.merge(pd.concat(rows), on="id", validate="one_to_one")
    if len(df) != len(d):
        raise ValueError("Incomplete OOF coverage")
    return df


def internal_summary(st):
    tag = "A"
    folds = pd.read_csv(st.table / tag / "A_RICH_nested.csv")
    rows = []
    for n in folds.loc[folds.status == "complete", "model"].unique():
        if n.endswith("_siteG_metric"):
            continue
        try:
            d = oof(st, "A", n)
        except FileNotFoundError:
            rows.append(dict(model=n, status="incomplete outer-fold coverage"))
            continue
        rows.append(
            dict(
                model=n,
                status="computed; pooled OOF; evaluation G from RICH",
                **metrics(d, d.p.to_numpy(), d.threshold.to_numpy()),
            )
        )
    csv(st.table / tag / "pooled_oof.csv", rows)
    return pd.DataFrame(rows)


def subgroups(st, task):
    d, p, m = st.results(task)
    rows = []
    for model in ["M1", "M3"] + (["M4", "M5"] if "M5" in p else []):
        datasets = [("external", d, p[model], m["plans"][model]["threshold"])]
        if task == "A":
            od = oof(st, task, model)
            datasets.append(("development_OOF", od, od.p.to_numpy(), od.threshold.to_numpy()))
        for stage, dd, pr, threshold in datasets:
            cuts = {"all": np.ones(len(dd), bool)}
            for s in sorted(dd.sex.dropna().unique()):
                cuts[f"sex_code_{s:g}"] = dd.sex == s
            age = dd.age
            cuts.update(
                {"age_20_39": age < 40, "age_40_59": (age >= 40) & (age < 60), "age_60_plus": age >= 60}
            )
            if task == "A":
                cuts["FPG_below_5.6"] = dd.fpg < 5.6
                cuts["eight_all_observed"] = dd[CORE].notna().all(axis=1)
                cuts["eight_any_missing"] = dd[CORE].isna().any(axis=1)
                cuts["HDL_missing"] = dd.hdl.isna()
            if stage == "development_OOF":
                for site in sorted(dd.site.unique()):
                    cuts[f"site_{site}"] = dd.site == site
            for label, mask in cuts.items():
                pick = np.asarray(mask)
                part = dd.loc[pick].reset_index(drop=True)
                stat = status(part) if "time" in part else part.event.to_numpy()
                cases = int(sum(stat == 1))
                controls = int(sum(stat == 0))
                row = dict(
                    model=model,
                    stage=stage,
                    group=label,
                    n=len(part),
                    events=cases,
                    controls=controls,
                    unknown=int(sum(stat == -1)),
                )
                if cases < st.cfg["min_cases"] or controls < st.cfg["min_controls"]:
                    row["status"] = "insufficient known cases or controls"
                else:
                    th = threshold if np.ndim(threshold) == 0 else threshold[pick]
                    row.update(status="estimated", **metrics(part, pr[pick], th))
                    row.update(bootstrap(part, pr[pick], None, st.b["bootstrap"], st.seed + 31, th, True))
                rows.append(row)
    csv(st.table / st.tag(task) / "subgroups.csv", rows)
    frame = pd.DataFrame(rows)
    disparities = []
    for (model, stage), group in frame[frame.group.str.startswith("sex_code_")].groupby(["model", "stage"]):
        supported = group[group.status == "estimated"]
        result = dict(
            model=model,
            stage=stage,
            supported_categories=len(supported),
            status="insufficient paired support",
        )
        if len(supported) == 2:
            result["status"] = "descriptive absolute differences"
            for k in ["sensitivity", "specificity", "fpr", "fnr", "brier", "ece"]:
                if supported[k].notna().all():
                    result[k + "_gap"] = abs(supported[k].iloc[1] - supported[k].iloc[0])
        disparities.append(result)
    csv(st.table / st.tag(task) / "sex_gaps.csv", disparities)
    return frame


def grad_raw(models, d, feature):
    each = []
    for f in models:
        x = f.prep.transform(d)[0]
        j = f.prep.names.index(feature)
        t = torch.from_numpy(x).requires_grad_(True)
        f.obj.eval()
        grad = torch.autograd.grad(f.obj(t).sigmoid().sum(), t)[0][:, j].detach().numpy()
        each.append(grad / f.prep.scale[j])
    return np.mean(each, axis=0)


def probability(models, d):
    return np.mean([f.predict(d) for f in models], axis=0)


def ig(model, d, steps=64):
    pp = model.prep
    x = torch.tensor(pp.transform(d)[0])
    base = x.clone()
    base[:, : len(pp.names)] = torch.tensor((pp.med - pp.mu) / pp.scale, dtype=torch.float32)
    delta = x - base
    grads = torch.zeros_like(x)
    model.obj.eval()
    for i in range(steps):
        u = (base + delta * ((i + 0.5) / steps)).requires_grad_(True)
        grads += torch.autograd.grad(model.obj(u).sigmoid().sum(), u)[0].detach()
    a = (delta * grads / steps).detach().numpy()
    with torch.no_grad():
        basep = model.obj(base).sigmoid().numpy()
    return (a, basep)


def explain(st, task):
    tag = st.tag(task)
    folder, m = st.frozen(task)
    src = st.data(st.sources(task)[0])
    target = st.data(st.sources(task)[1])
    d = target.sample(min(512, len(target)), random_state=st.seed + 19).reset_index(drop=True)
    directions = []
    attributions = []
    ice_agg = []
    steps = {"fpg": 0.1, "bmi": 1.0, "age": 1.0, "sbp": 5.0, "tg": 0.1, "hdl": 0.1}
    figdir = st.fig / tag
    figdir.mkdir(parents=True, exist_ok=True)
    names = ["M1", "M3"] + [n for n in ("fixed_zero", "fixed_positive", "expanded_prior") if n in m["plans"]]
    for n in names:
        files = [f for f in m["files"] if f.rsplit("_", 1)[0] == n]
        nets = [restore(folder / f) for f in files]
        pp = nets[0].prep
        for role, models in [("frozen_average", nets)] + (
            [(f"seed_{f.rsplit('_', 1)[-1].split('.')[0]}", [net]) for f, net in zip(files, nets)]
            if len(nets) > 1
            else []
        ):
            for feature in (
                list(DIRECTIONS) if n == "expanded_prior" else ["bmi", "fpg"] if task == "A" else ["bmi"]
            ):
                j = pp.names.index(feature)
                step = steps[feature]
                eps = 0.005 * pp.scale[j]
                use = d[feature].between(pp.lo[j] + eps, pp.hi[j] - max(step, eps))
                dd = d.loc[use].copy()
                row = dict(
                    model=n,
                    deployment=role,
                    feature=feature,
                    raw_step=step,
                    n_sample=len(d),
                    n_eligible=len(dd),
                )
                if len(dd):
                    center = probability(models, dd)

                    def shifted(amount):
                        tmp = dd.copy()
                        tmp[feature] = dd[feature] + amount
                        return probability(models, tmp)

                    change = shifted(step) - center
                    sign = DIRECTIONS[feature]
                    deriv = grad_raw(models, dd, feature)
                    numeric = (shifted(eps) - shifted(-eps)) / (2 * eps)
                    err = np.abs(deriv - numeric) * pp.scale[j]
                    agree = np.allclose(deriv * pp.scale[j], numeric * pp.scale[j], atol=3e-05, rtol=0.015)
                    row.update(
                        violations=int(sum(sign * change < -1e-06)),
                        violation_rate=np.mean(sign * change < -1e-06),
                        mean_wrong_way_change=np.maximum(0, -sign * change).mean(),
                        meaningful_fraction=np.mean(abs(change) >= 0.001),
                        flat_fraction=np.mean(abs(change) <= 1e-06),
                        gradient_violation_rate=np.mean(sign * deriv * pp.scale[j] < -1e-06),
                        max_standardized_check_error=err.max(),
                        finite_gradient_check=bool(agree),
                    )
                    if not agree:
                        st.record("diagnostics", dict(task=task, **row))
                    if role == "frozen_average" and feature in ("fpg", "bmi"):
                        grid = np.linspace(pp.lo[j], pp.hi[j], 25)
                        records = dd.iloc[: min(24, len(dd))].copy()
                        curves = []
                        for g in grid:
                            test = records.copy()
                            test[feature] = g
                            curves.append(probability(models, test))
                        mat = np.array(curves)
                        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
                        ax.plot(grid, mat, alpha=0.3, linewidth=1)
                        ax.set(
                            xlabel=f"{feature} ({UNITS[feature]})",
                            ylabel="Predicted probability",
                            title=f"{n} · individual sensitivity",
                        )
                        priv = st.private / "examples" / tag
                        priv.mkdir(parents=True, exist_ok=True)
                        fig.savefig(priv / f"{n}_{feature}_ICE.png", dpi=130)
                        plt.close(fig)
                        for g, v in zip(grid, mat):
                            ice_agg.append(dict(model=n, feature=feature, value=g, mean=v.mean(), n=len(v)))
                directions.append(row)
        arr, base = zip(*(ig(net, d) for net in nets))
        arr = np.mean(arr, axis=0)
        base = np.mean(base, axis=0)
        val = probability(nets, d)
        err = val - base - arr.sum(1)
        for i, feature in enumerate(pp.names):
            attributions.append(
                dict(
                    model=n,
                    feature=feature,
                    mean_abs_IG=abs(arr[:, i]).mean(),
                    mean_signed_IG=arr[:, i].mean(),
                    n=len(d),
                    steps=64,
                    max_abs_completeness_error=abs(err).max(),
                )
            )
        examples = pd.DataFrame(arr[:5, : len(pp.names)], columns=pp.names)
        examples.insert(0, "example", np.arange(1, len(examples) + 1))
        examples["baseline_probability"] = base[:5]
        examples["prediction"] = val[:5]
        examples["integration_residual"] = err[:5]
        csv(st.private / "examples" / tag / (n + "_IG.csv"), examples)
    if ice_agg:
        f = pd.DataFrame(ice_agg)
        fig, axs = plt.subplots(
            1, f.feature.nunique(), figsize=(5 * f.feature.nunique(), 4), squeeze=False, layout="constrained"
        )
        for ax, (feat, g) in zip(axs.ravel(), f.groupby("feature")):
            for n, h in g.groupby("model"):
                ax.plot(h.value, h["mean"], label=n)
            ax.set(xlabel=f"{feat} ({UNITS[feat]})", ylabel="Mean predicted probability")
            ax.legend()
        fig.suptitle(f"{st.mode} · {tag} · average raw-input perturbations")
        fig.savefig(figdir / "average_perturbation.png", dpi=150)
        plt.close(fig)
    for name, arr in [("directions", directions), ("attributions", attributions), ("perturbation", ice_agg)]:
        csv(st.table / tag / (name + ".csv"), arr)
    if st.cfg["options"]["tree_shap"]:
        import shap

        model = restore(folder / next((f for f in m["files"] if f.startswith("RF_"))))
        bg = model.prep.transform(src.sample(min(64, len(src)), random_state=st.seed))[0]
        vals = shap.TreeExplainer(
            model.obj, data=bg, feature_perturbation="interventional", model_output="probability"
        ).shap_values(model.prep.transform(d.iloc[:32])[0])
        a = np.asarray(vals)
        if isinstance(vals, list):
            a = np.asarray(vals[1])
        elif a.ndim == 3:
            a = a[:, :, 1]
        names = model.prep.names + [k + "_missing" for k in model.prep.names]
        csv(
            st.table / tag / "rf_shap.csv",
            pd.DataFrame(
                dict(feature=names, mean_abs_SHAP=abs(a).mean(0), background=len(bg), n=min(32, len(d)))
            ),
        )


def updating(st):
    d, p, m = st.results("A")
    th = m["plans"]["M3"]["threshold"]
    size = st.cfg["update_n"]
    if len(d) <= size:
        raise ValueError("Target too small for reserved update split")
    order = np.random.default_rng(st.seed + 66).permutation(len(d))
    a, b = (order[:size], order[size:])
    dump(
        st.private / "splits" / "target_update.pkl",
        dict(update=d.id.iloc[a].tolist(), evaluation=d.id.iloc[b].tolist()),
    )
    row = []
    dd = d.iloc[a]
    stat = status(dd)
    if sum(stat == 1) < st.cfg["min_cases"] or sum(stat == 0) < st.cfg["min_controls"]:
        row.append(
            dict(
                status="skipped for insufficient updating support",
                n=len(a),
                cases=int(sum(stat == 1)),
                controls=int(sum(stat == 0)),
            )
        )
    else:
        y, w, _ = weights(dd)
        offset = calibration(y, w, p["M3"][a])[0]
        after = expit(logit(np.clip(p["M3"][b], 1e-07, 1 - 1e-07)) + offset)
        for name, v in [("frozen", p["M3"][b]), ("intercept_updated", after)]:
            row.append(
                dict(
                    model=name,
                    status="estimated",
                    update_n=len(a),
                    update_events=int(sum(stat == 1)),
                    offset=offset,
                    **metrics(d.iloc[b], v, th),
                )
            )
    csv(st.table / "A" / "target_update.csv", row)
    return row


def evidence(st):
    rows = []
    for task in ["A", "B"]:
        tag = st.tag(task)
        table = st.table / tag
        for claim, file in [
            ("External predictive error and calibration", "external_metrics.csv"),
            ("Paired risk error difference", "paired_brier.csv"),
            ("Subgroup estimates conditional on support", "subgroups.csv"),
            ("Signed local clinical behavior", "directions.csv"),
            ("Model-attributed probability contributions", "attributions.csv"),
        ]:
            rows.append(
                dict(
                    task=tag,
                    claim=claim,
                    artifact=file,
                    status="computed" if (table / file).is_file() else "not run",
                    scope=(
                        "software check only"
                        if st.mode in ("SMOKE", "AUDIT")
                        else "extract-level screening" if task == "B" else "recorded detection by 3 years"
                    ),
                )
            )
        for claim in [
            "Clinical benefit",
            "Adjudicated subtype",
            "Clinician agreement",
            "Physiological dynamics",
            "Repeated prediabetes transitions",
        ]:
            rows.append(
                dict(
                    task=tag,
                    claim=claim,
                    status="unmeasured",
                    artifact="",
                    scope="requires independent evidence",
                )
            )
    csv(st.report / "claim_to_evidence.csv", rows)
    lines = [f"# Objective 1 · {st.mode}", "Reanalysis. Results below are read from this run's saved tables."]
    if st.mode in ("AUDIT", "SMOKE"):
        lines += ["Software and data checks only. These outputs support no study performance conclusion."]
    for tag in ["A", st.tag("B")]:
        if tag == "B_as_coded":
            lines.append(
                "The binary-zero screening subset has unresolved producer recoding for borderline and missing responses. Interpret it at the extract level."
            )
        p = st.table / tag / "paired_brier.csv"
        if p.exists():
            z = pd.read_csv(p).iloc[0]
            label = (
                "includes zero"
                if z.delta_lo <= 0 <= z.delta_hi
                else "lies below zero" if z.delta_hi < 0 else "lies above zero"
            )
            if not np.isfinite([z.delta_lo, z.delta_hi]).all():
                label = "is unavailable for this run"
            lines.append(
                f"## {tag}\n\nM3 minus M1 Brier difference {z.delta:.6g}. The paired 95% interval [{z.delta_lo:.6g}, {z.delta_hi:.6g}] {label}. Negative values favor M3. Uncertainty is conditional on these fitted models."
            )
            if z.lambda_candidate == 0:
                lines.append(
                    "M3 selected a zero clinical penalty. This comparison estimates performance under the primary tuning rule."
                )
        else:
            lines.append(f"## {tag}\n\nThe paired comparison has not run.")
        matched = st.table / tag / "matched_positive_paired_brier.csv"
        if matched.exists():
            result = pd.read_csv(matched).iloc[0]
            lines.append(
                f"The secondary matched comparison selected lambda {result.lambda_candidate:g} in RICH. Positive minus zero Brier difference is {result.delta:.6g}, with paired 95% bounds [{result.delta_lo:.6g}, {result.delta_hi:.6g}]. Negative values favor the positive-penalty fit."
            )
        ext = st.table / tag / "external_metrics.csv"
        if ext.exists():
            models = pd.read_csv(ext).set_index("model")
            for name in ["M1", "M3"]:
                r = models.loc[name]
                lines.append(
                    f"{name} had Brier {r.brier:.5g}, AUROC {r.auroc:.4g}, average precision {r.average_precision:.4g}, mean prediction {r.predicted:.4g}, and adjusted observed fraction {r.observed:.4g}. Calibration intercept {r.cal_intercept:.4g} and slope {r.cal_slope:.4g} use {int(r.n)} participants with {int(r.events)} known cases."
                )
        dirs = st.table / tag / "directions.csv"
        if dirs.exists():
            di = pd.read_csv(dirs)
            if "finite_gradient_check" in di:
                failed = di.finite_gradient_check.eq(False).sum()
                lines.append(
                    f"Analytical versus numerical direction checks recorded {failed} disagreements outside tolerance. See eligible denominators and flatness in directions.csv."
                )
        groups = st.table / tag / "subgroups.csv"
        if groups.exists():
            sg = pd.read_csv(groups)
            lines.append(
                f"{(sg.status != 'estimated').sum()} of {len(sg)} model-by-group estimates lacked the prespecified support."
            )
    lines.append(
        "Five- and seven-year hypotheses are inactive. Study claims require a scientific-profile run and review of support, calibration, missingness and fit failures."
    )
    atomic(st.report / "summary.md", lambda p: p.write_text("\n\n".join(lines), encoding="utf8"))
    js(
        st.report / "run_manifest.json",
        dict(
            mode=st.mode,
            run=st.run,
            identity=st.identity,
            options=st.cfg["options"],
            code=st.code,
            budget=st.b,
            artifacts=[str(p.relative_to(st.root)) for p in st.table.rglob("*.csv")],
            conditional_uncertainty="participant resampling of frozen predictions; separate seed file",
            frozen={
                q.name: digest(q / "manifest.json")
                for q in (st.private / "frozen").glob("*")
                if (q / "manifest.json").exists()
            },
        ),
    )


def export(st, package):
    evidence(st)
    dest = st.report / ("CKINN_" + st.mode + "_aggregate.zip")

    def write(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for folder in [st.table, st.fig, st.report, st.auditpath]:
                for f in folder.rglob("*"):
                    if f.is_file() and f.suffix in (".csv", ".json", ".md", ".png"):
                        z.write(f, str(f.relative_to(st.root)))
            for f in Path(package).rglob("*"):
                if (
                    f.is_file()
                    and f.suffix in (".py", ".ipynb", ".txt", ".md", ".json", ".csv", ".png")
                    and (not f.is_relative_to(st.root / "data"))
                ):
                    z.write(f, "software/" + str(f.relative_to(package)))
            for f in st.log.glob("*.json*"):
                z.write(f, "logs/" + f.name)

    atomic(dest, write)
    return dest
