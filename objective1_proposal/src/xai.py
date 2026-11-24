import itertools
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from data import CORE, csv, restore
from models import DIRECTIONS
from analysis import probability, ig
from proposal import table_dir, cached, mark, read

STEPS = {"fpg": 0.1, "bmi": 1.0, "age": 1.0, "sbp": 5.0, "tg": 0.1, "hdl": 0.1}
SIX = ["age", "bmi", "sbp", "fpg", "tg", "hdl"]
MODIFIABLE = ["fpg", "bmi", "sbp", "tg", "hdl"]
LABELS = {
    "age": "age",
    "sex": "sex",
    "bmi": "BMI",
    "sbp": "systolic blood pressure",
    "alt": "ALT",
    "fpg": "fasting glucose",
    "tg": "triglycerides",
    "hdl": "HDL cholesterol",
}


def frozen(st, task="A"):
    folder, m = st.frozen(task)
    return ({n: [restore(folder / f) for f in m["files"] if f.rsplit("_", 1)[0] == n] for n in m["plans"]}, m)


def grad_prob(models, d):
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


def directions(st, task="A"):
    marker, mkey, done = cached(st, task, "directions")
    if done:
        return read(table_dir(st, task) / "directions_gradient.csv")
    models, m = frozen(st, task)
    nets = [n for n in ["M1", "CKINN", "CKINN_FPG_BMI", "M4", "M5"] if n in models]
    rows, steps = ([], [])
    for cohort in st.sources(task):
        d = st.data(cohort)
        for n in nets:
            g, inside, names = grad_prob(models[n], d)
            h = m["plans"][n]["h"]
            for f in SIX:
                j = names.index(f)
                ok = inside[:, j]
                s = DIRECTIONS[f]
                rows.append(
                    dict(
                        cohort=cohort,
                        model=n,
                        lambda_value=h["lambda"],
                        feature=f,
                        constrained=f in h.get("directions", []) and h["lambda"] > 0,
                        eligible=int(ok.sum()),
                        wrong_way_share=float(np.mean(s * g[ok, j] < 0)),
                        median_abs_dp_dz=float(np.median(np.abs(g[ok, j]))),
                    )
                )
    target = st.data(st.sources(task)[1])
    sample = target.sample(
        min(st.cfg["validation"]["explain_n"], len(target)), random_state=st.seed + 19
    ).reset_index(drop=True)
    for n in nets:
        pp = models[n][0].prep
        for f in SIX:
            j = pp.names.index(f)
            step = STEPS[f]
            dd = sample.loc[sample[f].between(pp.lo[j], pp.hi[j] - step)].copy()
            if not len(dd):
                continue
            change = probability(models[n], dd.assign(**{f: dd[f] + step})) - probability(models[n], dd)
            s = DIRECTIONS[f]
            steps.append(
                dict(
                    model=n,
                    feature=f,
                    raw_step=step,
                    eligible=len(dd),
                    wrong_way_share=float(np.mean(s * change < -1e-06)),
                    mean_change=float(change.mean()),
                    flat_share=float(np.mean(np.abs(change) <= 1e-06)),
                )
            )
    t = table_dir(st, task)
    csv(t / "directions_gradient.csv", rows)
    csv(t / "directions_finite_step.csv", steps)
    mark(marker, mkey)
    return pd.DataFrame(rows)


def shap_values(st, fs, sample, background, seed):
    import shap

    def f(x):
        return probability(fs, pd.DataFrame(x, columns=CORE))

    ex = shap.explainers.Permutation(
        f, shap.maskers.Independent(background, max_samples=len(background)), seed=seed
    )
    out = ex(
        sample,
        max_evals=max(
            2 * len(CORE) + 1,
            st.cfg["validation"]["shap_max_evals"] if st.mode != "SMOKE" else 2 * len(CORE) + 1,
        ),
        silent=True,
    )
    return (np.asarray(out.values), np.asarray(out.base_values))


def explain(st, task="A"):
    marker, mkey, done = cached(st, task, "explain")
    if done:
        return read(table_dir(st, task) / "shap_importance.csv")
    models, m = frozen(st, task)
    src, target = (st.data(st.sources(task)[0]), st.data(st.sources(task)[1]))
    n_ex = st.cfg["validation"]["explain_n"] if st.mode != "SMOKE" else 40
    n_bg = st.cfg["validation"]["background_n"] if st.mode != "SMOKE" else 20
    sample = target.sample(min(n_ex, len(target)), random_state=st.seed + 19).reset_index(drop=True)
    background = src.sample(min(n_bg, len(src)), random_state=st.seed + 23)[CORE].to_numpy(dtype=float)
    X = sample[CORE].to_numpy(dtype=float)
    imp, direc, stab, igrows = ([], [], [], [])
    store = {}
    for n in [x for x in ["M1", "CKINN", "LR", "XGB"] if x in models]:
        per_seed = [shap_values(st, [f], X, background, st.seed + 29) for f in models[n]]
        vals = np.mean([v for v, _ in per_seed], axis=0)
        store[n] = vals
        for j, feat in enumerate(CORE):
            imp.append(
                dict(
                    model=n,
                    feature=feat,
                    mean_abs_shap=float(np.mean(np.abs(vals[:, j]))),
                    mean_shap=float(np.mean(vals[:, j])),
                )
            )
            if feat in SIX:
                obs = np.isfinite(X[:, j])
                rho = (
                    spearmanr(X[obs, j], vals[obs, j])[0]
                    if obs.sum() > 2 and np.std(X[obs, j]) > 0
                    else np.nan
                )
                direc.append(
                    dict(
                        model=n,
                        feature=feat,
                        prior_sign=DIRECTIONS[feat],
                        spearman_value_vs_shap=float(rho),
                        agrees_with_prior=(
                            bool(np.sign(rho) == DIRECTIONS[feat]) if np.isfinite(rho) else None
                        ),
                    )
                )
        if len(per_seed) > 1:
            ranks = [np.mean(np.abs(v), axis=0) for v, _ in per_seed]
            for a, b in itertools.combinations(range(len(ranks)), 2):
                stab.append(
                    dict(model=n, seed_a=a, seed_b=b, spearman=float(spearmanr(ranks[a], ranks[b])[0]))
                )
        if models[n][0].kind == "net":
            arr, base = zip(*(ig(f, sample) for f in models[n]))
            arr, base = (np.mean(arr, axis=0), np.mean(base, axis=0))
            err = probability(models[n], sample) - base - arr.sum(1)
            for j, feat in enumerate(models[n][0].prep.names):
                igrows.append(
                    dict(
                        model=n,
                        feature=feat,
                        mean_abs_ig=float(np.abs(arr[:, j]).mean()),
                        max_abs_completeness_error=float(np.abs(err).max()),
                    )
                )
    I = pd.DataFrame(imp)
    I["rank"] = I.groupby("model").mean_abs_shap.rank(ascending=False).astype(int)
    I["share_of_total"] = I.mean_abs_shap / I.groupby("model").mean_abs_shap.transform("sum")
    t = table_dir(st, task)
    csv(t / "shap_importance.csv", I)
    csv(t / "shap_direction.csv", direc)
    csv(t / "shap_stability.csv", stab)
    csv(t / "integrated_gradients.csv", igrows)
    counterfactuals(st, task, models, sample)
    reasoning(st, task, models, sample, store)
    mark(marker, mkey)
    return I


def counterfactuals(st, task, models, sample):
    rows = []
    for n in [x for x in ["M1", "CKINN", "LR", "XGB"] if x in models]:
        fs = models[n]
        med = dict(zip(fs[0].prep.names, fs[0].prep.med))
        p0 = probability(fs, sample)
        joint = sample.copy()
        for f in MODIFIABLE:
            s = DIRECTIONS[f]
            x = sample[f]
            applicable = x > med[f] if s > 0 else x < med[f]
            applicable = applicable.fillna(False).to_numpy()
            cf = sample.copy()
            cf.loc[applicable, f] = med[f]
            joint.loc[applicable, f] = med[f]
            delta = probability(fs, cf) - p0
            rows.append(
                dict(
                    model=n,
                    factor=f,
                    applicable=int(applicable.sum()),
                    mean_risk_change=float(delta[applicable].mean()) if applicable.any() else np.nan,
                    share_risk_decreased=(
                        float(np.mean(delta[applicable] < 0)) if applicable.any() else np.nan
                    ),
                    share_risk_increased=(
                        float(np.mean(delta[applicable] > 1e-09)) if applicable.any() else np.nan
                    ),
                )
            )
        delta = probability(fs, joint) - p0
        rows.append(
            dict(
                model=n,
                factor="all modifiable",
                applicable=len(sample),
                mean_risk_change=float(delta.mean()),
                share_risk_decreased=float(np.mean(delta < 0)),
                share_risk_increased=float(np.mean(delta > 1e-09)),
            )
        )
    csv(table_dir(st, task) / "counterfactual_summary.csv", rows)


def reasoning(st, task, models, sample, store):
    if "CKINN" not in models or "CKINN" not in store:
        return
    fs = models["CKINN"]
    p = probability(fs, sample)
    order = np.argsort(-p)[: min(5, len(sample))]
    med = dict(zip(fs[0].prep.names, fs[0].prep.med))
    g, inside, names = grad_prob(fs, sample.iloc[order].reset_index(drop=True))
    texts, summary = ([], [])
    for k, i in enumerate(order):
        phi = store["CKINN"][i]
        share = np.abs(phi) / np.abs(phi).sum() if np.abs(phi).sum() > 0 else np.zeros_like(phi)
        top = np.argsort(-np.abs(phi))[:3]
        parts = [
            f"{LABELS[CORE[j]]} ({100 * share[j]:.0f}% of the attribution, {('raising' if phi[j] > 0 else 'lowering')} risk)"
            for j in top
        ]
        wrong = [f for f in SIX if DIRECTIONS[f] * g[k, names.index(f)] < 0 and inside[k, names.index(f)]]
        cf = sample.iloc[[i]].copy()
        moved = []
        for f in ["fpg", "bmi"]:
            if np.isfinite(cf[f].iloc[0]) and cf[f].iloc[0] > med[f]:
                cf[f] = med[f]
                moved.append(LABELS[f])
        p_cf = float(probability(fs, cf)[0])
        text = (
            f"Example {k + 1}. Predicted three-year risk {100 * p[i]:.1f}%. The largest contributions come from "
            + "; ".join(parts)
            + ". "
            + (
                "The local behaviour of the model agrees with every constrained clinical direction. "
                if not wrong
                else "The local gradient disagrees with the prior for " + ", ".join(wrong) + ". "
            )
            + (
                f"If {' and '.join(moved)} were at the development median, the model's predicted risk would be {100 * p_cf:.1f}%, according to its learned relationships. This describes model behaviour and is not medical advice."
                if moved
                else "Fasting glucose and BMI are at or below the development median."
            )
        )
        texts.append(text)
        summary.append(
            dict(
                example=k + 1,
                top_contributor=CORE[top[0]],
                constrained_directions_respected=not wrong,
                counterfactual_available=bool(moved),
            )
        )
    priv = st.private / "examples" / st.tag(task)
    priv.mkdir(parents=True, exist_ok=True)
    (priv / "ckinn_reasoning_examples.md").write_text("\n\n".join(texts), encoding="utf-8")
    csv(table_dir(st, task) / "reasoning_summary.csv", summary)
