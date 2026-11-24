import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from stats import weights
from proposal import table_dir, read

STYLE = {
    "M1": dict(color="#1f5fbf", marker="o", linestyle="-", label="M1 (MLP)"),
    "CKINN": dict(color="#d9480f", marker="s", linestyle="--", label="M3 (CKINN)"),
    "LR": dict(color="#2b8a3e", marker="^", linestyle=":", label="Logistic regression"),
    "XGB": dict(color="#6b4fbb", marker="D", linestyle="-.", label="XGBoost"),
    "treat_all": dict(color="#555555", marker="", linestyle="-", label="Treat all"),
    "treat_none": dict(color="#000000", marker="", linestyle="-", label="Treat none"),
}
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}
)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def calibration(st, task):
    cal = read(table_dir(st, task) / "calibration_deciles_ci.csv")
    if cal is None:
        return
    d, p, m = st.results(task)
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1]), layout="constrained"
    )
    top = float(np.nanmax(cal[["predicted", "observed_hi"]].to_numpy())) * 1.08
    for ax, group in zip(axes[:2], [["M1", "CKINN"], ["LR", "XGB"]]):
        for k, n in enumerate([g for g in group if g in set(cal.model)]):
            c = cal[cal.model == n].sort_values("bin")
            s = STYLE[n]
            x = c.predicted * (1 + 0.025 * (k - 0.5))
            err = [
                np.clip(np.nan_to_num(c.observed - c.observed_lo), 0, None),
                np.clip(np.nan_to_num(c.observed_hi - c.observed), 0, None),
            ]
            ax.errorbar(
                x,
                c.observed,
                yerr=err,
                fmt=s["marker"],
                color=s["color"],
                mfc="white" if k else s["color"],
                ms=6,
                capsize=2,
                lw=1,
                label=s["label"],
            )
            ax.plot(x, c.observed, linestyle=s["linestyle"], color=s["color"], lw=1.2)
        ax.plot([0, top], [0, top], color="#888888", lw=1, linestyle=(0, (4, 3)), label="Perfect calibration")
        ax.set(
            xlim=(0, top),
            ylim=(0, top),
            xlabel="Mean predicted three-year risk",
            ylabel="Observed risk (IPCW)",
        )
        ax.legend(frameon=False, loc="upper left")
    ax = axes[2]
    bins = np.logspace(
        np.log10(max(1e-05, min((float(p[n].min()) for n in ["M1", "CKINN"])))),
        np.log10(max((float(p[n].max()) for n in ["M1", "CKINN"]))),
        40,
    )
    for n in ["M1", "CKINN"]:
        s = STYLE[n]
        ax.hist(
            p[n],
            bins=bins,
            histtype="step",
            color=s["color"],
            linestyle=s["linestyle"],
            lw=1.5,
            label=s["label"],
        )
    ax.set(xscale="log", xlabel="Predicted three-year risk (log scale)", ylabel="Participants")
    ax.legend(frameon=False)
    for a, letter in zip(axes, "abc"):
        a.set_title(letter, loc="left", fontweight="bold")
    save(fig, st.fig / st.tag(task) / "v4", "calibration")


def roc_pr(st, task):
    d, p, m = st.results(task)
    y, w, _ = weights(d)
    k = w > 0
    if np.unique(y[k]).size < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    for n in [x for x in ["M1", "CKINN", "LR", "XGB"] if x in p]:
        s = STYLE[n]
        f, tp, _ = roc_curve(y[k], p[n][k], sample_weight=w[k])
        axes[0].plot(f, tp, color=s["color"], linestyle=s["linestyle"], lw=1.5, label=s["label"])
        pr, re, _ = precision_recall_curve(y[k], p[n][k], sample_weight=w[k])
        axes[1].step(
            re, pr, where="post", color=s["color"], linestyle=s["linestyle"], lw=1.3, label=s["label"]
        )
    axes[0].plot([0, 1], [0, 1], color="#888888", lw=1, linestyle=(0, (4, 3)))
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity")
    axes[1].axhline(float(np.average(y[k], weights=w[k])), color="#888888", lw=1, linestyle=(0, (4, 3)))
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)")
    for a, letter in zip(axes, "ab"):
        a.legend(frameon=False)
        a.set_title(letter, loc="left", fontweight="bold")
    save(fig, st.fig / st.tag(task) / "v4", "roc_pr")


def decision_curve(st, task):
    nb = read(table_dir(st, task) / "net_benefit.csv")
    if nb is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")
    for n in ["treat_none", "treat_all", "LR", "XGB", "M1", "CKINN"]:
        g = nb[nb.model == n].sort_values("threshold")
        if not len(g):
            continue
        s = STYLE[n]
        ax.plot(
            100 * g.threshold,
            1000 * g.net_benefit,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"] or None,
            ms=4,
            lw=1.4,
            label=s["label"],
        )
        if n in ("M1", "CKINN"):
            ax.fill_between(100 * g.threshold, 1000 * g.lo, 1000 * g.hi, color=s["color"], alpha=0.12, lw=0)
    lo = 1000 * nb[nb.model.isin(["M1", "CKINN", "LR", "XGB"])].net_benefit.min()
    ax.set(ylim=(min(-1, lo * 1.3), None), xlabel="Risk threshold (%)", ylabel="Net benefit per 1,000 people")
    ax.legend(frameon=False)
    save(fig, st.fig / st.tag(task) / "v4", "decision_curve")


def directions(st, task):
    dr = read(table_dir(st, task) / "directions_gradient.csv")
    if dr is None:
        return
    feats = ["age", "bmi", "sbp", "fpg", "tg", "hdl"]
    names = {"age": "Age ↑", "bmi": "BMI ↑", "sbp": "SBP ↑", "fpg": "FPG ↑", "tg": "TG ↑", "hdl": "HDL ↓"}
    cohorts = list(dict.fromkeys(dr.cohort))
    fig, axes = plt.subplots(
        1, len(cohorts), figsize=(5.5 * len(cohorts), 4), layout="constrained", squeeze=False
    )
    for ax, c in zip(axes[0], cohorts):
        g = (
            dr[dr.cohort == c]
            .pivot_table(index="feature", columns="model", values="wrong_way_share")
            .reindex(feats)
        )
        x = np.arange(len(feats))
        for k, n in enumerate([m for m in ["M1", "CKINN"] if m in g.columns]):
            s = STYLE[n]
            ax.bar(
                x + (k - 0.5) * 0.38,
                100 * g[n].to_numpy(),
                width=0.38,
                color=s["color"],
                alpha=0.85 if k == 0 else 0.6,
                hatch=None if k == 0 else "//",
                edgecolor=s["color"],
                label=s["label"],
            )
        ax.set_xticks(x, [names[f] for f in feats])
        ax.set(ylabel="Participants with wrong-way gradient (%)", ylim=(0, 100), title=c)
        ax.legend(frameon=False)
    save(fig, st.fig / st.tag(task) / "v4", "clinical_directions")


def shap_importance(st, task):
    imp = read(table_dir(st, task) / "shap_importance.csv")
    if imp is None:
        return
    models = [m for m in ["M1", "CKINN", "LR", "XGB"] if m in set(imp.model)]
    order = (
        imp[imp.model == "CKINN"].sort_values("mean_abs_shap").feature.tolist()
        if "CKINN" in models
        else sorted(set(imp.feature))
    )
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    y = np.arange(len(order))
    hgt = 0.8 / len(models)
    for k, n in enumerate(models):
        g = imp[imp.model == n].set_index("feature").reindex(order)
        ax.barh(
            y + (k - (len(models) - 1) / 2) * hgt,
            100 * g.share_of_total,
            height=hgt,
            color=STYLE[n]["color"],
            label=STYLE[n]["label"],
        )
    ax.set_yticks(y, order)
    ax.set(xlabel="Share of mean |SHAP| (%)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, st.fig / st.tag(task) / "v4", "shap_importance")


def lambda_path(st, task):
    lp = read(table_dir(st, task) / "lambda_path_final.csv")
    if lp is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    for n, s in [
        ("CKINN", STYLE["CKINN"]),
        ("CKINN_FPG_BMI", dict(color="#7f7f7f", marker="v", linestyle=":", label="CKINN (FPG, BMI only)")),
    ]:
        g = lp[lp.model == n].sort_values("lambda_value")
        if not len(g):
            continue
        x = np.where(g.lambda_value == 0, 1, g.lambda_value)
        ax.errorbar(
            x,
            1000 * g.inner_cv_brier,
            yerr=1000 * g.paired_se,
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
            capsize=2,
            label=s["label"],
        )
        sel = g[g.selected]
        ax.scatter(
            np.where(sel.lambda_value == 0, 1, sel.lambda_value),
            1000 * sel.inner_cv_brier,
            s=120,
            facecolors="none",
            edgecolors=s["color"],
            lw=1.5,
        )
    ax.set(xscale="log", xlabel="λ (0 drawn at 1)", ylabel="Inner-validation IPCW Brier × 1,000")
    ax.legend(frameon=False)
    save(fig, st.fig / st.tag(task) / "v4", "lambda_selection")


def loso(st):
    ld, lp = (read(table_dir(st, "A") / "loso_ckinn_vs_m1.csv"), read(table_dir(st, "A") / "loso_pooled.csv"))
    if ld is None or not len(ld):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, key, scale, lab in [
        (axes[0], "brier", 1000, "Brier difference × 1,000 (CKINN − M1)"),
        (axes[1], "auroc", 1, "AUROC difference (CKINN − M1)"),
    ]:
        g = ld[ld.metric == key].reset_index(drop=True)
        y = np.arange(len(g))[::-1] + 1
        xerr = [
            np.clip(np.nan_to_num(scale * (g.CKINN_minus_M1 - g.lo)), 0, None),
            np.clip(np.nan_to_num(scale * (g.hi - g.CKINN_minus_M1)), 0, None),
        ]
        ax.errorbar(scale * g.CKINN_minus_M1, y, xerr=xerr, fmt="s", color=STYLE["CKINN"]["color"], capsize=2)
        labels = [f"Site {s} (n={n:,})" for s, n in zip(g.site, g.n)]
        if lp is not None and key in set(lp.metric):
            r = lp[lp.metric == key].iloc[0]
            ax.errorbar(
                [scale * r.pooled],
                [0],
                xerr=[[max(0, scale * (r.pooled - r.lo))], [max(0, scale * (r.hi - r.pooled))]],
                fmt="D",
                color="black",
                capsize=3,
            )
            labels = ["Pooled (random effects)"] + labels[::-1]
            ax.set_yticks([0] + list(y[::-1]), labels)
        ax.axvline(0, color="#888888", lw=1, linestyle=(0, (4, 3)))
        ax.set(xlabel=lab)
    save(fig, st.fig / "A" / "v4", "leave_one_site_out")


def size_curve(st):
    s = read(table_dir(st, "A") / "size_curve_summary.csv")
    if s is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for ax, key, scale, lab in [
        (axes[0], "auroc", 1, "AUROC difference in NAGALA (CKINN − M1)"),
        (axes[1], "brier", 1000, "Brier difference × 1,000 (CKINN − M1)"),
    ]:
        ax.errorbar(
            s["size"],
            scale * s[f"mean_{key}_minus_M1"],
            yerr=scale * s[f"sd_{key}_minus_M1"],
            color=STYLE["CKINN"]["color"],
            marker="s",
            capsize=3,
        )
        ax.axhline(0, color="#888888", lw=1, linestyle=(0, (4, 3)))
        ax.set(xscale="log", xlabel="RICH training participants", ylabel=lab)
    save(fig, st.fig / "A" / "v4", "training_size")


def all_figures(st):
    for task in ["A", "R"]:
        try:
            st.frozen(task)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        for fn in (calibration, roc_pr, decision_curve, directions, shap_importance, lambda_path):
            fn(st, task)
    loso(st)
    size_curve(st)
