from pathlib import Path
from importlib.metadata import version
import json, platform, time, sys, traceback
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data import *
from stats import weights, metrics, brier, point_threshold, support
from models import fit_model, split_indices


def env_info():
    from importlib.metadata import PackageNotFoundError

    versions = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "lifelines",
        "lightgbm",
        "xgboost",
        "openpyxl",
        "matplotlib",
        "nbformat",
        "shap",
    ]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return dict(
        python=sys.version,
        platform=platform.platform(),
        packages=versions,
        cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
    )


def model_spec(model, base, task):
    h = dict(base)
    h.setdefault("features", CORE if task == "A" else NONLAB)
    h.setdefault("directions", ["fpg", "bmi"] if task == "A" else ["bmi"])
    if model in ("M1", "fixed_zero", "nonlab_M1"):
        h["lambda"] = 0.0
    changes = {
        "M4": dict(site_balance=True),
        "M5": dict(site_balance=True, group_penalty=0.25),
        "fpg_only": dict(directions=["fpg"]),
        "bmi_only": dict(directions=["bmi"]),
        "expanded_prior": dict(directions=["fpg", "bmi", "age", "sbp", "tg", "hdl"]),
        "expanded_inputs": dict(features=PLUS),
        "no_masks": dict(masks=False),
        "single_stream": dict(topology="single"),
        "nonlab_M1": dict(features=NONLAB),
        "site_ipcw": dict(site_g=True),
    }
    h.update(changes.get(model, {}))
    return h


MATCHED_PAIR = ["fixed_zero", "fixed_positive"]
PRIOR_VARIANTS = ["fpg_only", "bmi_only", "expanded_prior"]
OTHER_VARIANTS = ["M4", "M5", "expanded_inputs", "no_masks", "single_stream", "nonlab_M1"]
VARIANTS = MATCHED_PAIR + PRIOR_VARIANTS + OTHER_VARIANTS
CLASSIC = ["reference", "LR", "RF", "XGB", "LGB", "SVM", "Cox"]


class Study:

    def __init__(self, root, cfg, paths=None):
        self.root = Path(root)
        self.cfg = cfg
        if cfg["tau"] != 3:
            raise ValueError("This protocol defines a three-year endpoint")
        self.mode = cfg["profile"]
        self.b = cfg["profiles"][self.mode]
        self.seed = cfg["seed"]
        self.device = (
            cfg["device"] if cfg["device"] != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.paths = paths or discover(self.root / "data/raw")
        self.code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        self.identity = dict(
            data={k: digest(v) for k, v in self.paths.items()},
            code=key(self.code),
            config=key({k: v for k, v in cfg.items() if k != "stage"}),
            device=self.device,
            runtime={k: v for k, v in env_info().items() if k != "packages"},
            versions=env_info()["packages"],
        )
        self.run = key(self.identity)[:16]
        cache_runtime = [platform.python_version(), pd.__version__, np.__version__, version("openpyxl")]
        self.cache = (
            self.root
            / "data/processed"
            / key(
                [
                    self.identity["data"],
                    self.code["data.py"],
                    self.code["dictionary.py"],
                    cfg["options"]["screen_no_code"],
                    cache_runtime,
                ]
            )[:16]
        )
        self.auditpath = self.root / "results/tables/audit" / self.cache.name
        self.private = self.root / "checkpoints" / self.mode / self.run
        self.table = self.root / "results/tables" / self.mode / self.run
        self.fig = self.root / "results/figures" / self.mode / self.run
        self.report = self.root / "results/reports" / self.mode / self.run
        self.log = self.root / "logs" / self.mode / self.run
        for p in [self.cache, self.private, self.table, self.fig, self.report, self.log]:
            p.mkdir(parents=True, exist_ok=True)
        js(self.root / "config" / (self.run + ".json"), cfg)
        js(self.log / "environment.json", env_info())
        self.mem = {}

    def audit(self):
        done = self.cache / "audit_complete.json"
        if not done.exists():
            report = audit(self.paths, self.auditpath, self.cache, self.cfg["options"]["screen_no_code"])
            if not report.shape_matches_reference.all():
                raise ValueError("Shape discrepancy. Inspect audit before proceeding.")
            js(done, {"signatures": {p.name: digest(p) for p in self.cache.glob("*.pkl")}})
        return pd.read_csv(self.auditpath / "structure.csv")

    def data(self, name):
        if name not in self.mem:
            if not (self.cache / "audit_complete.json").exists():
                raise RuntimeError("Run audit first")
            path = self.cache / (name + ".pkl")
            if (
                digest(path)
                != json.loads((self.cache / "audit_complete.json").read_text())["signatures"][path.name]
            ):
                raise ValueError("Validated cache changed")
            d = restore(path)
            cap = self.b["sample"]
            if cap and len(d) > cap:
                s = status(d) if "time" in d else d.event
                i, _ = train_test_split(np.arange(len(d)), train_size=cap, stratify=s, random_state=self.seed)
                d = d.iloc[np.sort(i)]
            self.mem[name] = d.reset_index(drop=True)
        return self.mem[name]

    def sources(self, task):
        if task == "B":
            opt = self.cfg["options"]
            if opt["screening"] == "verified" and (not opt["producer_recode_evidence"].strip()):
                raise ValueError(
                    "Verified screening requires the producer's documented crosswalk, including unknown/refused"
                )
            if opt["screening"] not in ("as_coded", "verified"):
                raise ValueError("Screening disabled")
            return PERIODS
        return ["RICH", "NAGALA"]

    def tag(self, task):
        return task if task == "A" else "B_" + self.cfg["options"]["screening"]

    def split(self, d, k, label, seed):
        path = self.private / "splits" / (label + ".pkl")
        ids = d.id.tolist()
        if path.exists():
            saved = restore(path)
            if saved["ids"] != ids or saved["k"] != k:
                raise ValueError("Immutable partition mismatch")
            return saved["folds"]
        folds = split_indices(d, k, seed)
        for a, b in folds:
            assert not set(d.id.iloc[a]).intersection(d.id.iloc[b])
        dump(path, dict(ids=ids, k=k, seed=seed, folds=folds))
        return folds

    def candidates(self, name, task):
        grid = self.cfg["search"]
        neural = name not in CLASSIC
        n = self.b["trials"] if neural else 1 if name == "reference" else self.b["comparator_trials"]
        rng = np.random.default_rng(self.seed)
        for i in range(n):
            if neural:
                h = dict(
                    branch=8 if i == 0 else int(rng.choice(grid["branch"])),
                    fusion=grid["fusion"][1 if i == 0 else rng.integers(3)],
                    lr=0.001 if i == 0 else float(np.exp(rng.uniform(*np.log(grid["lr"])))),
                    dropout=0.1 if i == 0 else float(rng.choice(grid["dropout"])),
                    decay=float(rng.choice(grid["decay"])),
                    batch=int(rng.choice(grid["batch"])),
                )
                h["lambda"] = grid["lambda"][i % len(grid["lambda"])]
            else:
                h = dict(
                    C=grid["comparator_C"][i],
                    ridge=grid["ridge"][i],
                    trees=12 if self.mode == "SMOKE" else grid["trees"][i],
                    depth=grid["depth"][i],
                )
            yield model_spec(name, h, task)

    def record(self, name, entry):
        p = self.log / (name + ".jsonl")
        with p.open("a", encoding="utf8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def fit(self, task, name, tr, va, h, seed, ep, label):
        if self.mode == "AUDIT":
            raise ValueError("AUDIT disables training")
        if set(tr.cohort) != {self.sources(task)[0]}:
            raise ValueError("Source-only fit guard")
        if va is not None and (set(va.cohort) != set(tr.cohort) or set(va.id) & set(tr.id)):
            raise ValueError("Validation leakage")
        kind = name if name in CLASSIC else "net"
        ident = dict(
            **self.identity,
            model=name,
            task=task,
            fold=label,
            seed=seed,
            h=h,
            epochs=ep,
            train_keys=key(tr.id.tolist()),
            validation_keys=key(va.id.tolist()) if va is not None else None,
        )
        folder = self.private / "fits" / task / label / name / key(ident)[:16]
        final = folder / "complete.pkl"
        if final.exists():
            return restore(final)
        st = time.perf_counter()
        try:
            model, n = fit_model(
                kind, tr, va, h, task, seed, ep, self.b["patience"], self.device, folder / "epoch.pkl", ident
            )
            res = (model, n)
            dump(final, res)
            self.record(
                "fits",
                dict(
                    task=task,
                    model=name,
                    fold=label,
                    seed=seed,
                    epochs=n,
                    seconds=time.perf_counter() - st,
                    status="complete",
                    config=h,
                    parameters=sum((p.numel() for p in model.obj.parameters())) if kind == "net" else None,
                ),
            )
            return res
        except Exception as e:
            self.record(
                "fits",
                dict(
                    task=task,
                    model=name,
                    fold=label,
                    status="failed",
                    seconds=time.perf_counter() - st,
                    error=repr(e),
                    config=h,
                ),
            )
            raise

    def tune(self, task, name, d, label, fixed=None, epochs=None, grid=None):
        split = self.split(d, self.b["inner"], label + "_inner", self.seed + 11)
        if grid is not None:
            candidates = grid
        elif fixed is not None:
            candidates = [fixed]
        else:
            candidates = self.candidates(name, task)
        best = None
        for trial, h in enumerate(candidates):
            preds = np.full(len(d), np.nan)
            used = []
            scores = []
            allw = np.zeros(len(d))
            ally = np.zeros(len(d))
            for i, (a, b) in enumerate(split):
                tr, va = (d.iloc[a], d.iloc[b])
                try:
                    model, n = self.fit(
                        task,
                        name,
                        tr,
                        va if epochs is None else None,
                        h,
                        self.seed,
                        epochs or self.b["epochs"],
                        f"{label}_c{trial}_i{i}",
                    )
                    preds[b] = model.predict(va)
                    y, w, _ = weights(va, h.get("site_g", False))
                    allw[b], ally[b] = (w, y)
                    scores.append(brier(y, w, preds[b]) * len(b))
                    used.append(n)
                except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as e:
                    self.record(
                        "candidates",
                        dict(model=name, label=label, trial=trial, inner=i, status="failed", error=repr(e)),
                    )
                    break
            if np.isfinite(preds).all():
                score = sum(scores) / len(d)
                refit_epochs = epochs if epochs is not None else max(1, round(np.median(used)))
                config = dict(
                    h=h,
                    epochs=refit_epochs,
                    score=score,
                    threshold=point_threshold(ally, allw, preds),
                    trial=trial,
                )
                self.record("candidates", dict(model=name, label=label, status="scored", **config))
                if best is None or score < best["score"]:
                    best = config
        if best is None:
            raise RuntimeError(f"All candidates failed for {name} / {label}; see logs")
        return best

    def plan(self, task, names, d, label, variants=False):
        p = self.private / "plans" / (label + ".pkl")
        if p.exists():
            return restore(p)
        plans = {}
        for name in names:
            plans[name] = self.tune(task, name, d, label)
        if variants:
            anchor = plans["M3"]
            base = anchor["h"]
            epochs = anchor["epochs"]
            positive_lambdas = sorted({v for v in self.cfg["search"]["lambda"] if v > 0})
            if not positive_lambdas:
                raise ValueError("Matched penalty comparison requires a strictly positive lambda candidate")
            plans["fixed_zero"] = self.tune(
                task, "fixed_zero", d, label, fixed=dict(base, **{"lambda": 0.0}), epochs=epochs
            )
            grid = [dict(base, **{"lambda": v}) for v in positive_lambdas]
            plans["fixed_positive"] = self.tune(task, "fixed_positive", d, label, grid=grid, epochs=epochs)
            for name in PRIOR_VARIANTS + OTHER_VARIANTS:
                ref = plans["M1" if name == "nonlab_M1" else "M3"]
                if name in PRIOR_VARIANTS:
                    ref = plans["fixed_positive"]
                h = model_spec(name, ref["h"], task)
                plans[name] = self.tune(task, name, d, label, fixed=h, epochs=ref["epochs"])
        dump(p, plans)
        return plans

    def model_names(self, task):
        return [n for n in self.cfg["models"] if task == "A" or n != "Cox"]

    def internal(self, task="A", source=None, site=False):
        src = source or self.sources(task)[0]
        d = self.data(src)
        tag = self.tag(task)
        label = tag + "_" + src + ("_site" if site else "_nested")
        if site:
            splits = []
            for g in sorted(d.site.unique()):
                test = np.flatnonzero(d.site.to_numpy() == g)
                train = np.flatnonzero(d.site.to_numpy() != g)
                splits.append((train, test))
            sp = self.private / "splits" / (label + ".pkl")
            if not sp.exists():
                dump(sp, dict(ids=d.id.tolist(), folds=splits))
            else:
                splits = restore(sp)["folds"]
        else:
            splits = self.split(d, self.b["outer"], label, self.seed)
        variants = task == "A" and src == "RICH" and (not site) and self.cfg["options"]["ablations"]
        rows = []
        for i, (a, b) in enumerate(splits):
            out = self.private / "oof" / (label + f"_o{i}.pkl")
            if out.exists():
                rows += restore(out)
                continue
            tr, va = (d.iloc[a], d.iloc[b])
            names = self.model_names(task) if src == "RICH" and (not site) else ["M1", "M3"]
            lab = label + f"_o{i}"
            if site and (
                (status(va) == 1).sum() < self.cfg["min_cases"]
                or (status(va) == 0).sum() < self.cfg["min_controls"]
            ):
                rows.append(
                    dict(
                        fold=i,
                        site=str(va.site.iloc[0]),
                        model="all",
                        status="insufficient support",
                        n=len(b),
                        events=int((status(va) == 1).sum()),
                    )
                )
                continue
            print(f"{label}: fitting fold {i + 1}/{len(splits)}", flush=True)
            plans = self.plan(task, names, tr, lab, variants)
            result = []
            if src == "RICH" and self.cfg["options"]["site_ipcw"] and (not site):
                ref = plans["M3"]
                h = model_spec("site_ipcw", ref["h"], task)
                try:
                    plans["site_ipcw"] = dict(
                        self.tune(task, "site_ipcw", tr, lab, fixed=h, epochs=ref["epochs"]),
                        epochs=ref["epochs"],
                    )
                except RuntimeError as exc:
                    if not str(exc).startswith("All candidates failed"):
                        raise
                    self.record(
                        "diagnostics",
                        dict(task=task, fold=i, model="site_ipcw", status="unavailable", error=str(exc)),
                    )
                    result.append(dict(fold=i, model="site_ipcw", status="unavailable; see candidate log"))
            for n, plan in plans.items():
                model, _ = self.fit(task, n, tr, None, plan["h"], self.seed, plan["epochs"], lab + "_refit")
                start = time.perf_counter()
                pred = model.predict(va)
                seconds = time.perf_counter() - start
                dump(
                    self.private / "predictions" / lab / (n + ".pkl"),
                    dict(id=va.id.tolist(), p=pred, threshold=plan["threshold"]),
                )
                result.append(
                    dict(
                        fold=i,
                        model=n,
                        site=str(va.site.iloc[0]) if site else "mixed",
                        status="complete",
                        scoring_seconds=seconds,
                        **metrics(va, pred, plan["threshold"]),
                    )
                )
                if "site_ipcw" in plans and n in ("M3", "site_ipcw"):
                    result.append(
                        dict(
                            fold=i,
                            model=n + "_siteG_metric",
                            site="mixed",
                            status="complete",
                            scoring_seconds=seconds,
                            **metrics(va, pred, plan["threshold"], True),
                        )
                    )
                if n == "M1" and variants:
                    result.append(dict(result[-1], model="M2"))
            dump(out, result)
            rows += result
        csv(self.table / tag / (label + ".csv"), rows)
        return pd.DataFrame(rows)

    def final_plan(self, task):
        source = self.sources(task)[0]
        d = self.data(source)
        return self.plan(
            task,
            self.model_names(task),
            d,
            self.tag(task) + "_final",
            task == "A" and self.cfg["options"]["ablations"],
        )

    def freeze(self, task):
        tag = self.tag(task)
        directory = self.private / "frozen" / tag
        manifest = directory / "manifest.json"
        if manifest.exists():
            self.frozen(task)
            return manifest
        plans = self.final_plan(task)
        d = self.data(self.sources(task)[0])
        files = {}
        for name, c in plans.items():
            for seed in self.cfg["final_seed_pool"][: self.b["final_seeds"] if name not in CLASSIC else 1]:
                model, _ = self.fit(task, name, d, None, c["h"], seed, c["epochs"], tag + "_final_refit")
                p = directory / f"{name}_{seed}.pkl"
                dump(p, model)
                files[p.name] = digest(p)
        js(
            manifest,
            dict(
                identity=self.identity,
                source=self.sources(task)[0],
                plans=plans,
                files=files,
                ensemble="equal arithmetic mean of probabilities",
            ),
        )
        return manifest

    def frozen(self, task):
        folder = self.private / "frozen" / self.tag(task)
        m = json.loads((folder / "manifest.json").read_text())
        if m["identity"] != self.identity or any((digest(folder / n) != h for n, h in m["files"].items())):
            raise ValueError("Frozen identity mismatch")
        return (folder, m)

    def selection_table(self, task, frozen=False):
        plans = self.frozen(task)[1]["plans"] if frozen else self.final_plan(task)
        definitions = {
            "M1": "unpenalized primary baseline",
            "M3": "primary model, zero eligible",
            "fixed_zero": "matched M3 architecture, zero penalty",
            "fixed_positive": "matched M3 architecture, strictly positive penalty",
        }
        rows = []
        for name, plan in plans.items():
            neural = name not in CLASSIC
            base = plan["h"]
            rows.append(
                dict(
                    model=name,
                    lambda_value=base["lambda"] if neural else None,
                    source_cv_brier=plan["score"],
                    threshold=plan["threshold"],
                    candidate_index=plan["trial"],
                    epochs=plan["epochs"] if neural else None,
                    role=definitions.get(name, ""),
                    settings=json.dumps(base, sort_keys=True),
                )
            )
        frame = pd.DataFrame(rows)
        csv(self.table / self.tag(task) / "source_selection.csv", frame)
        return frame

    def score(self, task):
        tag = self.tag(task)
        folder, m = self.frozen(task)
        d = self.data(self.sources(task)[1])
        out = {}
        rows = []
        seeds = []
        for n, plan in m["plans"].items():
            files = [f for f in m["files"] if f.rsplit("_", 1)[0] == n]
            st = time.perf_counter()
            each = np.stack([restore(folder / f).predict(d) for f in files])
            p = each.mean(0)
            seconds = time.perf_counter() - st
            out[n] = p
            rows.append(dict(model=n, scoring_seconds=seconds, **metrics(d, p, plan["threshold"])))
            for f, pr in zip(files, each):
                seeds.append(dict(model=n, seedfile=f, **metrics(d, pr, plan["threshold"])))
        out["M2"] = out["M1"].copy()
        assert np.array_equal(out["M2"], out["M1"])
        rows.append(dict(next((x for x in rows if x["model"] == "M1")), model="M2"))
        dump(
            self.private / "predictions" / (tag + "_frozen.pkl"),
            dict(id=d.id.tolist(), p=out, manifest=digest(folder / "manifest.json")),
        )
        csv(self.table / tag / "external_metrics.csv", rows)
        csv(self.table / tag / "seed_metrics.csv", seeds)
        return pd.DataFrame(rows)

    def results(self, task):
        folder, m = self.frozen(task)
        d = self.data(self.sources(task)[1])
        obj = restore(self.private / "predictions" / (self.tag(task) + "_frozen.pkl"))
        if d.id.tolist() != obj["id"] or obj["manifest"] != digest(folder / "manifest.json"):
            raise ValueError("Frozen scoring mismatch")
        return (d, obj["p"], m)
