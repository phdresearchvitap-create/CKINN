from pathlib import Path
import copy, random, inspect, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit
from lifelines import CoxPHFitter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from data import CORE, PLUS, NONLAB, allowed, dump, restore, key, status
from stats import weights, brier

DIRECTIONS = dict(fpg=1, bmi=1, age=1, sbp=1, tg=1, hdl=-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class Prep:

    def __init__(self, names, task, masks=True):
        self.names = allowed(names, task)
        self.task = task
        self.masks = masks

    def values(self, d):
        x = d.loc[:, self.names].to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x

    def fit(self, d):
        x = self.values(d)
        if np.all(np.isnan(x), axis=0).any():
            raise ValueError("Training partition has an entirely missing feature")
        self.med = np.nanmedian(x, axis=0)
        filled = np.where(np.isnan(x), self.med, x)
        self.mu = filled.mean(0)
        self.scale = filled.std(0)
        self.scale[self.scale < 1e-12] = 1.0
        self.lo, self.hi = np.nanquantile(x, [0.01, 0.99], axis=0)
        return self

    def transform(self, d):
        x = self.values(d)
        obs = ~np.isnan(x)
        z = (np.where(obs, x, self.med) - self.mu) / self.scale
        features = np.column_stack([z, (~obs).astype(float)]) if self.masks else z
        return (features.astype("float32"), obs & (x >= self.lo) & (x <= self.hi))


class Net(nn.Module):

    def __init__(self, prep, h):
        super().__init__()
        names = prep.names
        width = h["branch"]
        self.groups = []
        width_in = len(names) * (1 + int(prep.masks))
        if h.get("topology", "branches") == "branches":
            for group in (
                ["age", "sex", "smoking"],
                ["bmi", "sbp", "dbp"],
                ["alt", "fpg", "tg", "hdl", "tc", "ast"],
            ):
                inds = [names.index(k) for k in group if k in names]
                self.groups.append(inds + [i + len(names) for i in inds] if prep.masks else inds)
            self.branches = nn.ModuleList(
                [nn.Sequential(nn.Linear(max(len(ix), 1), width), nn.SiLU()) for ix in self.groups]
            )
            width_in = 3 * width
        h1, h2 = h["fusion"]
        self.fusion = nn.Sequential(
            nn.Linear(width_in, h1),
            nn.SiLU(),
            nn.Dropout(h["dropout"]),
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        if self.groups:
            x = torch.cat(
                [
                    b(x[:, ix] if ix else x.new_zeros((len(x), 1)))
                    for ix, b in zip(self.groups, self.branches)
                ],
                dim=1,
            )
        return self.fusion(x).flatten()


def clinical(p, x, inside, names, directions):
    if not directions:
        return x.new_zeros(())
    idx = [names.index(k) for k in directions]
    signs = x.new_tensor([DIRECTIONS[k] for k in directions])
    g = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, idx]
    terms = torch.relu(-g * signs).square()
    ok = inside[:, idx]
    return (terms * ok).sum() / ok.sum().clamp_min(1)


def side_weights(d):
    sites = d.site.fillna("missing").astype(str)
    counts = sites.value_counts()
    pooled = sites.where(sites.map(counts) >= 50, "__sparse__")
    counts = pooled.value_counts()
    raw = np.minimum(3.0, len(d) / (len(counts) * pooled.map(counts).to_numpy(float)))
    return raw / raw.mean()


def loss_terms(logits, x, inside, y, w, sex, prep, h):
    loss = w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    total = loss.mean()
    if h["lambda"]:
        total = total + h["lambda"] * clinical(logits.sigmoid(), x, inside, prep.names, h["directions"])
    if h.get("group_penalty", 0):
        m = sex == 1
        f = sex == 0
        if m.any() and f.any():
            total = total + h["group_penalty"] * (loss[m].mean() - loss[f].mean()).square()
    return total


def torch_risk(net, x, device="cpu", batch=2048):
    net.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                net(torch.from_numpy(z).to(device)).sigmoid().cpu().numpy()
                for z in np.array_split(x, max(1, int(np.ceil(len(x) / batch))))
            ]
        ).astype(float)


def train_net(d, valid, prep, h, seed, epochs, patience, device, checkpoint, identity, interrupt_epoch=None):
    seed_all(seed)
    net = Net(prep, h).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=h["lr"], weight_decay=h["decay"])
    X, inside = prep.transform(d)
    y, w, _ = weights(d, h.get("site_g", False))
    if h.get("site_balance"):
        w *= side_weights(d)
    if valid is not None:
        xv = prep.transform(valid)[0]
        vy, vw, _ = weights(valid, h.get("site_g", False))
    state = dict(epoch=0, best=np.inf, best_epoch=0, best_state=None, wait=0)
    generator = np.random.default_rng(seed)
    if checkpoint and Path(checkpoint).exists():
        saved = restore(checkpoint)
        if saved["identity"] != identity:
            raise ValueError("Incompatible epoch checkpoint")
        state = saved["progress"]
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        generator.bit_generator.state = saved["shuffle_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(state["epoch"], epochs):
        if valid is not None and state["wait"] >= patience:
            break
        net.train()
        order = generator.permutation(len(d))
        for start in range(0, len(d), h["batch"]):
            i = order[start : start + h["batch"]]
            x = torch.from_numpy(X[i]).to(device).requires_grad_(bool(h["lambda"]))

            def tensor(a):
                return torch.as_tensor(a, device=device, dtype=torch.float32)

            logits = net(x)
            loss = loss_terms(
                logits,
                x,
                torch.from_numpy(inside[i]).to(device),
                tensor(y[i]),
                tensor(w[i]),
                tensor(d.sex.to_numpy()[i]),
                prep,
                h,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite neural loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        state["epoch"] = epoch + 1
        if valid is not None:
            score = brier(vy, vw, torch_risk(net, xv, device))
            if score < state["best"] - 1e-10:
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                state.update(best=score, best_epoch=epoch + 1, best_state=best_state, wait=0)
            else:
                state["wait"] += 1
        if checkpoint:
            cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            optstate = copy.deepcopy(opt.state_dict())
            for slots in optstate["state"].values():
                for k, v in slots.items():
                    if torch.is_tensor(v):
                        slots[k] = v.cpu()
            dump(
                checkpoint,
                dict(
                    identity=identity,
                    preprocessing=prep,
                    model=cpu_state,
                    optimizer=optstate,
                    scheduler=None,
                    progress=state,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    shuffle_rng=generator.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                ),
            )
        if interrupt_epoch == epoch + 1:
            raise InterruptedError("synthetic interruption")
    if valid is not None:
        net.load_state_dict(state["best_state"])
    return (net.cpu().eval(), state["best_epoch"] if valid is not None else state["epoch"])


def split_indices(d, k, seed):
    if not d.id.is_unique:
        raise ValueError("Repeated person or row key")
    base = pd.Series(status(d) if "time" in d else d.event.to_numpy(), dtype=str)
    if base.value_counts().min() < k:
        raise ValueError(f"Too few participants in a status stratum for {k} folds")
    joint = base + "/" + d.site.reset_index(drop=True).astype(str)
    strata = joint.copy()
    for label in base.unique():
        block = base == label
        counts = joint[block].value_counts()
        rare = block & joint.isin(counts[counts < k].index)
        if rare.any():
            strata.loc[rare] = label + "/pooled" if rare.sum() >= k else counts.idxmax()
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(d)), strata))


def fit_weighted(model, x, y, w):
    if "sample_weight" not in inspect.signature(model.fit).parameters:
        raise TypeError("Estimator lacks sample_weight")
    good = w > 0
    if np.unique(y[good]).size < 2:
        raise ValueError("Single known training class")
    model.fit(x[good], y[good], sample_weight=w[good])
    return model


def estimator(name, h, seed):
    if name == "LR":
        return LogisticRegression(C=h["C"], max_iter=500)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=h["trees"], max_depth=h["depth"], min_samples_leaf=8, n_jobs=1, random_state=seed
        )
    if name == "XGB":
        return XGBClassifier(
            n_estimators=h["trees"],
            max_depth=h["depth"],
            learning_rate=0.05,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
    if name == "LGB":
        return LGBMClassifier(
            n_estimators=h["trees"],
            num_leaves=2 ** h["depth"],
            max_depth=h["depth"],
            learning_rate=0.05,
            verbosity=-1,
            n_jobs=1,
            random_state=seed,
        )
    if name == "SVM":
        return LinearSVC(C=h["C"], dual=False, max_iter=3000, random_state=seed)
    raise ValueError(name)


class Fitted:

    def __init__(self, kind, prep, obj, extra=None):
        self.kind, self.prep, self.obj, self.extra = (kind, prep, obj, extra)

    def predict(self, d):
        x = self.prep.transform(d)[0]
        if self.kind == "net":
            return torch_risk(self.obj, x)
        if self.kind == "reference":
            return np.full(len(x), self.obj)
        if self.kind == "Cox":
            pos, H = self.extra
            hazard = self.obj.predict_partial_hazard(
                pd.DataFrame(x[:, pos], columns=[f"x{i}" for i in pos])
            ).to_numpy()
            return -np.expm1(-H * hazard)
        if self.kind == "SVM":
            return self.extra.predict_proba(self.obj.decision_function(x).reshape(-1, 1))[:, 1]
        return self.obj.predict_proba(x)[:, 1]


def fit_model(kind, d, valid, h, task, seed, epochs, patience, device, checkpoint, identity):
    pp = Prep(h["features"], task, h.get("masks", True)).fit(d)
    if kind == "net":
        model, ep = train_net(d, valid, pp, h, seed, epochs, patience, device, checkpoint, identity)
        return (Fitted(kind, pp, model), ep)
    x = pp.transform(d)[0]
    y, w, _ = weights(d)
    if kind == "reference":
        return (Fitted(kind, pp, np.average(y, weights=w)), 0)
    if kind == "Cox":
        pos = np.flatnonzero(x.std(0) > 1e-08)
        cd = pd.DataFrame(x[:, pos], columns=[f"x{i}" for i in pos])
        cd["T"] = np.minimum(d.time, 3).to_numpy()
        cd["E"] = ((d.time <= 3) & (d.event == 1)).to_numpy()
        if not cd.E.any():
            raise ValueError("No training detections by three")
        model = CoxPHFitter(penalizer=h["ridge"], l1_ratio=0.0).fit(cd, duration_col="T", event_col="E")
        H = float(model.baseline_cumulative_hazard_.iloc[-1, 0])
        return (Fitted(kind, pp, model, (pos, H)), 0)
    calibrator = None
    if kind == "SVM":
        scores = np.empty(len(d))
        wy = np.empty(len(d))
        folds = split_indices(d, 2, seed)
        if checkpoint:
            sp = Path(checkpoint).with_name("svm_calibration_folds.pkl")
            if sp.exists():
                saved = restore(sp)
                if saved["ids"] != d.id.tolist():
                    raise ValueError("SVM calibration partition mismatch")
                folds = saved["folds"]
            else:
                dump(sp, dict(ids=d.id.tolist(), folds=folds))
        for a, b in folds:
            train, val = (d.iloc[a], d.iloc[b])
            p = Prep(h["features"], task, h.get("masks", True)).fit(train)
            yt, wt, _ = weights(train)
            m = fit_weighted(estimator(kind, h, seed), p.transform(train)[0], yt, wt)
            scores[b] = m.decision_function(p.transform(val)[0])
            wy[b] = weights(val)[1]
        calibrator = fit_weighted(LogisticRegression(C=10000.0, max_iter=500), scores[:, None], y, wy)
    model = fit_weighted(estimator(kind, h, seed), x, y, w)
    return (Fitted(kind, pp, model, calibrator), 0)
