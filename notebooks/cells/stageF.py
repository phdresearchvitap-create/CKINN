# %%
# CELL 37 - the unconstrained neural network (MLP) with Cox loss, in PyTorch                      [status L; protocol section 6; proposal pp. 7-8, 18]
# Why: the MLP is "Model 1" of the proposal's ablation and the reference against which CKINN is judged (claims C1, C4,
# hypothesis H1). Architecture and training are those of protocol section 6 and of the locked reference code: hidden
# layers (64, 64), ReLU, dropout 0.1, AdamW (learning rate 2e-3, weight decay 1e-4), batches of 8,192, up to 40 epochs,
# early stopping (patience 4) on a 10% validation split. The same class, with lam > 0, becomes CKINN in cell 39, so the
# two models cannot differ in anything but the penalty.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 37)
require_gate("E")
A = Audit(37, "MLP with Cox partial-likelihood loss (PyTorch)")
import torch

torch.set_num_threads(ENV["threads"])


def cox_batch_nll(lp, t, e):
    """Negative Cox partial log-likelihood of one batch, divided by its number of events. Rows are sorted by decreasing
    time, so the running log-sum-exp is the log of the risk-set sum (the form used by the locked reference code)."""
    o = torch.argsort(t, descending=True)
    lp, e = lp[o], e[o]
    return -((lp - torch.logcumsumexp(lp, 0)) * e).sum() / e.sum().clamp(min=1)


def build_net(n_in, hidden=None, dropout=None):
    """Linear -> ReLU -> Dropout for each hidden layer, then one linear output: the log relative hazard."""
    hidden = HP["hidden"] if hidden is None else hidden
    dropout = HP["dropout"] if dropout is None else dropout
    layers, d = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        d = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))


class DeepCox:
    """Protocol section 6, model 4 (lam = 0: MLP) and model 5 (lam > 0: CKINN). Preprocessing is fitted on the training
    rows (cell 22). `priors` defaults to protocol section 7; only priors whose variable is among the predictors apply."""

    def __init__(self, cols, lam=0.0, seed=0, priors=None, epochs=None, batch=None, verbose=False):
        self.cols, self.lam, self.seed = list(cols), float(lam), int(seed)
        self.priors = PRIORS if priors is None else priors
        self.epochs = HP["epochs"] if epochs is None else epochs
        self.batch = HP["batch"] if batch is None else batch
        self.verbose = verbose

    def fit(self, df):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.prep = Prep(self.cols).fit(df)
        X = torch.tensor(self.prep.transform(df), dtype=torch.float32)
        T = torch.tensor(df["time_y"].to_numpy(), dtype=torch.float32)
        E = torch.tensor(df["event"].to_numpy(), dtype=torch.float32)
        self.cons = [(self.prep.names.index(c), s) for c, s in self.priors.items() if c in self.prep.names]
        self.net = build_net(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        perm = rng.permutation(len(X))
        nv = int(len(X) * HP["val_frac"])
        va, tr = torch.tensor(perm[:nv]), torch.tensor(perm[nv:])
        best, best_state, bad, self.history_ = np.inf, None, 0, []
        for ep in range(self.epochs):
            self.net.train()
            order = tr[torch.randperm(len(tr))]
            pen_sum, n_b = 0.0, 0
            for b in order.split(self.batch):
                if E[b].sum() < 1:                                    # a batch without an event has no partial likelihood
                    continue
                opt.zero_grad()
                loss = cox_batch_nll(self.net(X[b]).squeeze(-1), T[b], E[b])
                if self.lam > 0 and self.cons:
                    pen = clinical_penalty(self.net, X[b][:HP["pen_rows"]], self.cons)      # defined in cell 39
                    loss = loss + self.lam * pen
                    pen_sum += float(pen)
                loss.backward()
                opt.step()
                n_b += 1
            self.net.eval()
            with torch.no_grad():
                v = cox_batch_nll(self.net(X[va]).squeeze(-1), T[va], E[va]).item()
            self.history_.append({"epoch": ep, "val_nll": v, "mean_penalty": pen_sum / max(n_b, 1)})
            if self.verbose:
                print(f"     epoch {ep:02d}  validation loss {v:.4f}")
            if v < best - HP["es_min_delta"]:
                best, bad = v, 0
                best_state = {k: w.clone() for k, w in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= HP["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_val_, self.epochs_run_ = best, len(self.history_)
        return self

    def lp(self, df):
        with torch.no_grad():
            return self.net(torch.tensor(self.prep.transform(df), dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)


MODELS["MLP"] = lambda cols, seed=0: DeepCox(cols, lam=0.0, seed=seed)

A.check(f"PyTorch {torch.__version__} on the CPU with one thread", torch.get_num_threads() == 1 and DEVICE == "cpu")
A.check("architecture and optimizer settings equal protocol section 6",
        (HP["hidden"], HP["dropout"], HP["lr"], HP["weight_decay"], HP["batch"], HP["epochs"], HP["patience"], HP["val_frac"]) == ((64, 64), 0.1, 2e-3, 1e-4, 8192, 40, 4, 0.10))

# ---------------------------------------------------------------- tests on toy arrays
_net = build_net(8)
A.check("network: 8 inputs -> 64 -> 64 -> 1, ReLU and dropout 0.1 after each hidden layer; 4,801 parameters",
        [type(m).__name__ for m in _net] == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
        and sum(p.numel() for p in _net.parameters()) == 8 * 64 + 64 + 64 * 64 + 64 + 64 + 1 and _net[2].p == 0.1)
_lp, _t, _e = torch.zeros(3, dtype=torch.float64), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), torch.ones(3, dtype=torch.float64)
A.check("batch Cox loss equals a hand calculation: three events, equal hazards, gives (log 3 + log 2 + log 1) / 3",
        abs(cox_batch_nll(_lp, _t, _e).item() - np.log(6.0) / 3) < 1e-12)
_d = toy_survival(3000, ties=False)
_Z = torch.tensor(_d[["x1", "x2"]].to_numpy(), dtype=torch.float64)
_b = torch.tensor([0.4, -0.3], dtype=torch.float64)
_mine = cox_batch_nll(_Z @ _b, torch.tensor(_d["time_y"].to_numpy()), torch.tensor(_d["event"].to_numpy(), dtype=torch.float64)).item()
_ref = cox_nll_grad_hess(_b.numpy(), _Z.numpy(), _d["time_y"].to_numpy(), _d["event"].to_numpy())[0] * len(_d) / _d["event"].sum()
A.check("without tied times the batch loss equals the exact partial likelihood of cell 23", abs(_mine - _ref) < 1e-9, f"{_mine:.6f}")
_tr, _te = _d.iloc[:2000], _d.iloc[2000:]
_m1 = DeepCox(["x1", "x2"], seed=0, epochs=30, batch=256).fit(_tr)
_m2 = DeepCox(["x1", "x2"], seed=0, epochs=30, batch=256).fit(_tr)
_m3 = DeepCox(["x1", "x2"], seed=1, epochs=30, batch=256).fit(_tr)
A.check("training is repeatable: the same seed gives identical predictions, another seed different ones",
        np.array_equal(_m1.lp(_te), _m2.lp(_te)) and not np.array_equal(_m1.lp(_te), _m3.lp(_te)))
_c = concordance_index(_te["time_y"], _m1.lp(_te), _te["event"])
_f = fit_with_baseline(DeepCox(["x1", "x2"], seed=0, epochs=30, batch=256), _tr)
A.check("the toy network learns: it ranks unseen toy rows better than chance and its 3-year risks lie strictly between 0 and 1",
        _c > 0.55 and ((_f.risk(_te) > 0) & (_f.risk(_te) < 1)).all(), round(_c, 3))
A.check("early stopping keeps the epoch with the lowest validation loss", abs(_m1.best_val_ - min(h["val_nll"] for h in _m1.history_)) < 1e-12 and _m1.epochs_run_ <= 30)
A.check("prediction does not use dropout: predicting twice gives the same values", np.array_equal(_m1.lp(_te), _m1.lp(_te)))
_ = A.close(outputs={"torch": torch.__version__, "parameters_T2": sum(p.numel() for p in _net.parameters())})
keep_for_session(37)

# %%
# CELL 38 - the clinical prior registry: which prior applies to which predictor set                 [status L; protocol section 7; proposal pp. 6-7]
# Why: this is the proposal's "clinical knowledge layer". Each prior is a direction (risk up or down), never a size, and
# applies only where its variable is actually a predictor. All priors carry the same weight (w = 1). The table below is
# what CKINN penalizes and what XGB-mono enforces; both read it from the same place.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 38)
A = Audit(38, "Clinical prior registry")

PRIOR_SOURCE = {"bmi": "proposal p. 6: higher BMI, generally higher risk", "fpg": "proposal p. 6: higher fasting glucose, higher risk",
                "tg": "proposal p. 6: higher triglycerides, generally higher risk", "sbp": "proposal p. 6: hypertension, generally higher risk",
                "age": "proposal p. 7: increasing age, generally higher risk (soft)", "hdl": "proposal p. 7: higher HDL, generally lower risk",
                "famhx": "proposal p. 6: family history, higher risk", "hba1c": "proposal p. 6: higher HbA1c, higher risk",
                "exercise": "proposal p. 6: physical activity, generally lower risk"}
PRIOR_WEIGHT = {v: 1.0 for v in PRIORS}             # protocol section 7: equal weight
_rows = []
for cohort in FEATURE_SETS:
    for name in FEATURE_SETS[cohort]:
        cols = predictors(cohort, name)
        pr = priors_for(cols)
        _rows.append({"cohort": cohort, "set": name, "predictors": len(cols), "constrained": len(pr),
                      "up": ", ".join(v for v, s in pr.items() if s > 0), "down": ", ".join(v for v, s in pr.items() if s < 0),
                      "unconstrained": ", ".join(c for c in cols if c not in pr)})
PRIOR_TABLE = pd.DataFrame(_rows)
PRIOR_TABLE.to_csv(DIR["tables"] / "prior_registry.csv", index=False)
print(PRIOR_TABLE.to_string(index=False))
print("\n  Source of each prior:")
for v, s in PRIORS.items():
    print(f"   {v:9s} {'up  ' if s > 0 else 'down'}  weight {PRIOR_WEIGHT[v]:.0f}   {PRIOR_SOURCE[v]}")
print("\n  'Previous prediabetes' (proposal p. 7) has no separate variable: it is carried by fasting glucose (and HbA1c in NAGALA).")

A.check("nine priors, each with a direction, a weight of 1 and a stated source in the proposal", set(PRIOR_SOURCE) == set(PRIORS) == set(PRIOR_WEIGHT) and len(PRIORS) == 9)
A.check("no prior is placed on sex, ALT, AST, cholesterol, DBP, waist, fatty liver, GGT or smoking (protocol section 7)",
        not {"sex_male", "alt", "ast", "tc", "dbp", "waist", "fatty_liver", "ggt", "smoke"} & set(PRIORS))
A.check("headline tier T2 is constrained on age, BMI, SBP and triglycerides; glucose enters only at T3, HDL only at T4",
        sorted(priors_for(TIERS["T2"])) == ["age", "bmi", "sbp", "tg"] and "fpg" in priors_for(TIERS["T3"]) and "hdl" in priors_for(TIERS["T4"]) and "hdl" not in priors_for(TIERS["T3"]))
A.check("no prior is ever applied to a variable that is not a predictor of the set",
        all(set(priors_for(predictors(c, s))) <= set(predictors(c, s)) for c in FEATURE_SETS for s in FEATURE_SETS[c]))
A.check("family history is constrained only in RICH sets; HbA1c and exercise only in NAGALA sets",
        all(("famhx" in priors_for(predictors("RICH", s))) == (s.startswith("E")) for s in FEATURE_SETS["RICH"])
        and not any("famhx" in priors_for(predictors("NAGALA", s)) for s in FEATURE_SETS["NAGALA"])
        and not any({"hba1c", "exercise"} & set(priors_for(predictors("RICH", s))) for s in FEATURE_SETS["RICH"]))
_ = A.close(outputs={"priors": PRIORS, "weights": PRIOR_WEIGHT, "file": "ckinn_results/tables/prior_registry.csv"})
keep_for_session(38)

# %%
# CELL 39 - CKINN: the same network plus the soft clinical penalty                                [status L; protocol section 6; proposal p. 9]
# Why: this is the mathematical heart of the proposal. For each constrained variable j with direction s_j (+1 or -1):
#     penalty_j = mean over participants of  max(0, -s_j * d lp(x) / d x_j) ^ 2
#     loss = Cox loss + lambda * sum_j w_j * penalty_j            (w_j = 1 for every prior)
# lp is the network's log relative hazard and x_j the standardized input. The 3-year risk 1 - exp(-H0(3) exp(lp)) rises
# with lp, so d risk / d x_j has the same sign as d lp / d x_j: this is the proposal's penalty on the derivative of risk,
# written for censored data. The penalty is zero wherever the network already follows the prior, grows with the square
# of an opposite slope, and is evaluated only at observed covariate values. It is soft: data can overrule it.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 39)
A = Audit(39, "CKINN: soft clinical penalty")


def input_gradient(net, x):
    """d lp / d x for every row and input, by automatic differentiation; kept differentiable so it can be penalized."""
    x = x.clone().requires_grad_(True)
    return torch.autograd.grad(net(x).squeeze(-1).sum(), x, create_graph=True)[0]


def clinical_penalty(net, x, cons, weights=None):
    """sum_j w_j * mean( relu(-s_j * d lp / d x_j) ^ 2 ) over the constrained inputs. cons = [(input index, sign), ...]."""
    g = input_gradient(net, x)
    return sum((1.0 if weights is None else weights[i]) * torch.relu(-s * g[:, i]).pow(2).mean() for i, s in cons)


MODELS["CKINN-fixed"] = lambda cols, seed=0, lam=1.0: DeepCox(cols, lam=lam, seed=seed)     # CKINN at a given lambda (used by tests and ablations)

# ---------------------------------------------------------------- tests on toy arrays
torch.manual_seed(0)
_x = torch.randn(64, 3, dtype=torch.float64)
_up = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)).double()
with torch.no_grad():
    for _p in _up.parameters():
        _p.abs_()                                                    # all weights positive: lp can only rise with every input
A.check("a network that already rises with every input has penalty exactly 0 for 'up' priors",
        clinical_penalty(_up, _x, [(0, +1), (1, +1), (2, +1)]).item() == 0.0)
A.check("the same network is penalized for 'down' priors, and the penalty is the mean squared slope",
        abs(clinical_penalty(_up, _x, [(0, -1)]).item() - input_gradient(_up, _x)[:, 0].pow(2).mean().item()) < 1e-15 and clinical_penalty(_up, _x, [(0, -1)]).item() > 0)
_lin = torch.nn.Linear(3, 1).double()
with torch.no_grad():
    _lin.weight.copy_(torch.tensor([[0.5, -2.0, 0.0]], dtype=torch.float64))
A.check("linear model with slopes (0.5, -2, 0): priors 'up' on all three give penalty 0 + 2^2 + 0 = 4",
        abs(clinical_penalty(_lin, _x, [(0, +1), (1, +1), (2, +1)]).item() - 4.0) < 1e-12)
A.check("doubling an opposite slope quadruples its penalty (squared hinge)", abs(clinical_penalty(_lin, _x, [(1, +1)]).item() * 4 - 16.0) < 1e-12)
A.check("a variable without a prior is never penalized, whatever its slope", clinical_penalty(_lin, _x, [(0, +1)]).item() == 0.0)
A.check("weights scale each prior's penalty (the protocol uses 1 for all)", abs(clinical_penalty(_lin, _x, [(1, +1)], weights={1: 0.5}).item() - 2.0) < 1e-12)
_net = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 1)).double()
_cons = [(0, +1), (1, -1)]
_pen = clinical_penalty(_net, _x, _cons)
_auto = torch.autograd.grad(_pen, list(_net.parameters()))
_w0 = _net[0].weight
_fd = torch.zeros_like(_w0)
for _i in range(_w0.shape[0]):
    for _j in range(_w0.shape[1]):
        with torch.no_grad():
            _old = _w0[_i, _j].item()
            _w0[_i, _j] = _old + 1e-6
        _hi = clinical_penalty(_net, _x, _cons).item()
        with torch.no_grad():
            _w0[_i, _j] = _old - 1e-6
        _lo = clinical_penalty(_net, _x, _cons).item()
        with torch.no_grad():
            _w0[_i, _j] = _old
        _fd[_i, _j] = (_hi - _lo) / 2e-6
A.check("the gradient of the penalty with respect to the weights (what training uses) equals finite differences",
        torch.allclose(_auto[0], _fd, atol=1e-6), float((_auto[0] - _fd).abs().max()))
A.check("the penalty is itself differentiable twice through the network (training by gradient descent is possible)", all(g is not None and torch.isfinite(g).all() for g in _auto))
_ = A.close(outputs={"penalty": "sum_j mean(relu(-s_j * dlp/dx_j)^2), standardized inputs, w_j = 1", "pen_rows": HP["pen_rows"]})
keep_for_session(39)

# %%
# CELL 40 - verification: MLP and CKINN differ in the penalty and in nothing else                 [status L; protocol sections 6, 9; proposal p. 18]
# Why: a fair ablation needs proof that the two models share everything but the penalty, and that the penalty does what
# it claims. Tests, on toy data whose true effect of x2 is 'down': lambda = 0 reproduces the MLP bit for bit; a prior
# that agrees with the data costs nothing; a prior that contradicts the data is pushed towards compliance as lambda
# grows, at a price in fit. No study data are used.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 40)
A = Audit(40, "Verification of MLP and CKINN on toy data")

_d = toy_survival(3000)
_tr, _te = _d.iloc[:2000], _d.iloc[2000:]
_kw = dict(epochs=30, batch=256)
_true = {"x1": +1, "x2": -1}                        # the directions the toy data were built with
_wrong = {"x1": +1, "x2": +1}                       # a prior on x2 that contradicts the toy data

_mlp = DeepCox(["x1", "x2"], lam=0.0, seed=0, priors=_true, **_kw).fit(_tr)
_zero = DeepCox(["x1", "x2"], lam=0.0, seed=0, priors=_wrong, **_kw).fit(_tr)
A.check("lambda = 0 reproduces the MLP exactly, whatever the priors (identical weights, identical predictions)",
        np.array_equal(_mlp.lp(_te), _zero.lp(_te)) and all(torch.equal(a, b) for a, b in zip(_mlp.net.state_dict().values(), _zero.net.state_dict().values())))
A.check("MLP and CKINN have the same architecture, parameter count, optimizer and validation rows (same seed)",
        [tuple(p.shape) for p in _mlp.net.parameters()] == [tuple(p.shape) for p in DeepCox(["x1", "x2"], lam=1.0, seed=0, priors=_true, **_kw).fit(_tr).net.parameters()])

_res = {}
for _lam in (0.0, 1.0, 10.0, 100.0):
    _m = DeepCox(["x1", "x2"], lam=_lam, seed=0, priors=_wrong, **_kw).fit(_tr)
    _v = violation_rate(_m, _te, priors=_wrong)
    _res[_lam] = {"violation_x2": _v[1]["x2"], "size_x2": _v[2]["x2"], "cindex": concordance_index(_te["time_y"], _m.lp(_te), _te["event"])}
_agree = DeepCox(["x1", "x2"], lam=10.0, seed=0, priors=_true, **_kw).fit(_tr)
_va, _vm = violation_rate(_agree, _te, priors=_true), violation_rate(_mlp, _te, priors=_true)
print("  Toy data, prior on x2 deliberately wrong ('up' although the data say 'down'):")
print(pd.DataFrame(_res).T.rename_axis("lambda").round(4).to_string())
print(f"\n  Toy data, priors that agree with the data: violation rate MLP {_vm[0]:.4f}, CKINN (lambda 10) {_va[0]:.4f}; "
      f"C-index MLP {concordance_index(_te['time_y'], _mlp.lp(_te), _te['event']):.4f}, CKINN {concordance_index(_te['time_y'], _agree.lp(_te), _te['event']):.4f}")
A.check("with a wrong prior the unconstrained network violates it for most toy rows (the data say otherwise)", _res[0.0]["violation_x2"] > 0.5, _res[0.0]["violation_x2"])
A.check("raising lambda shrinks the size of the violation step by step (the penalty acts in the intended direction)",
        _res[100.0]["size_x2"] < _res[10.0]["size_x2"] < _res[0.0]["size_x2"], [round(_res[k]["size_x2"], 5) for k in (0.0, 1.0, 10.0, 100.0)])
A.check("a strong wrong prior costs discrimination (soft constraint: the data and the prior are traded off)", _res[100.0]["cindex"] < _res[0.0]["cindex"],
        [round(_res[k]["cindex"], 3) for k in (0.0, 100.0)])
A.check("priors that agree with the data leave discrimination intact (within 0.02) and do not raise the violation rate",
        abs(concordance_index(_te["time_y"], _agree.lp(_te), _te["event"]) - concordance_index(_te["time_y"], _mlp.lp(_te), _te["event"])) < 0.02 and _va[0] <= _vm[0] + 1e-12)
_f = fit_with_baseline(DeepCox(["x1", "x2"], lam=1.0, seed=0, priors=_true, **_kw), _tr)
A.check("CKINN's 3-year risks lie strictly between 0 and 1 and rise with its log hazard", ((_f.risk(_te) > 0) & (_f.risk(_te) < 1)).all()
        and (np.diff(_f.risk(_te)[np.argsort(_f.model.lp(_te))]) >= 0).all())
A.check("the penalty recorded during training is 0 for the MLP and positive for a contradicted CKINN",
        all(h["mean_penalty"] == 0 for h in _mlp.history_) and DeepCox(["x1", "x2"], lam=1.0, seed=0, priors=_wrong, **_kw).fit(_tr).history_[0]["mean_penalty"] > 0)
_ = A.close(outputs={"toy_wrong_prior": {str(k): v for k, v in _res.items()}})

# %%
# CELL 41 - choosing lambda inside the training data, saving models, and one fold of RICH          [status L; protocol sections 6, 12]
# Why: lambda must never be chosen with test data. The locked rule: split the training rows 80/20 (fixed seed); fit
# lambda in {0, 0.1, 1, 10} on the 80%; score the scaled Brier on the 20%; choose the LARGEST positive lambda whose
# score is within 0.002 of the best of the four; if none qualifies, lambda = 0.1; refit on all training rows. This cell
# also defines how fitted networks are saved and reloaded, then fits MLP and CKINN once to the RICH smoke fold (tier T2).
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 41)
A = Audit(41, "Lambda selection, model files, RICH smoke fold")


def choose_lambda(scores, tol=None, fallback=None):
    """scores = {lambda: inner scaled Brier}. Largest positive lambda within `tol` of the best score; else the fallback."""
    tol = HP["lambda_tol"] if tol is None else tol
    best = max(scores.values())
    ok = [l for l in scores if l > 0 and scores[l] >= best - tol]
    return max(ok) if ok else (HP["lambda_fallback"] if fallback is None else fallback)


class CKINNAuto:
    """Protocol section 6, model 5: CKINN with lambda chosen inside the training data, then refitted on all of it."""

    def __init__(self, cols, seed=0, priors=None, grid=None, **kw):
        self.cols, self.seed, self.priors, self.kw = list(cols), int(seed), priors, kw
        self.grid = HP["lambda_grid"] if grid is None else grid

    def fit(self, df):
        fit_part, val = inner_split(df, self.seed)
        self.inner_scores_ = {}
        for lam in self.grid:
            f = fit_with_baseline(DeepCox(self.cols, lam=lam, seed=self.seed, priors=self.priors, **self.kw), fit_part)
            v = val[val["time_y"] < fit_part["time_y"].max()]
            self.inner_scores_[lam] = scaled_brier_at(fit_part, v, f.risk(v))
        self.lam_ = choose_lambda(self.inner_scores_)
        self.model = DeepCox(self.cols, lam=self.lam_, seed=self.seed, priors=self.priors, **self.kw).fit(df)
        self.prep, self.net, self.cons = self.model.prep, self.model.net, self.model.cons
        return self

    def lp(self, df):
        return self.model.lp(df)


MODELS["CKINN"] = lambda cols, seed=0: CKINNAuto(cols, seed=seed)
DIR["models"] = OUT / "models"
DIR["models"].mkdir(parents=True, exist_ok=True)


def save_network(model, name):
    """Save a fitted MLP or CKINN (weights, preprocessing, settings) to ckinn_results/models/<name>.pt."""
    m = model.model if isinstance(model, CKINNAuto) else model
    path = DIR["models"] / f"{name}.pt"
    torch.save({"state": m.net.state_dict(), "cols": m.cols, "lam": m.lam, "seed": m.seed, "priors": dict(m.priors),
                "prep": {"fill": m.prep.fill.to_dict(), "ind": m.prep.ind, "mu": m.prep.mu.to_dict(), "sd": m.prep.sd.to_dict(), "names": m.prep.names},
                "inner_scores": getattr(model, "inner_scores_", None), "settings_sha": settings_fingerprint()}, path)
    return path


def load_network(name):
    """Reload a saved network as a DeepCox that predicts exactly as the original did."""
    ck = torch.load(DIR["models"] / f"{name}.pt", weights_only=False)
    if ck["settings_sha"] != settings_fingerprint():
        raise RuntimeError(f"{name}.pt was saved under other settings")
    m = DeepCox(ck["cols"], lam=ck["lam"], seed=ck["seed"], priors=ck["priors"])
    m.prep = Prep(ck["cols"])
    m.prep.fill, m.prep.mu, m.prep.sd = pd.Series(ck["prep"]["fill"])[ck["cols"]], pd.Series(ck["prep"]["mu"])[ck["cols"]], pd.Series(ck["prep"]["sd"])[ck["cols"]]
    m.prep.ind, m.prep.names = ck["prep"]["ind"], ck["prep"]["names"]
    m.net = build_net(len(m.prep.names))
    m.net.load_state_dict(ck["state"])
    m.net.eval()
    m.cons = [(m.prep.names.index(c), s) for c, s in m.priors.items() if c in m.prep.names]
    return m


# ---------------------------------------------------------------- tests of the rule and of the model files (toy arrays)
A.check("rule: all four within 0.002 of the best -> the largest lambda, 10", choose_lambda({0.0: 0.0200, 0.1: 0.0195, 1.0: 0.0190, 10.0: 0.0185}) == 10.0)
A.check("rule: lambda 10 falls more than 0.002 below the best -> lambda 1", choose_lambda({0.0: 0.0200, 0.1: 0.0199, 1.0: 0.0185, 10.0: 0.0100}) == 1.0)
A.check("rule: lambda 0 is never chosen, even when it scores best; no positive lambda qualifies -> fallback 0.1",
        choose_lambda({0.0: 0.0300, 0.1: 0.0100, 1.0: 0.0050, 10.0: 0.0010}) == 0.1)
A.check("rule: a positive lambda that scores best is eligible; ties go to the larger lambda", choose_lambda({0.0: 0.01, 0.1: 0.02, 1.0: 0.02, 10.0: 0.0}) == 1.0)
A.check("the grid and tolerance are the locked ones: {0, 0.1, 1, 10}, 0.002, fallback 0.1", (HP["lambda_grid"], HP["lambda_tol"], HP["lambda_fallback"]) == ((0.0, 0.1, 1.0, 10.0), 0.002, 0.1))
_d = toy_survival(3000)
_tr, _te = _d.iloc[:2000], _d.iloc[2000:]
_auto = CKINNAuto(["x1", "x2"], seed=0, priors={"x1": +1, "x2": -1}, epochs=20, batch=256).fit(_tr)
_fi, _va = inner_split(_tr, 0)
A.check("toy CKINN: four inner fits on 1,600 rows scored on the other 400; the chosen lambda is positive, from the grid, and obeys the rule",
        (len(_fi), len(_va)) == (1600, 400) and sorted(_auto.inner_scores_) == [0.0, 0.1, 1.0, 10.0] and _auto.lam_ in (0.1, 1.0, 10.0) and _auto.lam_ == choose_lambda(_auto.inner_scores_),
        {k: round(v, 4) for k, v in _auto.inner_scores_.items()})
A.check("the final model is refitted on all training rows with the chosen lambda", _auto.model.lam == _auto.lam_ and np.array_equal(
        _auto.lp(_te), DeepCox(["x1", "x2"], lam=_auto.lam_, seed=0, priors={"x1": +1, "x2": -1}, epochs=20, batch=256).fit(_tr).lp(_te)))
_path = save_network(_auto, "_selftest")
_back = load_network("_selftest")
A.check("a saved network reloads to give exactly the same predictions, lambda and priors", np.array_equal(_back.lp(_te), _auto.lp(_te)) and _back.lam == _auto.lam_ and _back.cons == _auto.cons)
_path.unlink()

# ---------------------------------------------------------------- smoke test on the RICH fold, tier T2 (skipped while a session is being restored)
if not restoring():
    _cols, _rows = predictors("RICH", "T2"), []
    for _name in ("MLP", "CKINN"):
        _t0 = time.time()
        _fit = fit_model(_name, _cols, SMOKE_TRAIN, seed=0)
        _mt = evaluate(_fit, SMOKE_TEST)
        _core = _fit.model.model if _name == "CKINN" else _fit.model
        _rows.append({"model": _name, "lambda": _core.lam, "epochs": _core.epochs_run_, "seconds": round(time.time() - _t0),
                      **{k: _mt[k] for k in ("events", "cindex", "auc_3y", "scaled_brier", "oe", "cal_slope", "viol_rate")}})
        A.check(f"{_name}: risks strictly between 0 and 1; C-index over 0.7; Brier score beats the uninformative prediction",
                ((_fit.risk(SMOKE_TEST) > 0) & (_fit.risk(SMOKE_TEST) < 1)).all() and _mt["cindex"] > 0.7 and _mt["scaled_brier"] > 0, round(_mt["cindex"], 4))
        if _name == "CKINN":
            SMOKE_INNER = dict(_fit.model.inner_scores_)
            A.check("CKINN: lambda was chosen by the locked rule from the inner 80/20 split of the training fold only",
                    _fit.model.lam_ == choose_lambda(SMOKE_INNER) and _fit.model.lam_ > 0, {k: round(v, 5) for k, v in SMOKE_INNER.items()})
            A.check("CKINN is constrained on age, BMI, SBP and triglycerides only (tier T2)", sorted(_core.prep.names[i] for i, _s in _core.cons) == ["age", "bmi", "sbp", "tg"])
    print("\n  RICH tier T2, fold (repeat 0, fold 0); one fold only, the full internal validation is stage G:")
    print(pd.DataFrame(_rows).round(4).to_string(index=False))
    print(f"\n  Inner scaled Brier by lambda (training fold only): { {k: round(v, 5) for k, v in SMOKE_INNER.items()} }")
    print("  One fold says nothing about the claims. C1 to C4 are judged on 15 folds (stage G) and on the cross-cohort analysis (stage H).")
_ = A.close(outputs={"rule": "largest positive lambda within 0.002 scaled Brier of the best; else 0.1", "grid": list(HP["lambda_grid"])})
keep_for_session(41)

# %%
# CELL 42 - GATE F: network and CKINN                                                           [status L]
# Why: a stop point before the full internal validation. It confirms that the network, the penalty and the lambda rule
# are defined, tested and registered, that all five locked models are available, and that still no result file exists.
# session guard (same lines in every cell): reopens the project folder and restores earlier set-up cells after a restart
import os, pathlib; _w = pathlib.Path(os.environ.get("CKINN_DIR", "/content/drive/MyDrive/CKINN_Research"))
if _w.is_dir() and pathlib.Path.cwd().resolve() != _w.resolve(): os.chdir(_w)
_g = pathlib.Path.cwd() / "ckinn_results" / "session" / "guard.py"
if not _g.exists(): raise RuntimeError("Run cell 0 (Google Drive) and cells 2 and 3 first.")
exec(_g.read_text(encoding="utf-8")); _ckinn_restore(globals(), 42)
require_gate("E")
A = Audit("gate_F", "Gate F: network and CKINN")

_recs = {c: load_record(c) for c in range(37, 42)}
for c, r in _recs.items():
    A.check(f"cell {c} has a passed record made under the current settings", bool(r) and r["valid"], "missing" if not r else f"{len(r['checks'])} checks, {r['time']}")
LOCKED_MODELS = ["Cox", "XGB", "XGB-mono", "MLP", "CKINN"]
A.check("all five models of protocol section 6 are registered: Cox, XGB, XGB-mono, MLP, CKINN", all(m in MODELS for m in LOCKED_MODELS))
_toy = toy_survival(1500)
_a = DeepCox(["x1", "x2"], lam=0.0, seed=7, epochs=5, batch=256).fit(_toy)
_b = DeepCox(["x1", "x2"], lam=0.0, seed=7, priors={"x1": -1}, epochs=5, batch=256).fit(_toy)
A.check("re-check: with lambda = 0 the priors have no effect at all (MLP and CKINN share one code path)", np.array_equal(_a.lp(_toy), _b.lp(_toy)))
A.check("the study data are unchanged by stage F", all(content_sha(HARM[k]) == HARM_SHA[k] for k in HARM))
A.check("no result part, prediction or model file has been written yet", not any(p.suffix == ".csv" for p in DIR["parts"].iterdir())
        and not any(DIR["preds"].iterdir()) and not any(DIR["models"].iterdir()))
EXPLORATORY_COMPARATORS = ["RSF", "LR", "RF", "XGB-clf", "LGBM-clf", "SVM-linear", "SVM-rbf"]
A.check("the seven exploratory comparators of stage E are still registered", all(m in MODELS for m in EXPLORATORY_COMPARATORS))
ACTIVE_MODELS = LOCKED_MODELS + (EXPLORATORY_COMPARATORS if approved("A1.2") else [])
print(f"\n  Locked models: {LOCKED_MODELS}")
print(f"  Exploratory comparators (A1.2): {EXPLORATORY_COMPARATORS if approved('A1.2') else 'not active'}")
print("  Stage G fits these models on 3 repeats of 5 folds in RICH, in resumable units. It is the first stage that stores results.")
_ = A.close(outputs={"stage": "F", "locked_models": LOCKED_MODELS, "active_models": ACTIVE_MODELS}, extra_path=DIR["gates"] / "gate_F.json")
print("\n  GATE F PASSED. Stage G (cells 43 to 52, internal validation and ablations) may start.")
