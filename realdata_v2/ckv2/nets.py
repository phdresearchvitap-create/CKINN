"""Neural networks M1, M2 and CKINN: architecture, clinical penalty, augmentation, training and attributions (protocol_v2.md, section 4)."""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import BINARY, GROUP_OF, GROUPS


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)


class Net(nn.Module):
    """Group encoders (inputs value*mask and mask) -> trunk -> one logit. use_mask=False feeds a mask of ones (M1).
    The sign table is stored in buffers read by the penalty; it adds no trainable parameter."""

    def __init__(self, features: Sequence[str], hidden=(128, 64), enc_dim: int = 32, dropout: float = 0.2,
                 signs: Optional[Dict[str, int]] = None, use_mask: bool = True):
        super().__init__()
        self.features, self.use_mask = list(features), use_mask
        groups = {g: [i for i, f in enumerate(self.features) if GROUP_OF[f] == g] for g in GROUPS}
        self.groups = {g: ix for g, ix in groups.items() if ix}
        self.enc = nn.ModuleDict({g: nn.Sequential(nn.Linear(2 * len(ix), enc_dim), nn.ReLU()) for g, ix in self.groups.items()})
        layers, d = [], enc_dim * len(self.groups)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk, self.head = nn.Sequential(*layers), nn.Linear(d, 1)
        signs = signs or {}
        self.signs = {f: int(s) for f, s in signs.items() if f in self.features and s != 0}
        self.register_buffer("sign_vec", torch.tensor([float(self.signs.get(f, 0)) for f in self.features]))
        self.register_buffer("is_binary", torch.tensor([f in BINARY for f in self.features], dtype=torch.bool))

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        if not self.use_mask:
            M = torch.ones_like(M)
        h = torch.cat([self.enc[g](torch.cat([X[:, ix] * M[:, ix], M[:, ix]], 1)) for g, ix in self.groups.items()], 1)
        return self.head(self.trunk(h)).squeeze(1)


def clinical_penalty(model: nn.Module, X: torch.Tensor, M: torch.Tensor, signs: torch.Tensor, is_binary: torch.Tensor,
                     create_graph: bool = False, tol: float = 1e-3) -> Tuple[torch.Tensor, float]:
    """Squared hinge on sign-adjusted derivatives (continuous constrained columns, autograd) and on finite differences
    (binary constrained columns), observed entries only, averaged over observed constrained entries. Whole batch (A5).
    Returns (penalty, violation rate). Runs in eval mode so that dropout does not enter the derivative."""
    was = model.training
    model.eval()
    try:
        cont = ((signs != 0) & ~is_binary).to(X.dtype)
        bin_cols = torch.nonzero((signs != 0) & is_binary).flatten().tolist()
        Xg = X.detach().clone().requires_grad_(True)
        z = model(Xg, M)
        (grad,) = torch.autograd.grad(z.sum(), Xg, create_graph=create_graph, retain_graph=True)
        adj, obs = signs * grad, M * cont
        hinge = (torch.relu(-adj) ** 2 * obs).sum()
        viol, count = ((adj.detach() < -tol).to(X.dtype) * obs).sum(), obs.sum()
        if bin_cols:
            k = X.shape[0]
            flips = []
            for j in bin_cols:
                Xf = X.detach().clone()
                Xf[:, j] = 1.0 - Xf[:, j]
                flips.append(Xf)
            zf = model(torch.cat(flips), M.repeat(len(flips), 1))
            for i, j in enumerate(bin_cols):
                toward_one = 1.0 - 2.0 * X[:, j].detach()
                fd, ob = signs[j] * toward_one * (zf[i * k:(i + 1) * k] - z), M[:, j]
                hinge = hinge + (torch.relu(-fd) ** 2 * ob).sum()
                viol, count = viol + ((fd.detach() < -tol).to(X.dtype) * ob).sum(), count + ob.sum()
    finally:
        model.train(was)
    total = count.clamp_min(1.0)
    return hinge / total, float(viol / total)


def augment(xb, mb, gen, panel_cols, other_cols, aug_prob, p_panel, p_feat):
    """Training-time missingness augmentation (v1): a share of rows lose the whole lab panel and scattered clinical values."""
    n = xb.shape[0]
    rows = (torch.rand(n, generator=gen) < aug_prob).float().unsqueeze(1)
    drop = torch.zeros_like(mb)
    if panel_cols:
        drop[:, panel_cols] = (torch.rand(n, 1, generator=gen) < p_panel).float().expand(-1, len(panel_cols))
    if other_cols:
        drop[:, other_cols] = torch.maximum(drop[:, other_cols], (torch.rand(n, len(other_cols), generator=gen) < p_feat).float())
    mb2 = mb * (1 - rows * drop)
    return xb * mb2, mb2


def weighted_val_score(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Log loss + Brier (the v1 validation score), optionally weighted."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ll = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()
    return float(ll + (w * (p - y) ** 2).sum() / w.sum())


def predict_logit(net: nn.Module, X: np.ndarray, M: np.ndarray, batch: int = 65536) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(net(torch.as_tensor(X[s:s + batch]), torch.as_tensor(M[s:s + batch])).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def train_net(net: Net, tr, va, hp: dict, cfg, seed: int, lam: float = 0.0, aug: bool = False,
              sign_override: Optional[torch.Tensor] = None) -> dict:
    """AdamW, gradient clipping at 1, early stopping on validation log loss + Brier; restores the best weights.
    tr and va are (X, M, y, w). The penalty uses the net's sign buffer unless sign_override is given."""
    Xtr, Mtr, ytr, wtr = [torch.as_tensor(np.asarray(a, np.float32)) for a in tr]
    Xva, Mva, yva, wva = va
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    panel_cols = [i for i, f in enumerate(net.features) if GROUP_OF[f] in ("panel", "glyc")]
    other_cols = [i for i, f in enumerate(net.features) if GROUP_OF[f] == "clin"]
    signs = net.sign_vec if sign_override is None else sign_override
    use_pen = lam > 0 and bool((signs != 0).any())
    best, best_state, bad, hist = np.inf, None, 0, {"val": [], "pen": [], "viol": []}
    n = len(ytr)
    for epoch in range(cfg.max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        pen_sum, viol_sum = 0.0, 0.0
        for s in range(0, n, cfg.batch):
            ix = perm[s:s + cfg.batch]
            xb, mb, yb, wb = Xtr[ix], Mtr[ix], ytr[ix], wtr[ix]
            if aug:
                xb, mb = augment(xb, mb, gen, panel_cols, other_cols, cfg.aug_prob, cfg.p_panel_drop, cfg.p_feat_drop)
            loss = (nn.functional.binary_cross_entropy_with_logits(net(xb, mb), yb, reduction="none") * wb).sum() / wb.sum().clamp_min(1e-12)
            if use_pen:
                pen, vr = clinical_penalty(net, xb, mb, signs, net.is_binary, create_graph=True, tol=cfg.viol_tol)
                loss = loss + lam * pen
                pen_sum += float(pen.detach()) * len(ix)
                viol_sum += vr * len(ix)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        pv = 1 / (1 + np.exp(-predict_logit(net, Xva, Mva)))
        sc = weighted_val_score(yva, pv, wva)
        hist["val"].append(sc)
        hist["pen"].append(pen_sum / n)
        hist["viol"].append(viol_sum / n)
        if sc < best - 1e-5:
            best, bad, best_state = sc, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    hist["epochs"], hist["best"] = len(hist["val"]), best
    return hist


def _logit_grad(net: nn.Module, X: torch.Tensor, M: torch.Tensor):
    Xg = X.clone().requires_grad_(True)
    z = net(Xg, M)
    (g,) = torch.autograd.grad(z.sum(), Xg)
    return z.detach(), g.detach()


def integrated_gradients(net: nn.Module, X: np.ndarray, M: np.ndarray, steps: int) -> np.ndarray:
    """IG of the logit with a zero baseline on the z-scale (the training mean), mask held fixed; midpoint rule."""
    net.eval()
    Xt, Mt = torch.as_tensor(X), torch.as_tensor(M)
    tot = torch.zeros_like(Xt)
    for s in range(steps):
        _, g = _logit_grad(net, Xt * ((s + 0.5) / steps), Mt)
        tot += g
    return (Xt * tot / steps).numpy()


def violation_rows(net: nn.Module, X: np.ndarray, M: np.ndarray, signs: Dict[str, int], features: Sequence[str], tol: float,
                   attributions: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per row (v1 manipulation-check statistics): wrong-sign derivative count and observed constrained count; integrated-gradient
    direction consistency on rows where the constrained value is away from the mean."""
    net.eval()
    Xt, Mt = torch.as_tensor(X), torch.as_tensor(M)
    sg = torch.tensor([float(signs.get(f, 0)) for f in features])
    isb = torch.tensor([f in BINARY for f in features], dtype=torch.bool)
    ctrl, cont = sg != 0, (sg != 0) & ~isb
    z0, g0 = _logit_grad(net, Xt, Mt)
    obs = Mt * cont.float()
    viol = ((sg * g0 < -tol).float() * obs).sum(1)
    vobs = obs.sum(1)
    for j in torch.nonzero(ctrl & isb).flatten().tolist():
        Xf = Xt.clone()
        Xf[:, j] = 1 - Xf[:, j]
        with torch.no_grad():
            zf = net(Xf, Mt)
        fd = sg[j] * (1 - 2 * Xt[:, j]) * (zf - z0)
        viol += (fd < -tol).float() * Mt[:, j]
        vobs += Mt[:, j]
    out = {"viol": viol.numpy(), "vobs": vobs.numpy()}
    if attributions is not None:
        A = torch.as_tensor(attributions)
        tested = ctrl.float() * Mt * ((~isb).float() * (Xt.abs() > 0.25).float() + isb.float() * (Xt == 1).float())
        cons = ((torch.sign(A) * sg * torch.sign(Xt)) >= 0).float() * tested
        out["ig_cons"], out["ig_n"] = cons.sum(1).numpy(), tested.sum(1).numpy()
    return out
