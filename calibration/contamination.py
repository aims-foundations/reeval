import math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# -------------- Utilities --------------

def center_params(theta, z):
    return theta - theta.mean(), z - z.mean()

def auc_roc(labels, scores):
    """Rank-based ROC-AUC (no sklearn)."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / denom) if denom > 0 else np.nan

# -------------- Synthetic data --------------

def make_synthetic(N=60, M=250, T_max=24.0, seed=123):
    """
    N models, M items. Times uniform on [0, T_max] months.
    Contamination considerable by design.
    """
    g = torch.Generator().manual_seed(seed)

    theta_true = torch.randn(N, generator=g, device=device)
    z_true = torch.randn(M, generator=g, device=device)
    theta_true, z_true = center_params(theta_true, z_true)

    # Item-level contamination gains, positive and fairly large on average
    delta_true = F.softplus(torch.randn(M, generator=g, device=device) * 0.7 + 1.3)

    # Release times
    s = torch.rand(N, generator=g, device=device) * T_max  # model release time
    t = torch.rand(M, generator=g, device=device) * T_max  # item publish time

    # All pairs
    idx_i, idx_j = torch.cartesian_prod(torch.arange(N), torch.arange(M)).T
    idx_i = idx_i.to(device)
    idx_j = idx_j.to(device)

    # Exposure features x_ij
    lag = (s[idx_i] - t[idx_j]).clamp_min(0.0)  # months of exposure
    m = (lag > 0).to(torch.float64)             # exposure mask
    x = torch.log1p(lag / 3.0).unsqueeze(1)     # 1 feature: log-lag
    # True contamination propensity pi_ij = sigmoid(alpha0 + alpha*x_ij) * m_ij
    alpha0_true = -0.6
    alpha_true = torch.tensor([2.2], dtype=torch.float64, device=device)
    pi_true = torch.sigmoid(alpha0_true + x @ alpha_true) * m

    # True Rasch logits
    eta_true = theta_true[idx_i] - z_true[idx_j]

    # Sample latent contamination and responses
    c_true = torch.bernoulli(pi_true, generator=g)
    p_clean = torch.sigmoid(eta_true)
    p_cont  = torch.sigmoid(eta_true + delta_true[idx_j])
    p = (1 - c_true) * p_clean + c_true * p_cont
    y = torch.bernoulli(p, generator=g)

    meta = {
        "theta_true": theta_true, "z_true": z_true, "delta_true": delta_true,
        "alpha0_true": alpha0_true, "alpha_true": alpha_true,
        "s": s, "t": t, "idx_i": idx_i, "idx_j": idx_j,
        "lag": lag, "m": m, "x": x, "pi_true": pi_true, "c_true": c_true, "y": y,
        "avg_pi_exposed": float(pi_true[m > 0].mean().cpu()),
        "avg_pi_overall": float(pi_true.mean().cpu())
    }
    return meta

# -------------- EM with LBFGS inner loops --------------

class EMContaminationRasch:
    def __init__(self, idx_i, idx_j, y, x, m,
                 lam_theta=1e-3, lam_z=1e-3, lam_delta=1e-3, lam_alpha=1e-3):
        """
        y: {0,1} tensor over all (i,j) pairs in the same order as idx_i, idx_j
        x: features matrix [P, P_feat]
        m: exposure mask [P], 1 if s_i >= t_j else 0
        """
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.y = y.to(torch.float64)
        self.x = x
        self.m = m
        self.P_feat = x.shape[1]

        self.N = int(idx_i.max().item() + 1)
        self.M = int(idx_j.max().item() + 1)

        # Parameters
        self.theta = torch.zeros(self.N, requires_grad=True, device=device)
        self.z = torch.zeros(self.M, requires_grad=True, device=device)
        self.delta_raw = torch.zeros(self.M, requires_grad=True, device=device)  # softplus to get delta>=0
        self.alpha0 = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha  = torch.zeros(self.P_feat, requires_grad=True, device=device)

        # Regularization
        self.lam_theta = lam_theta
        self.lam_z = lam_z
        self.lam_delta = lam_delta
        self.lam_alpha = lam_alpha

        # Warm start theta, z with plain Rasch
        self._warmstart_rasch(max_iter=100)

    def _warmstart_rasch(self, max_iter=100):
        def nll():
            eta = self.theta[self.idx_i] - self.z[self.idx_j]
            reg = self.lam_theta * self.theta.pow(2).mean() + self.lam_z * self.z.pow(2).mean()
            return (F.softplus(eta) - self.y * eta).mean() + reg

        opt = torch.optim.LBFGS([self.theta, self.z], lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-8, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = nll()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            self.theta.data, self.z.data = center_params(self.theta.data, self.z.data)

    @torch.no_grad()
    def estep(self):
        eta = self.theta[self.idx_i] - self.z[self.idx_j]
        delta = F.softplus(self.delta_raw)[self.idx_j]
        pi = torch.sigmoid(self.alpha0 + self.x @ self.alpha) * self.m

        p0 = torch.sigmoid(eta)
        p1 = torch.sigmoid(eta + delta)

        lik0 = p0 * self.y + (1 - p0) * (1 - self.y)
        lik1 = p1 * self.y + (1 - p1) * (1 - self.y)

        num = pi * lik1
        den = num + (1 - pi) * lik0 + 1e-12
        r = (num / den).clamp(0.0, 1.0)
        r = r * self.m  # force zero when not exposed
        return r

    def _mstep_theta_z(self, r, max_iter=50):
        params = [self.theta, self.z]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-8, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            eta = self.theta[self.idx_i] - self.z[self.idx_j]
            delta = F.softplus(self.delta_raw)[self.idx_j].detach()

            loss_clean = (1 - r) * (F.softplus(eta) - self.y * eta)
            loss_cont  = r * (F.softplus(eta + delta) - self.y * (eta + delta))
            loss = (loss_clean + loss_cont).mean()
            loss = loss + self.lam_theta * self.theta.pow(2).mean() + self.lam_z * self.z.pow(2).mean()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            self.theta.data, self.z.data = center_params(self.theta.data, self.z.data)

    def _mstep_delta(self, r, max_iter=50):
        params = [self.delta_raw]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-8, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            eta = (self.theta[self.idx_i] - self.z[self.idx_j]).detach()
            delta = F.softplus(self.delta_raw)[self.idx_j]
            term = r * (F.softplus(eta + delta) - self.y * (eta + delta))
            loss = term.mean() + self.lam_delta * self.delta_raw.pow(2).mean()
            loss.backward()
            return loss
        opt.step(closure)

    def _mstep_alpha(self, r, max_iter=50):
        # Only exposed pairs contribute
        exposed = (self.m > 0)
        if exposed.sum() == 0:
            return
        X = self.x[exposed]
        r_exp = r[exposed].detach()
        params = [self.alpha0, self.alpha]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-8, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            logits = self.alpha0 + X @ self.alpha
            loss = (F.softplus(logits) - r_exp * logits).mean()
            loss = loss + self.lam_alpha * (self.alpha.pow(2).mean() + self.alpha0.pow(2))
            loss.backward()
            return loss
        opt.step(closure)

    @torch.no_grad()
    def observed_nll(self):
        eta = self.theta[self.idx_i] - self.z[self.idx_j]
        delta = F.softplus(self.delta_raw)[self.idx_j]
        pi = torch.sigmoid(self.alpha0 + self.x @ self.alpha) * self.m
        p = (1 - pi) * torch.sigmoid(eta) + pi * torch.sigmoid(eta + delta)
        ll = self.y * torch.log(p + 1e-12) + (1 - self.y) * torch.log(1 - p + 1e-12)
        return float(-ll.mean().cpu())

    def fit(self, iters=20, inner_max_iter=(60, 40, 40), verbose=True):
        hist = []
        for it in range(iters):
            r = self.estep()
            self._mstep_theta_z(r, max_iter=inner_max_iter[0])
            self._mstep_delta(r,    max_iter=inner_max_iter[1])
            self._mstep_alpha(r,    max_iter=inner_max_iter[2])
            nll = self.observed_nll()
            hist.append(nll)
            if verbose:
                if it % 5 == 0:
                    print(f"[EM {it+1:02d}] NLL={nll:.6f}")
        return hist

# -------------- Run a demo --------------

meta = make_synthetic(N=200, M=1000, T_max=24.0, seed=8)
print(f"Avg true pi on exposed pairs: {meta['avg_pi_exposed']:.3f}")
print(f"Avg true pi overall:         {meta['avg_pi_overall']:.3f}")
breakpoint()
model = EMContaminationRasch(
    idx_i=meta["idx_i"],
    idx_j=meta["idx_j"],
    y=meta["y"],
    x=meta["x"],
    m=meta["m"],
    lam_theta=1e-3, lam_z=1e-3, lam_delta=1e-3, lam_alpha=1e-3
)

# You can shrink inner_max_iter to speed up. Increase to tighten convergence.
history = model.fit(iters=50, inner_max_iter=(40, 30, 30), verbose=True)

# Evaluate recovery
with torch.no_grad():
    theta_hat = model.theta.detach().cpu().numpy()
    z_hat = model.z.detach().cpu().numpy()
    delta_hat = F.softplus(model.delta_raw).detach().cpu().numpy()
    r_scores = model.estep().cpu().numpy()
    c_true_np = meta["c_true"].cpu().numpy()
    mask_exp = meta["m"].cpu().numpy().astype(bool)

print("\n--- Parameter recovery ---")
print(f"corr(theta_hat, theta_true) = {corr(theta_hat, meta['theta_true'].cpu().numpy()):.3f}")
print(f"corr(z_hat, z_true)         = {corr(z_hat, meta['z_true'].cpu().numpy()):.3f}")
print(f"corr(delta_hat, delta_true) = {corr(delta_hat, meta['delta_true'].cpu().numpy()):.3f}")

print("\n--- Contamination detection on exposed pairs ---")
print(f"AUC(r_ij vs c_ij)           = {auc_roc(c_true_np[mask_exp], r_scores[mask_exp]):.3f}")

print("\n--- Propensity parameters ---")
print(f"alpha0_true = {meta['alpha0_true']:.3f}")
print(f"alpha_true  = {float(meta['alpha_true'][0]):.3f}")
print(f"alpha0_hat  = {float(model.alpha0.detach().cpu().numpy()):.3f}")
print(f"alpha_hat   = {float(model.alpha.detach().cpu().numpy()[0]):.3f}")

print("\n--- Mean delta ---")
print(f"mean(delta_true) = {float(meta['delta_true'].mean().cpu().numpy()):.3f}")
print(f"mean(delta_hat)  = {float(F.softplus(model.delta_raw).mean().detach().cpu().numpy()):.3f}")
