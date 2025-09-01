import math
import torch
from torch import nn
from itertools import permutations
from factor_analyzer import Rotator

@torch.no_grad()
def _canonicalize(L):
    """
    Canonicalize columns: flip sign so largest entry per column is positive,
    then order columns by column norm descending.
    """
    Lc = L.clone()
    p, m = Lc.shape
    for k in range(m):
        j = torch.argmax(Lc[:, k].abs())
        if Lc[j, k] < 0:
            Lc[:, k] *= -1
    order = torch.argsort(Lc.pow(2).sum(dim=0), descending=True)
    return Lc[:, order], order

def _kaiser_normalize(L, eps=1e-12):
    """
    Kaiser row normalization. Each row divided by its Euclidean norm.
    Returns normalized loadings and the row norms for later de-normalization.
    """
    row_norm = L.norm(dim=1).clamp_min(eps)
    L_norm = L / row_norm.unsqueeze(1)
    return L_norm, row_norm

def _softplus_inv(y):
    # inverse of softplus: x = log(exp(y) - 1), numerically stable
    y = torch.as_tensor(y, dtype=torch.float64)
    return torch.log(torch.expm1(y))

class _CholeskyOblique(nn.Module):
    """
    Parameterizes T = C (lower triangular with positive diagonal) via an
    unconstrained matrix A. Diagonal uses softplus to keep positivity.
    """
    def __init__(self, m, init_diag=1.0, device=None, dtype=torch.float64):
        super().__init__()
        self.m = m
        A = torch.zeros((m, m), dtype=dtype, device=device)
        # set diagonal so softplus(diag) ~ init_diag
        A_diag = _softplus_inv(init_diag)
        A = torch.tril(A)
        A[range(m), range(m)] = A_diag
        self.A = nn.Parameter(A)

    def T(self):
        # Build strictly lower triangle from A, diagonal from softplus(diag(A))
        A = self.A
        C_strict = torch.tril(A, diagonal=-1)                       # strictly lower, safe view
        diag_pos = torch.nn.functional.softplus(torch.diagonal(A))  # from A directly
        C = C_strict + torch.diag(diag_pos + 1e-8)                  # assemble without in-place
        return C

    def R(self):
        # R = T^{-1}
        C = self.T()
        I = torch.eye(self.m, dtype=C.dtype, device=C.device)
        R = torch.linalg.solve(C, I)  # more stable than inverse
        return R

def geomin_oblique(
    L,
    epsilon=1e-3,
    rstarts=20,
    max_iter=2000,
    lr=0.05,
    tol=1e-7,
    kaiser=True,
    seed=None,
    device=None,
    dtype=torch.float64,
    verbose=False,
):
    """
    Geomin oblique rotation for a loading matrix.

    Arguments
    ---------
    L : torch.Tensor of shape (p, m)
        Unrotated loadings (variables x factors).
    epsilon : float
        Geomin epsilon. Typical values: 1e-3 for 3 factors, 1e-4 to 1e-2 reasonable.
    rstarts : int
        Random restarts. Increase if you suspect local minima.
    max_iter : int
        Maximum Adam steps per restart.
    lr : float
        Adam learning rate.
    tol : float
        Relative improvement tolerance to early stop.
    kaiser : bool
        Apply Kaiser row normalization before rotation. Recommended True.
    seed : int or None
        Random seed base.
    device : torch.device or None
    dtype : torch.dtype

    Returns
    -------
    result : dict with keys
        'L_rot' : rotated loadings (p, m)
        'Phi'   : factor correlation matrix (m, m)
        'T'     : oblique transform so that Phi = T T^T
        'R'     : inverse transform used on loadings, L_rot = L_norm @ R, then de-normalized
        'objective' : geomin criterion value at optimum
        'converged' : bool
        'iters'     : steps used in the best restart
        'order'     : column permutation applied in canonicalization
        'row_norm'  : row norms used for Kaiser de-normalization (if kaiser)
    """
    assert L.ndim == 2, "L must be a 2D tensor (p x m)"
    p, m = L.shape

    device = device or L.device
    L = L.to(device=device, dtype=dtype)

    if kaiser:
        Ln, row_norm = _kaiser_normalize(L)
    else:
        Ln, row_norm = L, torch.ones(p, dtype=dtype, device=device)

    # Precompute constant parts
    Ln = Ln.contiguous()

    def loss_from_R(R):
        # L* = L_n @ R
        Ls = Ln @ R
        # Geomin: sum_i exp( (1/m) * sum_k log(Ls_{ik}^2 + epsilon) )
        x2 = Ls.pow(2).add(epsilon)
        row_log = torch.log(x2).sum(dim=1) / m
        ge = torch.exp(row_log).sum()
        return ge

    best = {
        "objective": math.inf,
        "L_rot": None,
        "Phi": None,
        "T": None,
        "R": None,
        "iters": 0,
        "converged": False,
        "order": None,
        "row_norm": row_norm.clone(),
    }

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))

    for r in range(max(1, rstarts)):
        # Initialize Cholesky C near identity with small lower-tri noise
        model = _CholeskyOblique(m, init_diag=1.0, device=device, dtype=dtype)
        with torch.no_grad():
            noise = torch.zeros((m, m), dtype=dtype, device=device)
            tril_mask = torch.tril(torch.ones_like(noise), diagonal=-1).bool()
            noise[tril_mask] = 0.01 * torch.randn(tril_mask.sum().item(), generator=g, dtype=dtype, device=device)
            model.A.add_(noise)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        prev = None
        converged = False
        steps = 0

        for t in range(max_iter):
            opt.zero_grad(set_to_none=True)
            R = model.R()
            obj = loss_from_R(R)
            obj.backward()
            opt.step()
            steps = t + 1

            val = float(obj.detach())
            if prev is not None:
                rel = abs(prev - val) / (abs(prev) + 1e-12)
                if rel < tol:
                    converged = True
                    if verbose:
                        print(f"[restart {r}] early stop at {steps}, obj={val:.6f}, rel={rel:.3e}")
                    break
            prev = val

            if verbose and (t % 200 == 0 or t == max_iter - 1):
                print(f"[restart {r}] step {t:5d} | obj = {val:.6f}")

        # Evaluate and keep best
        with torch.no_grad():
            T = model.T()
            R = model.R()
            L_rot_norm = Ln @ R
            # De-normalize rows if Kaiser used
            L_rot = (row_norm.unsqueeze(1)) * L_rot_norm
            # Factor correlation Phi = T T^T
            Phi = T @ T.T
            # Canonicalize columns for stable reporting
            L_rot_c, order = _canonicalize(L_rot)
            # Adjust Phi, T, R to match the column permutation and sign flips
            # Build permutation-sign matrix P so that L_rot_c = L_rot @ P
            # We already flipped signs inside _canonicalize, so we only need a permutation.
            # To capture signs, recompute sign flips explicitly.
            # First recover signs used
            signs = torch.ones(m, dtype=dtype, device=device)
            for k in range(m):
                j = torch.argmax(L_rot[:, k].abs())
                if L_rot[j, k] < 0:
                    signs[k] = -1.0
            P = torch.eye(m, dtype=dtype, device=device)
            P = P[:, order] * signs[order]  # apply perm then sign
            # Update transforms so that (L @ P) corresponds to new coordinates
            R_c = R @ P
            # For Phi, new T_c should satisfy Phi_c = T_c T_c^T and also R_c = T_c^{-1}
            # Solve T_c by inverting R_c
            I = torch.eye(m, dtype=dtype, device=device)
            T_c = torch.linalg.solve(R_c, I)
            Phi_c = T_c @ T_c.T

            obj_val = float(loss_from_R(R_c).detach())

            if obj_val < best["objective"]:
                best.update(
                    {
                        "objective": obj_val,
                        "L_rot": L_rot_c,
                        "Phi": Phi_c,
                        "T": T_c,
                        "R": R_c,
                        "iters": steps,
                        "converged": converged,
                        "order": order.cpu(),
                    }
                )

    return best

def align_to_target(V_est: torch.Tensor, V_ref: torch.Tensor):
    """
    Align V_est columns to V_ref by trying all permutations (small m)
    and flipping column signs to maximize agreement.
    Returns aligned V_est and max-abs error.
    """
    m = V_ref.shape[1]
    best_err = float("inf")
    best_V = None
    best_perm = None
    best_signs = None
    for perm in permutations(range(m)):
        W = V_est[:, perm]
        # flip signs so each column points roughly the same way as target
        signs = torch.sign((W * V_ref).sum(dim=0))
        signs[signs == 0] = 1
        W2 = W * signs
        err = (V_ref - W2).abs().max().item()
        if err < best_err:
            best_err = err
            best_V = W2
            best_perm = perm
            best_signs = signs
    return best_V, best_err, best_perm, best_signs


# ---------------------------
# Example
if __name__ == "__main__":
    torch.manual_seed(0)
    p, m = 12, 3
    # make a simple structure loading matrix then scramble it with an oblique transform
    L_true = torch.zeros(p, m, dtype=torch.float64)
    L_true[:4, 0] = torch.tensor([0.8, 0.7, 0.6, 0.65], dtype=torch.float64)
    L_true[4:8, 1] = torch.tensor([0.75, 0.7, 0.55, 0.6], dtype=torch.float64)
    L_true[8:, 2] = torch.tensor([0.9, 0.6, 0.55, 0.5], dtype=torch.float64)
    # oblique scramble
    A = torch.tensor([[1.0, 0.2, 0.1],
                      [0.1, 1.0, 0.25],
                      [0.15, 0.2, 1.0]], dtype=torch.float64)
    T_scramble = torch.linalg.cholesky(A)  # PD
    R_scramble = torch.linalg.solve(T_scramble, torch.eye(m, dtype=torch.float64))
    L_unrot = L_true @ R_scramble  # what EFA would estimate before rotation
    V_unrot = L_unrot
    device = V_unrot.device
    dtype = V_unrot.dtype
    res = geomin_oblique(L_unrot, epsilon=1e-3, rstarts=10, max_iter=2000, lr=0.05, verbose=False)
    print("Objective:", res["objective"])
    print("Phi:\n", res["Phi"])
    print("Rotated loadings:\n", torch.round(res["L_rot"], decimals=3))
    Phi_hat = res["Phi"]     # factor correlation after rotation (T_hat T_hat^T)

        
    R = res["R"]                             # (m × m) right-side transform (includes perm/sign)
    V_rot = V_unrot @ R                      # V' = V_unrot R

    # 2) Make some synthetic factor scores U (no special distribution required)
    N = 500
    U = torch.randn(N, m, dtype=dtype, device=device)

    # 3) Rotate U properly so the product is invariant: U' = U R^{-T}
    I = torch.eye(m, dtype=dtype, device=device)
    U_rot = U @ torch.linalg.solve(R.T, I)

    # 4) Check invariance: U V_unrot^T == U' V'^T (up to numerical precision)
    orig = U @ V_unrot.T
    rot  = U_rot @ V_rot.T
    max_abs_err = (orig - rot).abs().max().item()
    print("max |U V^T - U' V'^T| =", max_abs_err) 
        
        # L: unrotated loadings, shape (p, m)
    rot = Rotator(method="geomin_obl", delta=0.001, max_iter=2000, tol=1e-6)
    V_rot_2 = rot.fit_transform(V_unrot)   # rotated loadings
    R_2 = rot.rotation_              # right-multiplier so that L_rot = L @ R
    Phi_2 = rot.phi_                 # factor correlation matrix (oblique only)
    
    
    
    
    
    breakpoint()   
    
    
    #==============
    # torch.manual_seed(0)
    # dtype = torch.float64

    # p, m = 12, 3

    # # 1) Simple-structure loadings (this is your ground-truth V)
    # V_true = torch.zeros(p, m, dtype=dtype)
    # V_true[:4, 0] = torch.tensor([0.8, 0.7, 0.6, 0.65], dtype=dtype)
    # V_true[4:8, 1] = torch.tensor([0.75, 0.7, 0.55, 0.6], dtype=dtype)
    # V_true[8:, 2]  = torch.tensor([0.9, 0.6, 0.55, 0.5], dtype=dtype)

    # # 2) Build two different oblique scrambles.
    # #    Use lower-triangular T with positive diagonal so A = T T^T is PD, and R = T^{-1}.
    # T1 = torch.tensor([[1.0, 0.0, 0.0],
    #                    [0.20, 1.0, 0.0],
    #                    [0.10, 0.25, 1.0]], dtype=dtype)
    # A1 = T1 @ T1.T
    # R1 = torch.linalg.solve(T1, torch.eye(m, dtype=dtype))

    # T2 = torch.tensor([[1.0, 0.0, 0.0],
    #                    [-0.30, 1.0, 0.0],
    #                    [0.25, 0.20, 1.0]], dtype=dtype)
    # A2 = T2 @ T2.T
    # R2 = torch.linalg.solve(T2, torch.eye(m, dtype=dtype))

    # # 3) Apply first scramble, then second scramble
    # V1 = V_true @ R1              # once-scrambled
    # V2 = V_true @ (R1 @ R2)       # twice-scrambled
    # print("get first R")
    # # 4) Rotate each scrambled matrix with geomin oblique
    # res1 = geomin_oblique(V1, epsilon=1e-3, rstarts=20, max_iter=2000, lr=0.05, verbose=False)
    # Rhat1 = res1["R"]; Phi1 = res1["Phi"]; V1_rot = V1 @ Rhat1
    # print("get second R")
    # res2 = geomin_oblique(V2, epsilon=1e-3, rstarts=20, max_iter=2000, lr=0.05, verbose=False)
    # Rhat2 = res2["R"]; Phi2 = res2["Phi"]; V2_rot = V2 @ Rhat2
    
    # print("align")
    # # Align each rotated solution to V_true
    # V1_aligned, err1, perm1, sgn1 = align_to_target(V1_rot, V_true)
    # V2_aligned, err2, perm2, sgn2 = align_to_target(V2_rot, V_true)
    # breakpoint()
    # print("max |V_true - align(V1_rot)| =", err1)
    # print("max |V_true - align(V2_rot)| =", err2)

    # # Or align V1_rot directly to V2_rot
    # V2_to_V1, err12, perm12, sgn12 = align_to_target(V2_rot, V1_rot)
    # print("max |V1_rot - align(V2_rot->V1)| =", err12)

    # # Covariance invariance check (both should match Sigma_true)
    # I = torch.eye(m, dtype=dtype)
    # Sigma_true = V_true @ V_true.T
    # Sigma1_hat = V1 @ Rhat1 @ Phi1 @ Rhat1.T @ V1.T
    # Sigma2_hat = V2 @ Rhat2 @ Phi2 @ Rhat2.T @ V2.T
    # print("max |Σ_true - Σ_hat (1 scramble)| =", (Sigma_true - Sigma1_hat).abs().max().item())
    # print("max |Σ_true - Σ_hat (2 scrambles)| =", (Sigma_true - Sigma2_hat).abs().max().item())

    # # Optional: see that Rhat1 and Rhat2 invert your scrambles up to signed permutation
    # M1 = Rhat1 @ R1              # ≈ signed permutation
    # M2 = Rhat2 @ (R1 @ R2)       # ≈ signed permutation
    # print("Rhat1 @ R1:\n", torch.round(M1, 3))
    # print("Rhat2 @ (R1 @ R2):\n", torch.round(M2, 3))