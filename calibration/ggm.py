from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from sklearn.covariance import GraphicalLasso, graphical_lasso

import numpy as np
from util import get_official_provider_benchmark ,get_helm_benchmark
from tqdm import tqdm
import torch
from tqdm import tqdm


def calculate_tetrachoric(data):
    n_questions = data.shape[1]
    corr_matrix = np.zeros((n_questions, n_questions))
    for i in tqdm(range(n_questions)):
        if i % 5 == 0:
            print(f"proc {i}")
        for j in range(i, n_questions):
            if j % 10 == 0:
                print(j)
            r = tetrachoric_corr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = r
    # return np.nanmean(corr_matrix, axis=1)
    return corr_matrix



def torch_tetrachoric_corr(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    vec1, vec2: 1D torch tensors (binary 0/1; NaNs allowed). Returns a scalar tensor.
    """
    assert vec1.ndim == 1 and vec2.ndim == 1, "Inputs must be 1D"
    assert vec1.shape[0] == vec2.shape[0], "Length mismatch"

    # mask out NaNs pairwise
    mask = torch.isfinite(vec1) & torch.isfinite(vec2)
    x = vec1[mask]
    y = vec2[mask]

    # no data left
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=vec1.device)

    # ensure binary {0,1}; if values contain -1/1 or scores, binarize by > 0
    if not ((x == 0) | (x == 1)).all() or not ((y == 0) | (y == 1)).all():
        x = (x > 0).to(torch.int8)
        y = (y > 0).to(torch.int8)

    # exact replicas
    if torch.equal(x, y):
        return torch.tensor(1.0, device=vec1.device)

    # counts
    A = ((x == 0) & (y == 0)).sum().to(torch.float32)
    B = ((x == 0) & (y == 1)).sum().to(torch.float32)
    C = ((x == 1) & (y == 0)).sum().to(torch.float32)
    D = ((x == 1) & (y == 1)).sum().to(torch.float32)

    # degenerate cases
    if (B == 0) or (C == 0):
        return torch.tensor(float("nan"), device=vec1.device)

    AD = A * D
    # r = cos(pi / (1 + sqrt(AD / (B*C))))
    r = torch.cos(torch.pi / (1.0 + torch.sqrt(AD / (B * C))))
    return r


def calculate_tetrachoric_torch(data, device: str = None, return_matrix: bool = False):
    """
    data: (n_samples, n_questions) array-like (NumPy or torch). May contain NaNs and non-binary values.
    device: e.g., 'cuda:0', 'cuda:1', or 'cpu'. If None, stays where it is or goes to CPU.
    return_matrix: if True, returns full (Q x Q) matrix; else returns per-question nanmean across each row.

    Returns:
        - if return_matrix: (Q, Q) torch.Tensor on device
        - else: (Q,) torch.Tensor on device (nanmean over each row)
    """
    # convert to torch tensor on device
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)

    if device is not None:
        print(device)
        data = data.to(device)

    assert data.ndim == 2, "data must be 2D: (n_samples, n_questions)"
    n_samples, n_questions = data.shape

    corr = torch.empty((n_questions, n_questions), dtype=torch.float32, device=data.device)
    corr[:] = float("nan")

    for i in tqdm(range(n_questions)):
        # diagonal is perfect correlation with itself (unless all NaN; handle via function for consistency)
        ri = torch_tetrachoric_corr(data[:, i], data[:, i])
        corr[i, i] = ri
        for j in range(i + 1, n_questions):

                
            r = torch_tetrachoric_corr(data[:, i], data[:, j])
            corr[i, j] = r
            corr[j, i] = r

    if return_matrix:
        return corr
    else:
        # nanmean over columns per row (per-question average correlation)
        return torch.nanmean(corr, dim=1)


import torch

@torch.no_grad()
def tetrachoric_matrix_torch(data: torch.Tensor, device: str = None):
    """
    data: (N, Q) tensor; may contain NaN; can be {0,1}, {-1,1}, or real-valued.
    device: 'cuda:0', 'cuda:1', or 'cpu'. If None, keep current device.
    Returns: (Q, Q) tetrachoric correlation matrix as torch.float32 on device.
    """
    # Move to torch/device
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if device is not None:
        data = data.to(device)

    N, Q = data.shape

    # Build mask: valid if finite
    M = torch.isfinite(data).to(torch.float32)        # (N,Q) in {0,1}

    # Binarize:
    #  - if already {0,1}, keep
    #  - if {-1,1} or general real, threshold at >0
    X = data.clone()
    if not (((X == 0) | (X == 1) | ~torch.isfinite(X)).all()):
        X = (X > 0).to(torch.float32)
    else:
        X = X.to(torch.float32)

    # Zero out invalids
    Xw = X * M                                        # (N,Q)

    # Pairwise counts via matmul
    V = M.T @ M                                       # valid pair counts  (Q,Q)
    D = Xw.T @ Xw                                     # n11                 (Q,Q)
    T = M.T @ Xw                                      # helper              (Q,Q)
    U = Xw.T @ M                                      # helper              (Q,Q)

    B = T - D                                         # n01
    C = U - D                                         # n10
    A = V - B - C - D                                 # n00

    # Avoid division by zero: mask where B==0 or C==0 or V==0
    eps_mask = (B > 0) & (C > 0) & (V > 0)

    # r = cos(pi / (1 + sqrt( (A*D) / (B*C) )))
    R = torch.full((Q, Q), float("nan"), dtype=torch.float32, device=data.device)
    num = (A * D)[eps_mask]
    den = (B * C)[eps_mask]
    val = torch.cos(torch.pi / (1.0 + torch.sqrt(num / den)))
    R[eps_mask] = val

    # For identical columns (B=C=0 but some valid data), set r=1 (matches your scalar impl)
    same_col = (B == 0) & (C == 0) & (V > 0)
    R[same_col] = 1.0

    # Ensure diagonal = 1 (if there is at least one valid entry)
    diag_valid = torch.diag(V) > 0
    R.diagonal().masked_fill_(diag_valid, 1.0)

    return R


@torch.no_grad()
def tetrachoric_row_mean_torch(data: torch.Tensor, device: str = None):
    """Return per-question mean tetrachoric (row-wise nanmean)."""
    R = tetrachoric_matrix_torch(data, device)
    # return torch.nanmean(R, dim=1)
    return R



n,m = 20000, 200
U = torch.rand(n, 2)   # values in [0,1)
V = torch.rand(m, 2)   # values in [0,1)
probs = torch.sigmoid(U @ V.T)
data_withneg1 = torch.bernoulli(probs)



# data_withneg1[~data_idtor] = float("nan")

# tetrachoric_row_mean_torch
S = tetrachoric_row_mean_torch(data_withneg1,"cuda:5")
# S2 = calculate_tetrachoric_torch(data_withneg1,"cuda:5",return_matrix=True)
# S3 = calculate_tetrachoric(data_withneg1)

breakpoint()
# Replace any residual NaNs (e.g., after cc) and enforce symmetry/diag
np.fill_diagonal(S, 1.0)
S = 0.5 * (S + S.T)
S = np.nan_to_num(S, nan=0.0)

cov_, precision_ = graphical_lasso(emp_cov=S, alpha=0.02, max_iter=200)

# S = calculate_tetrachoric_torch(data_withneg1,"cuda:1")
print("getting ggm")
gl = GraphicalLasso(alpha=0.02, max_iter=200)
print("fitting ggm")
breakpoint()
gl.fit(S.cpu())                     # treats input as data; set gl.covariance_ = S if needed by wrapping
print("got omega")
Omega = gl.precision_         # sparse precision
print(Omega)
breakpoint()
