import pandas as pd
import torch
from torch.distributions import Bernoulli
from torchmetrics.functional import spearman_corrcoef
from tqdm import tqdm

torch.manual_seed(0)
data = pd.read_csv("Subset.csv")
data = torch.tensor(data.values).float()
n_test_takers, n_items = data.shape

z = torch.tensor(torch.zeros(n_items), requires_grad=True)


def closure():
    optim.zero_grad()
    probs = torch.sigmoid(thetas[:, :, None] + z[None, None, :])
    loss = -Bernoulli(probs=probs).log_prob(data).mean()
    loss.backward()
    return loss


optim = torch.optim.LBFGS(
    [z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
)

thetas = torch.randn(500, n_test_takers)
# >>> 500 x n_test_takers

pbar = tqdm(range(100))
for iteration in pbar:
    if iteration > 0:
        previous_z = z.clone()
        previous_loss = loss.clone()

    loss = optim.step(closure)

    if iteration > 0:
        d_loss = previous_loss - loss
        d_z = torch.norm(previous_z - z, p=2)
        grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
        pbar.set_postfix({"grad_norm": grad_norm, "d_z": d_z, "d_loss": d_loss})
        if loss_diff < 1e-5 and z_diff < 1e-5 and grad_norm < 1e-5:
            break

item_difficulty = pd.read_csv("item_difficulty.csv", header=None)
item_difficulty = torch.tensor(item_difficulty.values)[:, 1]

# compute the spearman correlation
print("Spearman with CTT: ", spearman_corrcoef(z, data.mean(0)))
print("Spearman with mirt: ", spearman_corrcoef(z, item_difficulty))
