import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.distributions import Bernoulli
from torch.nn import Parameter
from torch.optim import LBFGS
from torchmetrics.functional import spearman_corrcoef
from tqdm import tqdm

torch.manual_seed(0)
gpu_id = 0
tol = 1e-5

data_folder = snapshot_download(
    repo_id="stair-lab/reeval_another_repo", repo_type="dataset"
)
df = pd.read_parquet(f"{data_folder}/mmlu/data.parquet", engine="fastparquet")
question_keys = pd.read_parquet(
    f"{data_folder}/mmlu/question.parquet", engine="fastparquet"
)

subjects = df["subject"].unique()

np.random.seed(1)
np.random.shuffle(subjects)
subjects_subset = subjects  # [0:20]

df_subject = df[df["subject"].isin(subjects_subset)]
df_subject = df_subject[["model_id", "instance_id", "exact_match"]]

# create a pivot table with model_id as index, instance_id as columns, and exact_match as values
matrix = df_subject.pivot(index="model_id", columns="instance_id", values="exact_match")

# print number of columns and rows
print("Number of questions: ", matrix.shape[1])

question_keys_subject = question_keys[question_keys["subject"].isin(subjects_subset)]

# shuffle the column of the `matrix` and the row of `question_keys_subject`
# to make sure that the order of embedding and the column matches
perm = np.random.permutation(matrix.shape[1])
matrix = matrix.iloc[:, perm]
question_keys_subject = question_keys_subject.iloc[perm]

# Make sure that the order of embedding and the column matches!
assert matrix.columns.tolist() == question_keys_subject["instance_id"].values.tolist()

# drop the columns with only (0 and -1) or (1 and -1)
invalid_columns = matrix.columns[
    matrix.isin([0, -1]).all() | matrix.isin([1, -1]).all()
]
invalid_instance_ids = invalid_columns.values
question_keys_subject = question_keys_subject.drop(index=invalid_instance_ids)
matrix = matrix.drop(columns=invalid_instance_ids)

assert matrix.shape[1] == question_keys_subject.shape[0]

embed = torch.tensor(
    question_keys_subject["embedding"].values.tolist(),
    dtype=torch.float32,
    device="cuda",
)
data = torch.tensor(matrix.values, dtype=torch.float32, device="cuda")
n_test_takers = data.shape[0]
n_questions = data.shape[1]
n_embed_dim = embed.shape[1]
n_mc_e_step = 512

# partition embed and data to train and test set with 80-20 ratio
n_train = int(0.8 * n_questions)
embed_train, embed_test = embed[:n_train], embed[n_train:]
data_train, data_test = data[:, :n_train], data[:, n_train:]

weights = Parameter(torch.randn(n_embed_dim, 1, device="cuda"))
optim = LBFGS(
    [weights], lr=0.1, max_iter=20, history_size=50, line_search_fn="strong_wolfe"
)

thetas = torch.randn(n_mc_e_step, n_test_takers, device="cuda")
# >>> n_mc_e_step x n_test_takers

data_train_ = data_train[None, :, :].repeat(n_mc_e_step, 1, 1)
mask = ~torch.isnan(data_train_)


def closure():
    optim.zero_grad()
    z = torch.matmul(embed_train, weights).squeeze()
    probs = torch.sigmoid(thetas[:, :, None] + z[None, None, :])
    loss = -Bernoulli(probs=probs[mask]).log_prob(data_train_[mask]).mean()
    loss.backward()
    return loss


pbar = tqdm(range(1000))
for iteration in pbar:
    if iteration > 0:
        previous_weights = weights.clone()
        previous_loss = loss.clone()

    loss = optim.step(closure)

    if iteration > 0:
        d_loss = (previous_loss - loss).item()
        d_w = torch.norm(previous_weights - weights, p=2).item()
        max_grad = optim.param_groups[0]["params"][0].grad.max().abs().item()
        pbar.set_postfix({"max_grad": max_grad, "d_w": d_w, "d_loss": d_loss})
        if d_loss < tol and d_w < tol and max_grad < tol:
            break

# compute the spearman correlation
z_train = torch.matmul(embed_train, weights).squeeze()
z_test = torch.matmul(embed_test, weights).squeeze()

print(
    "Spearman with CTT on train set: ", spearman_corrcoef(z_train, data_train.mean(0))
)
print("Spearman with CTT on test set: ", spearman_corrcoef(z_test, data_test.mean(0)))

# Spearman with CTT on train set:  tensor(0.7814, device='cuda:0')
# Spearman with CTT on test set:  tensor(0.4446, device='cuda:0')
# :(
