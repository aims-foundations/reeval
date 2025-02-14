from tqdm import tqdm
import pandas as pd
import torch
from torchmetrics.functional import spearman_corrcoef
from torch.distributions import Bernoulli
from huggingface_hub import HfApi, snapshot_download

torch.manual_seed(0)

data_folder = snapshot_download(repo_id="stair-lab/reeval_another_repo", repo_type="dataset")
df = pd.read_parquet(f"{data_folder}/mmlu/data.parquet", engine="fastparquet")
question_keys = pd.read_parquet(f"{data_folder}/mmlu/question.parquet", engine="fastparquet")

subjects = df["subject"].unique()
subjects_subset = subjects[0:1]

df_subject = df[df["subject"].isin(subjects_subset)]
df_subject = df_subject[["model_id", "instance_id", "exact_match"]]
# create a pivot table with model_id as index, instance_id as columns, and exact_match as values
matrix = df_subject.pivot(index="model_id", columns="instance_id", values="exact_match")

question_keys_subject = question_keys[question_keys["subject"].isin(subjects_subset)]
# Make sure that the order of embedding and the column matches! 
assert matrix.columns.tolist() == question_keys_subject["instance_id"].values.tolist()

# drop the columns with only (0 and -1) or (1 and -1)
invalid_columns = matrix.columns[matrix.isin([0, -1]).all() | matrix.isin([1, -1]).all()]
invalid_instance_ids = invalid_columns.values
question_keys_subject = question_keys_subject.drop(index=invalid_instance_ids)
assert matrix.shape[1] == question_keys_subject.shape[0]

embed = torch.tensor(question_keys_subject["embedding"].values.tolist()).float()
data = torch.tensor(matrix.values).float()
n_test_takers, n_items = data.shape
linear_reg_paras = torch.nn.Parameter(torch.zeros(embed.shape[1], 1))
optim = torch.optim.LBFGS([linear_reg_paras], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

def closure():
    optim.zero_grad()
    z = torch.matmul(embed, linear_reg_paras).squeeze()
    probs = torch.sigmoid(thetas[:, :, None] + z[None, None, :])
    loss = -Bernoulli(probs=probs).log_prob(data).mean()
    loss.backward()
    return loss

thetas = torch.randn(500, n_test_takers)
# >>> 500 x n_test_takers

pbar = tqdm(range(100))
for iteration in pbar:
    z_temp = torch.matmul(embed, linear_reg_paras).squeeze()
    if iteration > 0:
        previous_z = z_temp.clone()
        previous_loss = loss.clone()
    
    loss = optim.step(closure)
    
    if iteration > 0:
        d_loss = previous_loss - loss
        d_z = torch.norm(previous_z - z_temp, p=2)
        grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
        pbar.set_postfix({"grad_norm": grad_norm, "d_z": d_z, "d_loss": d_loss})
        if d_loss < 1e-5 and d_z < 1e-5 and grad_norm < 1e-5:
            break

# item_difficulty = pd.read_csv(f"item_difficulty.csv", header=None)
# item_difficulty = torch.tensor(item_difficulty.values)[:, 1]

# compute the spearman correlation
print("Spearman with CTT: ", spearman_corrcoef(z_temp, data.mean(0)))
# print("Spearman with mirt: ", spearman_corrcoef(z, item_difficulty))
