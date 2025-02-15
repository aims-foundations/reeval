from tqdm import tqdm
import pandas as pd
import torch
from torch.optim import LBFGS
from torch.nn import Parameter
from torchmetrics.functional import spearman_corrcoef
from torch.distributions import Bernoulli
from huggingface_hub import HfApi, snapshot_download
import numpy as np

torch.manual_seed(0)
gpu_id = 0
tol = 1e-5

data_folder = snapshot_download(repo_id="stair-lab/reeval_another_repo", repo_type="dataset")
df = pd.read_parquet(f"{data_folder}/mmlu/data.parquet", engine="fastparquet")
question_keys = pd.read_parquet(f"{data_folder}/mmlu/question.parquet", engine="fastparquet")

subjects = df["subject"].unique()

np.random.seed(1) # 1
np.random.shuffle(subjects)
subjects_subset = subjects#[0:20]

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
invalid_columns = matrix.columns[matrix.isin([0, -1]).all() | matrix.isin([1, -1]).all()]
invalid_instance_ids = invalid_columns.values
question_keys_subject = question_keys_subject.drop(index=invalid_instance_ids)
matrix = matrix.drop(columns=invalid_instance_ids)

assert matrix.shape[1] == question_keys_subject.shape[0]
model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
embed = question_keys_subject[f"embedding_{model_name.replace('/', '_')}"].values.tolist()
embed = torch.tensor(embed, dtype=torch.float32, device="cuda")
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
optim = LBFGS([weights], lr=0.1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

thetas = torch.randn(n_mc_e_step, n_test_takers, device="cuda")
# >>> n_mc_e_step x n_test_takers

data_train_ = data_train[None, :, :].repeat(n_mc_e_step, 1, 1)
mask = ~torch.isnan(data_train_)

# kernel ridge regression with feature being `embed_train` and target being `torch.nanmean(data_train, dim=0)`
# the target is the average of the data_train
# the kernel is the sigmoid kernel
# the loss is the mean squared error between the prediction and the target
# the prediction is the sigmoid of the kernel ridge regression
# the sigmoid is applied element-wise

from sklearn.kernel_ridge import KernelRidge

# Convert the embeddings to a numpy array for scikit-learn.
X_train = embed_train.cpu().numpy()  # shape: (n_items, embedding_dim)
y_train = torch.nanmean(data_train, dim=0).cpu().numpy()  # shape: (n_items,)

# # Create and fit a Kernel Ridge Regression model with an RBF kernel.
# # Note: You may need to adjust alpha and gamma based on your data.
# # krr = KernelRidge(alpha=0.0001, kernel='rbf', gamma=1e-4)
# krr = KernelRidge(alpha=1e-5, kernel='sigmoid')
# krr.fit(X_train, y_train)

# # Predict the latent difficulty using the trained KRR model
# z_train_pred = krr.predict(X_train)  # shape: (n_items,)
# z_test_pred = krr.predict(embed_test.cpu().numpy())  # shape: (n_items,)

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(1024, 512),
                   activation='relu',
                   solver='adam',
                   alpha=0.1,
                   learning_rate_init=1e-4,
                   max_iter=300,
                   early_stopping=True,
                   validation_fraction=0.1,
                   n_iter_no_change=10)

mlp.fit(X_train, y_train)
z_train_pred = mlp.predict(X_train)
z_test_pred = mlp.predict(embed_test.cpu().numpy())

# ------------------------------
# Compute Spearman correlation
# ------------------------------
from scipy.stats import spearmanr

rho, pval = spearmanr(z_train_pred, y_train)
print("Spearman with CTT: ", rho)

y_test = torch.nanmean(data_test, dim=0).cpu().numpy()
rho, pval = spearmanr(z_test_pred, y_test)
print("Spearman with CTT on test set: ", rho)


# krr = KernelRidge(alpha=0.0001, kernel='sigmoid')
# Spearman with CTT:  0.658785803755998
# Spearman with CTT on test set:  0.6502665819181468