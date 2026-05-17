import pickle

import pandas as pd
import torch
from huggingface_hub import snapshot_download

########################################################################################
# New response matrix
########################################################################################
dataset = "mmlu/mmlu"
data_folder = snapshot_download(
    repo_id="stair-lab/reeval_matrices", repo_type="dataset"
)
question_keys = pd.read_csv(f"{data_folder}/{dataset}/question_keys.csv")
model_keys = pd.read_csv(f"{data_folder}/{dataset}/model_keys.csv")
response_matrix = torch.load(f"{data_folder}/{dataset}/response_matrix.pt")

print(
    "num test takers: ",
    response_matrix.shape[0],
    "num questions: ",
    response_matrix.shape[1],
    "num correct: ",
    (response_matrix == 1).sum().sum(),
    "num incorrect: ",
    (response_matrix == 0).sum().sum(),
    "num skipped: ",
    (response_matrix == -1).sum().sum(),
)

########################################################################################
# New long form data
########################################################################################
dataset = "classic"
data_folder = snapshot_download(repo_id="stair-lab/reeval_csv", repo_type="dataset")
df_responses = pickle.load(open(f"{data_folder}/{dataset}/responses.pkl", "rb"))
df_responses.to_csv("responses_long.csv", index=False)

instances = pd.read_csv(f"{data_folder}/{dataset}/instances.csv")
instances.to_csv("instances.csv", index=False)


########################################################################################
# Old response matrix
########################################################################################
data_folder = snapshot_download(
    repo_id="stair-lab/reeval_responses", repo_type="dataset"
)
question_keys = pd.read_csv(f"{data_folder}/mmlu/question_keys.csv")
model_keys = pd.read_csv(f"{data_folder}/mmlu/model_keys.csv")
response_matrix = torch.load(f"{data_folder}/mmlu/response_matrix.pt").int()

print(
    "num test takers: ",
    response_matrix.shape[0],
    "num questions: ",
    response_matrix.shape[1],
    "num correct: ",
    (response_matrix == 1).sum().sum(),
    "num incorrect: ",
    (response_matrix == 0).sum().sum(),
    "num skipped: ",
    (response_matrix == -1).sum().sum(),
)

# drop columns that have all -1, 0, or 1
for i in torch.unique(response_matrix):
    response_matrix = response_matrix[:, (response_matrix != i.item()).any(0)]

# save to csv
response_matrix_df = pd.DataFrame(response_matrix.numpy())
response_matrix_df.to_csv("response_matrix.csv", index=False, header=False)
