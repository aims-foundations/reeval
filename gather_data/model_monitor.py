"""
How Is ChatGPT’s Behavior Changing over Time?
Lingjiao Chen, Matei Zaharia, James Zou
"""

# totally 2 models * 2 timepoints = 4 models, the row name of the response matrix is model_name-timepoint
# every question of a dataset have a question id, the column name of the response matrix is question_id
# the content of the response matrix: if model answer == ref answer, then 1, else 0

import os

import pandas as pd

generation_dir = "generation"
response_matrix_dir = "response_matrix"
os.makedirs(response_matrix_dir, exist_ok=True)

for filename in os.listdir(generation_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(generation_dir, filename)
        data = pd.read_csv(file_path)

        models = data["model"].unique()
        ids = data["id"].unique()

        response_matrix = pd.DataFrame(0, index=models, columns=ids)

        for _, row in data.iterrows():
            id_ = row["id"]
            answer = row["answer"]
            ref_answer = row["ref_answer"]

            if answer == ref_answer:
                response_matrix.at[row["model"], id_] = 1

        response_matrix_file_path = os.path.join(
            response_matrix_dir, f"mARC_EVALatrix_{filename}"
        )
        response_matrix.to_csv(response_matrix_file_path)
