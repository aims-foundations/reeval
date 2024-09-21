import os
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("..")
from utils import item_response_fn_1PL, set_seed

if __name__ == "__main__":
    set_seed(42)
    
    question_num = 1000
    testtaker_num = 500

    z_true = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    theta_true = torch.normal(mean=0.0, std=1.0, size=(testtaker_num,))
    
    response_matrix = np.zeros((testtaker_num, question_num))
    for i in range(testtaker_num):
        for j in range(question_num):
            prob = item_response_fn_1PL(z_true[j], theta_true[i])
            response_matrix[i, j] = torch.distributions.Bernoulli(prob).sample()
            
    output_dir = "../../data/pre_calibration/synthetic"
    os.makedirs(output_dir, exist_ok=True)

    matrix_df = pd.DataFrame(response_matrix.astype(int))
    matrix_df.insert(0, '', [f'testtaker_{i}' for i in range(testtaker_num)])
    matrix_df.to_csv(f"{output_dir}/matrix.csv", index=False)

    theta_true_df = pd.DataFrame(theta_true.numpy(), columns=["theta"])
    theta_true_df.to_csv(f"{output_dir}/theta_true.csv", index=False)
    
    z_true_df = pd.DataFrame(z_true.numpy(), columns=["z"])
    z_true_df.to_csv(f"{output_dir}/z_true.csv", index=False)
    