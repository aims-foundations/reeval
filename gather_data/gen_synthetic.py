import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
from utils import item_response_fn_1PL, item_response_fn_2PL, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["1PL", "2PL"])
    args = parser.parse_args()

    set_seed(42)

    question_num = 500
    testtaker_num = 500
    if args.model == "1PL":
        z_true = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    elif args.model == "2PL":
        z2_true = torch.distributions.LogNormal(0.0, 0.5).sample((question_num,))
        z3_true = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    theta_true = torch.normal(mean=0.0, std=1.0, size=(testtaker_num,))

    # response_matrix = np.zeros((testtaker_num, question_num))
    # for i in range(testtaker_num):
    #     for j in range(question_num):
    #         if args.model == "1PL":
    #             prob = item_response_fn_1PL(z_true[j], theta_true[i])
    #         elif args.model == "2PL":
    #             prob = item_response_fn_2PL(z2_true[j], z3_true[j], theta_true[i])
    #         response = torch.distributions.Bernoulli(prob).sample()
    #         # if np.random.rand() < 0.2:
    #         #     response = 1 - response
    #         response_matrix[i, j] = response

    theta_true_matrix = theta_true.unsqueeze(1)
    z2_true_matrix = z2_true.unsqueeze(0)
    z3_true_matrix = z3_true.unsqueeze(0)
    prob_matrix = item_response_fn_2PL(
        z2_true_matrix, z3_true_matrix, theta_true_matrix
    )
    response_matrix = (prob_matrix > 0.5).int()

    output_dir = f"../../data/pre_calibration/synthetic_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    matrix_df = pd.DataFrame(response_matrix)
    matrix_df.insert(0, "", [f"testtaker_{i}" for i in range(testtaker_num)])
    matrix_df.to_csv(f"{output_dir}/matrix.csv", index=False)

    theta_true_df = pd.DataFrame(theta_true.numpy(), columns=["theta"])
    theta_true_df.to_csv(f"{output_dir}/theta_true.csv", index=False)

    if args.model == "1PL":
        z_true_df = pd.DataFrame(
            {
                "z": z_true.numpy(),
            }
        )
    elif args.model == "2PL":
        z_true_df = pd.DataFrame(
            {
                "z2": z2_true.numpy(),
                "z3": z3_true.numpy(),
            }
        )
    z_true_df.to_csv(f"{output_dir}/z_true.csv", index=False)
