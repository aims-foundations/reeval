from argparse import ArgumentParser
import os
import torch
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import set_seed, perform_t_test

def inverse_item_response_fn_1PL(y,theta):
    return -theta - torch.log((1 - y) / y)

def beta_params_from_mode(mode, concentration=10):
    alpha = mode * (concentration - 2) + 1
    beta_param = (1 - mode) * (concentration - 2) + 1
    return alpha, beta_param

def construct_Z(Y_bar, question_num, theta):
    alpha, beta = beta_params_from_mode(Y_bar)
    beta_dist = torch.distributions.Beta(alpha, beta)
    Y = beta_dist.sample((question_num,))
    Z = inverse_item_response_fn_1PL(Y, theta)
    return Z

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--Y_bar", type=float, default=0.7)
    parser.add_argument("--theta_1", type=float, default=1)
    parser.add_argument("--theta_2", type=float, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    Z_1 = construct_Z(args.Y_bar, args.question_num, args.theta_1)
    Z_2 = construct_Z(args.Y_bar, args.question_num, args.theta_2)

    testtaker1 = SimulatedTestTaker(theta=args.theta_1, model="1PL")
    testtaker2 = SimulatedTestTaker(theta=args.theta_2, model="1PL")
    
    asked_question_list = list(range(args.question_num))
    
    asked_answer_list_1 = []
    for i in range(args.question_num):
        asked_answer_list_1.append(testtaker1.ask(Z_1, i))
    
    asked_answer_list_2 = []
    for i in range(args.question_num):
        asked_answer_list_2.append(testtaker2.ask(Z_2, i))
    
    # CTT
    print("CTT")
    CTT_1_mean = sum(asked_answer_list_1) / len(asked_answer_list_1)
    CTT_1_std = torch.std(torch.stack(asked_answer_list_1))
    print(f"CTT score_1 mean: {CTT_1_mean}")
    print(f"CTT score_1 std: {CTT_1_std}")
    
    CTT_2_mean = sum(asked_answer_list_2) / len(asked_answer_list_2)
    CTT_2_std = torch.std(torch.stack(asked_answer_list_2))
    print(f"CTT score_2 mean: {CTT_2_mean}")
    print(f"CTT score_2 std: {CTT_2_std}")
    
    perform_t_test(asked_answer_list_1, asked_answer_list_2, label="CTT")
    
    # IRT via MCMC
    print("\nIRT via MCMC")
    asked_question_list = jnp.array(asked_question_list)
    
    asked_answer_list_1 = jnp.array(asked_answer_list_1)
    Z_1 = jnp.array(Z_1)
    mean_theta_1, std_theta_1, theta_1_samples = fit_theta_mcmc(Z_1, asked_question_list, asked_answer_list_1)
    print(f"IRT theta_1 mean: {mean_theta_1}")
    print(f"IRT theta_1 std: {std_theta_1}")
    
    asked_answer_list_2 = jnp.array(asked_answer_list_2)
    Z_2 = jnp.array(Z_2)
    mean_theta_2, std_theta_2, theta_2_samples= fit_theta_mcmc(Z_2, asked_question_list, asked_answer_list_2)
    print(f"IRT theta_2 mean: {mean_theta_2}")
    print(f"IRT theta_2 std: {std_theta_2}")
    
    perform_t_test(theta_1_samples, theta_2_samples, label="IRT")
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(theta_1_samples, bins=30, density=True, alpha=0.6, color='blue', label='theta_1_samples')
    plt.hist(theta_2_samples, bins=30, density=True, alpha=0.6, color='green', label='theta_2_samples')
    plt.title('Histogram of Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(Z_1, bins=30, density=True, alpha=0.6, color='red', label='Z_1')
    plt.hist(Z_2, bins=30, density=True, alpha=0.6, color='orange', label='Z_2')
    plt.title('Histogram of Z')
    plt.xlabel('Z')
    plt.ylabel('Density')
    plt.legend()

    fig_dir = '../plot/synthetic'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f'{fig_dir}/test_dependent_simulation.png')