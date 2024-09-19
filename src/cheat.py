from argparse import ArgumentParser
import os
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import torch
from utils import item_response_fn_1PL_cheat, set_seed, perform_t_test
from testtaker import CheatingTestTaker, RealTestTaker
import matplotlib.pyplot as plt

def model(Z, asked_question_list, asked_answer_list, contamination):
    theta_hat_true = numpyro.sample("theta_hat_true", dist.Normal(0.0, 1.0)) # prior_true
    theta_hat_cheat = numpyro.sample("theta_hat_cheat", dist.Normal(0.0, 1.0)) # prior_cheat
    Z_asked = Z[asked_question_list]
    contamination_asked = contamination[asked_question_list]
    probs = item_response_fn_1PL_cheat(
        Z_asked, contamination_asked, theta_hat_true, theta_hat_cheat, datatype="jnp"
        )
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)

def fit_cheat_theta_mcmc(
    Z, asked_question_list, asked_answer_list, contamination, num_samples=9000, num_warmup=1000
    ):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        Z=Z,
        asked_question_list=asked_question_list,
        asked_answer_list=asked_answer_list,
        contamination=contamination,
    )
    mcmc.print_summary()
    
    theta_true_samples = mcmc.get_samples()["theta_hat_true"]
    mean_theta_true = jnp.mean(theta_true_samples)
    std_theta_true = jnp.std(theta_true_samples)
    
    theta_cheat_samples = mcmc.get_samples()["theta_hat_cheat"]
    mean_theta_cheat = jnp.mean(theta_cheat_samples)
    std_theta_cheat = jnp.std(theta_cheat_samples)
    
    return mean_theta_true, std_theta_true, theta_true_samples, \
        mean_theta_cheat, std_theta_cheat, theta_cheat_samples
        
def synthetic_main(testtaker_true_theta, testtaker_cheat_gain, z3, contamination, asked_question_list):
    testtaker = CheatingTestTaker(
        true_theta=testtaker_true_theta, cheat_gain=testtaker_cheat_gain, model="1PL"
    )
    true_theta, cheat_theta = testtaker.get_ability()
    print(f"true theta: {true_theta}")
    print(f"cheat theta: {cheat_theta}")

    asked_answer_list = []
    for i in range(args.question_num):
        asked_answer_list.append(testtaker.ask(z3, contamination, i))
    
    # CTT
    print("CTT")
    CTT_mean = np.mean(asked_answer_list)
    CTT_std = np.std(asked_answer_list)
    print(f"CTT score mean: {CTT_mean}")
    print(f"CTT score std: {CTT_std}")
        
    # IRT via MCMC
    print("\nIRT via MCMC")
    z3 = jnp.array(z3)
    contamination = jnp.array(contamination)
    asked_question_list = jnp.array(asked_question_list)
    asked_answer_list_jnp = jnp.array(asked_answer_list)
    
    mean_theta_true, std_theta_true, theta_true_samples, \
        mean_theta_cheat, std_theta_cheat, theta_cheat_samples \
            = fit_cheat_theta_mcmc(
                z3, asked_question_list, asked_answer_list_jnp, contamination
                )

    print(f"mean_theta_true: {mean_theta_true}")
    print(f"std_theta_true: {std_theta_true}")
    print(f"mean_theta_cheat: {mean_theta_cheat}")
    print(f"std_theta_cheat: {std_theta_cheat}")
    
    return asked_answer_list, theta_true_samples, theta_cheat_samples

def real_main():
    z3_df = pd.read_csv('../data/real/irt_result/appendix1/Z/all_1PL_Z_clean.csv')
    index_search_df = pd.read_csv('../data/real/response_matrix/appendix1/index_search.csv')

    z3_list = z3_df['z3'].tolist()
    filtered_index_search_df = index_search_df[index_search_df['is_deleted'] != 1]
    text_list = filtered_index_search_df['text'].tolist()
    assert len(z3_list) == len(text_list)
    
    Z_1, subset1_text_list, Z_2, subset2_text_list = \
        sample_real_subsets(text_list, z3_list, args.Y_bar, args.subset_size)
        
    testtaker1 = RealTestTaker(subset1_text_list, model_string="Qwen_Qwen2-72B-Instruct")
    testtaker2 = RealTestTaker(subset2_text_list, model_string="meta_llama-3-8b-chat")
    
    asked_question_list = list(range(args.subset_size))
    
    asked_answer_list_1 = []
    for i in range(args.subset_size):
        asked_answer_list_1.append(testtaker1.ask(Z_1, i))
    
    asked_answer_list_2 = []
    for i in range(args.subset_size):
        asked_answer_list_2.append(testtaker2.ask(Z_2, i))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--contamination_percent", type=float, default=0.8)
    parser.add_argument("--testtaker1_true_theta", type=float, default=0)
    parser.add_argument("--testtaker1_cheat_gain", type=float, default=1.5)
    parser.add_argument("--testtaker2_true_theta", type=float, default=-0.5)
    parser.add_argument("--testtaker2_cheat_gain", type=float, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    z3 = torch.normal(mean=0.0, std=1.0, size=(args.question_num,))
    
    asked_question_list = list(range(args.question_num))
    
    num_ones = int(args.question_num * args.contamination_percent)
    contamination = torch.cat([
        torch.ones(num_ones),
        torch.zeros(args.question_num - num_ones)
        ])
    contamination = contamination[torch.randperm(args.question_num)]
    
    print("Test Taker 1")
    asked_answer_list_1, theta_true_samples_1, theta_cheat_samples_1 = synthetic_main(
        args.testtaker1_true_theta, 
        args.testtaker1_cheat_gain, 
        z3, 
        contamination, 
        asked_question_list
    )
    
    print("\n\n\nTest Taker 2")
    asked_answer_list_2, theta_true_samples_2, theta_cheat_samples_2 = synthetic_main(
        args.testtaker2_true_theta, 
        args.testtaker2_cheat_gain,
        z3,
        contamination,
        asked_question_list
    )
    
    print("\n\n\n")
    perform_t_test(asked_answer_list_1, asked_answer_list_2, label="CTT")
    perform_t_test(theta_true_samples_1, theta_true_samples_2, label="IRT")
    
    plt.figure(figsize=(12, 6))

    # Test Taker 1
    plt.subplot(1, 2, 1)
    plt.hist(theta_true_samples_1, bins=30, density=True, alpha=0.6, color='blue', label='Theta True')
    plt.hist(theta_cheat_samples_1, bins=30, density=True, alpha=0.6, color='green', label='Theta Cheat')
    plt.title('Histogram of Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()
    
    # Test Taker 2
    plt.subplot(1, 2, 2)
    plt.hist(theta_true_samples_2, bins=30, density=True, alpha=0.6, color='blue', label='Theta True')
    plt.hist(theta_cheat_samples_2, bins=30, density=True, alpha=0.6, color='green', label='Theta Cheat')
    plt.title('Histogram of Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    fig_dir = "../plot/synthetic"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f'{fig_dir}/cheat.png')