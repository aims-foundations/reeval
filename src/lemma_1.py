from argparse import ArgumentParser
import random
from utils import calculate_1d_wasserstein_distance, set_seed
import numpy as np
from utils import item_response_fn_1PL
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
from test_dependent_simulation import construct_Z
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc

def fit_gaussian_to_Z(Z):
    mean = torch.mean(Z).item()
    std = torch.std(Z, unbiased=True).item()
    return mean, std

def gaussian_diff_std(std1, std2):
    return jnp.sqrt(std1**2 + std2**2)

def ctt_diff_scatter(list1, list2):
    diffs = []
    assert len(list1) == len(list2)
    for _ in range(10):
        sample_size = int(0.9 * len(list1))
        sampled_indices = random.sample(range(len(list1)), sample_size)
        mean1 = torch.mean(torch.tensor([list1[i] for i in sampled_indices]))
        mean2 = torch.mean(torch.tensor([list2[i] for i in sampled_indices]))
        diff = abs((mean1 - mean2).item())
        diffs.append(diff)
    return diffs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--Y_bar", type=float, default=0.9)
    parser.add_argument("--theta_dumb", type=float, default=1)
    parser.add_argument("--theta_smart", type=float, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    plt.rcParams.update({'font.size': 10})
    
    testtaker_dumb = SimulatedTestTaker(theta=args.theta_dumb, model="1PL")
    testtaker_smart = SimulatedTestTaker(theta=args.theta_smart, model="1PL")

    Z_hard = construct_Z(args.Y_bar, args.question_num, args.theta_smart)
    Z_hard_jnp = jnp.array(Z_hard)
    mean_Z_hard, std_Z_hard = fit_gaussian_to_Z(Z_hard)
    print(f"mean_Z_hard: {mean_Z_hard}")
    
    asked_question_list = list(range(args.question_num))
    asked_question_list_jnp = jnp.array(asked_question_list)
    
    asked_answer_list_hard_to_smart = []
    for i in range(len(asked_question_list)):
        asked_answer_list_hard_to_smart.append(
            testtaker_smart.ask(Z_hard, i)
        )
    asked_answer_list_hard_to_smart_jnp = jnp.array(
        asked_answer_list_hard_to_smart
    )
    
    CTT_theta_smart_mean = sum(asked_answer_list_hard_to_smart) / len(asked_answer_list_hard_to_smart)
    
    IRT_theta_smart_mean, IRT_theta_smart_std, _ = fit_theta_mcmc(
        Z_hard_jnp,
        asked_question_list_jnp,
        asked_answer_list_hard_to_smart_jnp
    )
    
    wd_Z_list = []
    CTT_diff_list = []
    CTT_diff_scatter_list = []
    IRT_diff_mean_list = []
    IRT_diff_std_list = []
    
    # Z = 3 is easy, Z  = -3 is hard
    diff_max = 3 - mean_Z_hard
    for diff_mean in np.arange(0, diff_max, 0.1):
        mean_Z_easy = mean_Z_hard + diff_mean
        std_Z_easy = std_Z_hard
        print(f"mean_Z_easy: {mean_Z_easy}")
        
        Z_easy = torch.normal(mean=mean_Z_easy, std=std_Z_easy, size=(args.question_num,))
        Z_easy_jnp = jnp.array(Z_easy)
        
        wd_Z = calculate_1d_wasserstein_distance(Z_easy, Z_hard)
        wd_Z_list.append(wd_Z)
        
        asked_answer_list_easy_to_dumb = []
        for i in range(len(asked_question_list)):
            asked_answer_list_easy_to_dumb.append(
                testtaker_dumb.ask(Z_easy, i)
            )
        asked_answer_list_easy_to_dumb_jnp = jnp.array(
            asked_answer_list_easy_to_dumb
        )
        
        CTT_theta_dumb_mean = sum(asked_answer_list_easy_to_dumb) / len(asked_answer_list_easy_to_dumb)
        CTT_diff_list.append(abs(CTT_theta_smart_mean - CTT_theta_dumb_mean))
        CTT_diff_scatter_list.append(ctt_diff_scatter(
            asked_answer_list_hard_to_smart, asked_answer_list_easy_to_dumb
        ))
        
        IRT_theta_dumb_mean, IRT_theta_dumb_std, _ = fit_theta_mcmc(
            Z_easy_jnp,
            asked_question_list_jnp,
            asked_answer_list_easy_to_dumb_jnp
        )
        IRT_diff_mean_list.append(abs(IRT_theta_smart_mean - IRT_theta_dumb_mean))
        IRT_diff_std_list.append(gaussian_diff_std(IRT_theta_smart_std, IRT_theta_dumb_std))
    
    wd_Z_list_repeated = [w for w in wd_Z_list for _ in range(10)]
    CTT_diff_scatter_list_flatten = [item for sublist in CTT_diff_scatter_list for item in sublist]
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wasserstein Distance in Z')
    ax1.set_ylabel('CTT difference', color='tab:blue')
    ax1.plot(wd_Z_list, CTT_diff_list, color='tab:blue', label='CTT_diff_list')
    ax1.scatter(wd_Z_list_repeated, CTT_diff_scatter_list, color='lightblue', label='CTT_diff_scatter_list')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('IRT difference', color='tab:red')
    ax2.plot(wd_Z_list, IRT_diff_mean_list, color='tab:red', label='IRT_diff_list')
    IRT_diff_mean_list = jnp.array(IRT_diff_mean_list)
    IRT_diff_std_list = jnp.array(IRT_diff_std_list)
    ax2.fill_between(
        wd_Z_list, 
        IRT_diff_mean_list - IRT_diff_std_list, 
        IRT_diff_mean_list + IRT_diff_std_list, 
        alpha=0.2
    )
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 3)

    fig.tight_layout()
    plt.savefig('../plot/synthetic/lemma_1.png')
