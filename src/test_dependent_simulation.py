from argparse import ArgumentParser
import torch
from testtaker import SimulatedTestTaker, RealTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.style.use('seaborn-v0_8-paper')
from utils import set_seed, perform_t_test
import pandas as pd

def inverse_item_response_fn_1PL(y,theta):
    y = torch.tensor(y, dtype=torch.float32)
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

def sample_real_subsets(text_list, z3_list, Y_bar, subset_size):
    z3_tensor = torch.tensor(z3_list)
    z3_sorted, indices = torch.sort(z3_tensor)
    text_sorted = [text_list[i] for i in indices.tolist()]
    mean_all = z3_sorted.mean().item()
    std_all = z3_sorted.std().item()
    print(f"mean of all z3 values: {mean_all}")
    print(f"std of all z3 values: {std_all}")

    a = inverse_item_response_fn_1PL(Y_bar, 1).item()
    b = inverse_item_response_fn_1PL(Y_bar, 2).item()

    subset1_probs = torch.exp(-0.5 * ((z3_sorted - a) / (std_all / 2)) ** 2)
    subset1_probs /= subset1_probs.sum()
    subset2_probs = torch.exp(-0.5 * ((z3_sorted - b) / (std_all / 2)) ** 2)
    subset2_probs /= subset2_probs.sum()

    subset1_indices = torch.multinomial(subset1_probs, subset_size, replacement=False)
    subset2_indices = torch.multinomial(subset2_probs, subset_size, replacement=False)

    subset1_z3_list = z3_sorted[subset1_indices].tolist()
    subset2_z3_list = z3_sorted[subset2_indices].tolist()

    subset1_text_list = [text_sorted[i] for i in subset1_indices]
    subset2_text_list = [text_sorted[i] for i in subset2_indices]

    return subset1_z3_list, subset1_text_list, subset2_z3_list, subset2_text_list, subset1_indices, subset2_indices


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_type", type=str, required=True) # synthetic or real
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--Y_bar", type=float, default=0.7)
    # synthetic data
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--theta_1", type=float, default=1)
    parser.add_argument("--theta_2", type=float, default=2)
    # real data
    parser.add_argument("--subset_size", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.data_type == "synthetic":
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
    
    elif args.data_type == "real":
        if args.dataset == "airbench":
            z3_df = pd.read_csv('../data/real/irt_result/appendix1/Z/all_1PL_Z_clean.csv')
            index_search_df = pd.read_csv('../data/real/response_matrix/appendix1/index_search.csv')
            testtaker1_string = "Qwen_Qwen2-72B-Instruct"
            testtaker2_string = "meta_llama-3-8b-chat"
            
            z3_list = z3_df['z3'].tolist()
            filtered_index_search_df = index_search_df[index_search_df['is_deleted'] != 1]
            text_list = filtered_index_search_df['text'].tolist()
            assert len(z3_list) == len(text_list)
            
            Z_1, subset1_text_list, Z_2, subset2_text_list, _, _ = \
                sample_real_subsets(text_list, z3_list, args.Y_bar, args.subset_size)
                
            testtaker1 = RealTestTaker(subset1_text_list, model_string=testtaker1_string)
            testtaker2 = RealTestTaker(subset2_text_list, model_string=testtaker2_string)
            
            asked_question_list = list(range(args.subset_size))
            
            asked_answer_list_1 = []
            for i in range(args.subset_size):
                asked_answer_list_1.append(testtaker1.ask(Z_1, i))
            
            asked_answer_list_2 = []
            for i in range(args.subset_size):
                asked_answer_list_2.append(testtaker2.ask(Z_2, i))
                
        elif args.dataset == "mmlu":
            z3_df = pd.read_csv('../data/real/irt_result/appendix1_mmlu/Z/pyMLE_1PL_Z.csv')
            index_search_df = pd.read_csv('../data/real/response_matrix/appendix1_mmlu/non_mask_index_search.csv')
            testtaker1_string = "anthropic/claude-3-haiku-20240307"
            testtaker2_string = "meta/llama-3-70b"
            
            z3_list = z3_df['z3'].tolist()
            filtered_index_search_df = index_search_df[index_search_df['is_deleted'] != 1]
            text_list = filtered_index_search_df['text'].tolist()
            assert len(z3_list) == len(text_list)
            
            Z_1, _, Z_2, _, subset1_indices, subset2_indices = \
                sample_real_subsets(text_list, z3_list, args.Y_bar, args.subset_size)
            
            subset1_indices = subset1_indices.numpy()
            subset2_indices = subset2_indices.numpy()
            
            asked_question_list = list(range(args.subset_size))
            
            df = pd.read_csv('../data/real/response_matrix/appendix1_mmlu/two_model_answer.csv')
            asked_answer_list_1 = df.iloc[subset1_indices, 0].tolist()
            asked_answer_list_2 = df.iloc[subset2_indices, 1].tolist()
            asked_answer_list_1 = [torch.tensor(v, dtype=torch.float32) for v in asked_answer_list_1]
            asked_answer_list_2 = [torch.tensor(v, dtype=torch.float32) for v in asked_answer_list_2]
            assert len(asked_answer_list_1) == len(asked_answer_list_2) == len(asked_question_list)
        
        elif args.dataset == "syn_rea":
            z3_df = pd.read_csv('../data/real/irt_result/appendix1_syn_rea/Z/pyMLE_1PL_Z.csv')
            index_search_df = pd.read_csv('../data/real/response_matrix/appendix1_syn_rea/mask_index_search.csv')
            testtaker1_string = "openai/code-cushman-001"
            testtaker2_string = "ai21/j2-jumbo"
            
            z3_list = z3_df['z3'].tolist()
            filtered_index_search_df = index_search_df[index_search_df['is_deleted'] != 1]
            text_list = filtered_index_search_df['text'].tolist()
            assert len(z3_list) == len(text_list)
            
            Z_1, _, Z_2, _, subset1_indices, subset2_indices = \
                sample_real_subsets(text_list, z3_list, args.Y_bar, args.subset_size)
            
            subset1_indices = subset1_indices.numpy()
            subset2_indices = subset2_indices.numpy()
            
            asked_question_list = list(range(args.subset_size))
            
            df = pd.read_csv('../data/real/response_matrix/appendix1_syn_rea/two_model_answer.csv')
            asked_answer_list_1 = df.iloc[subset1_indices, 0].tolist()
            asked_answer_list_2 = df.iloc[subset2_indices, 1].tolist()
            asked_answer_list_1 = [torch.tensor(v, dtype=torch.float32) for v in asked_answer_list_1]
            asked_answer_list_2 = [torch.tensor(v, dtype=torch.float32) for v in asked_answer_list_2]
            assert len(asked_answer_list_1) == len(asked_answer_list_2) == len(asked_question_list)

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
    plt.hist(theta_1_samples, bins=30, density=True, alpha=0.4)
    plt.hist(theta_2_samples, bins=30, density=True, alpha=0.4)
    plt.xlabel(r'$\theta$', fontsize=25)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplot(1, 2, 2)
    plt.hist(Z_1, bins=30, density=True, alpha=0.4)
    plt.hist(Z_2, bins=30, density=True, alpha=0.4)
    plt.xlabel(r'$z$')
    plt.xlabel(r'$z$', fontsize=25)
    plt.tick_params(axis='both', labelsize=16)

    if args.data_type == "synthetic":
        plt.savefig(f'../plot/synthetic/test_dependent_simulation.png', dpi=300, bbox_inches='tight')
    elif args.data_type == "real":
        plt.savefig(f'../plot/real/{args.dataset}_test_dependent_simulation.png', dpi=300, bbox_inches='tight')