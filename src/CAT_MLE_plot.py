import matplotlib.pyplot as plt
import torch
from utils import load_state, item_response_fn_1PL

def compute_sem(single_theta, asked_question_list, z3):
    asked_z3 = z3[asked_question_list]
    I = 0
    for j in range(asked_z3.shape[0]):
        P = item_response_fn_1PL(asked_z3[j], single_theta)
        I += P * (1-P)
    return 1 / torch.sqrt(I)

def compute_Remp(true_thetas, asked_question_lists, z3):
    N = true_thetas.shape[0]
    sum_sem_square = 0
    for i in range(N):
        sum_sem_square += compute_sem(true_thetas[i], asked_question_lists[i], z3) ** 2
    
    avg_theta = torch.mean(true_thetas)
    sum_denominator = 0
    for i in range(N):
        sum_denominator += (true_thetas[i] - avg_theta) ** 2
    
    return 1- ((1/N) * sum_sem_square) / ((1/(N-1)) * sum_denominator)

def compute_mse(thetas, true_thetas):
    mse = 0
    assert thetas.shape == true_thetas.shape
    N = thetas.shape[0]
    for i in range(N):
        mse += (thetas[i] - true_thetas[i]) ** 2
    return mse / N

def compute_bias(thetas, true_thetas):
    bias = 0
    assert thetas.shape == true_thetas.shape
    N = thetas.shape[0]
    for i in range(N):
        bias += thetas[i] - true_thetas[i]
    return bias

def compute_3_metrics(true_thetas, theta_hats, asked_question_lists, z3):
    Remp_list = []
    mse_list = []
    bias_list = []
    for i in range(asked_question_lists.shape[1]):
        Remp = compute_Remp(true_thetas, asked_question_lists[:,:i+1], z3)
        mse = compute_mse(theta_hats[:,i], true_thetas)
        bias = compute_bias(theta_hats[:,i], true_thetas)
        Remp_list.append(Remp)
        mse_list.append(mse)
        bias_list.append(bias)
    return Remp_list, mse_list, bias_list

if __name__ == '__main__':
    question_num = 5000
    subset_question_num = 500
    testtaker_num = 50
    state_path = f"../data/synthetic/CAT/mle_{question_num}_{subset_question_num}_{testtaker_num}.pt"
    state = load_state(state_path)
        
    z3 = state['z3'] # [500]
    true_thetas = state['true_thetas'] # [10]
    theta_means = torch.tensor(state['theta_means']) # [10, 50]
    asked_question_lists = torch.tensor(state['asked_question_lists'])[:,:50]  # [10, 50]
    
    true_thetas_random = true_thetas[::2]
    theta_means_random = theta_means[::2]
    asked_question_lists_random = asked_question_lists[::2]
    Remp_random, mse_random, bias_random = compute_3_metrics(
        true_thetas_random, theta_means_random, asked_question_lists_random, z3
    )
    
    true_thetas_fisher = true_thetas[1::2]
    theta_means_fisher = theta_means[1::2]
    asked_question_lists_fisher = asked_question_lists[1::2]
    Remp_fisher, mse_fisher, bias_fisher = compute_3_metrics(
        true_thetas_fisher, theta_means_fisher, asked_question_lists_fisher, z3
    )
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(Remp_random, label='random')
    plt.plot(Remp_fisher, label='fisher')
    plt.title('Remp')
    plt.legend()
        
    plt.subplot(1, 3, 2)
    plt.plot(mse_random, label='random')
    plt.plot(mse_fisher, label='fisher')
    plt.title('MSE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(bias_random, label='random')
    plt.plot(bias_fisher, label='fisher')
    plt.title('Bias')
    plt.legend()
    
    plt.savefig('../plot/synthetic/cat_3_metrics.png')    
