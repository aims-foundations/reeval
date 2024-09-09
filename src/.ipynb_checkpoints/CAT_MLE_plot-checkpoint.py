import matplotlib.pyplot as plt
import torch
from utils import load_state, item_response_fn_1PL
from tqdm import tqdm

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
    for i in tqdm(range(asked_question_lists.shape[1])):
        Remp = compute_Remp(true_thetas, asked_question_lists[:,:i+1], z3)
        mse = compute_mse(theta_hats[:,i], true_thetas)
        bias = compute_bias(theta_hats[:,i], true_thetas)
        Remp_list.append(Remp)
        mse_list.append(mse)
        bias_list.append(bias)
    return Remp_list, mse_list, bias_list

if __name__ == '__main__':
    pre_cat_state = load_state("../data/synthetic/CAT_MLE/pre_cat.pt")
    z3 = pre_cat_state['z3']
    
    question_num = z3.shape[0]
    subset_question_num = pre_cat_state['subset_question_num']
    testtaker_num = 220

    true_thetas = pre_cat_state['true_thetas'][:testtaker_num]

    random_asked_question_lists = []
    fisher_asked_question_lists = []
    random_theta_hats = []
    fisher_theta_hats = []
    for serial in range(testtaker_num):
        random_state_path = f"../data/synthetic/CAT_MLE/sweep/mle_{serial}_random.pt"
        fisher_state_path = f"../data/synthetic/CAT_MLE/sweep/mle_{serial}_fisher.pt"
        random_state = load_state(random_state_path)
        fisher_state = load_state(fisher_state_path)

        random_asked_question_list=random_state['asked_question_list']
        random_theta_hat=random_state['theta_hats']
        fisher_asked_question_list=fisher_state['asked_question_list']
        fisher_theta_hat=fisher_state['theta_hats']
        
        random_asked_question_lists.append(random_asked_question_list[:-1])
        random_theta_hats.append(random_theta_hat)
        fisher_asked_question_lists.append(fisher_asked_question_list[:-1])
        fisher_theta_hats.append(fisher_theta_hat)

    random_theta_hats = torch.tensor(random_theta_hats)
    fisher_theta_hats = torch.tensor(fisher_theta_hats)
    random_asked_question_lists = torch.tensor(random_asked_question_lists)
    fisher_asked_question_lists = torch.tensor(fisher_asked_question_lists)

    print(random_theta_hats.shape)
    print(random_asked_question_lists.shape)
    
    Remp_random, mse_random, bias_random = compute_3_metrics(
        true_thetas, random_theta_hats, random_asked_question_lists, z3
    )
    Remp_fisher, mse_fisher, bias_fisher = compute_3_metrics(
        true_thetas, fisher_theta_hats, fisher_asked_question_lists, z3
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
