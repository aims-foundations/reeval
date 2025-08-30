from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from simpler_model import fit_logistic_mf
from grab_data import get_new_benchmark
torch.manual_seed(0)

def estimate_theta(theta, asked_ys, asked_zs):
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    asked_ys = torch.tensor(asked_ys)
    asked_zs = torch.tensor(asked_zs)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    for iteration in range(100):
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_theta = torch.norm(previous_theta - theta, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                break
    
    return theta.detach()

def compute_fisher_info(theta, remain_zs):
    p = torch.sigmoid(theta[:, None] + remain_zs[None, :])
    return p * (1 - p)




if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_new_benchmark(seed)
    
    
    theta_true = 1.5
    num_item_pool = 1000
    num_steps = 50
    zs = torch.randn(num_item_pool)
    ys = Bernoulli(probs=torch.sigmoid(theta_true + zs)).sample()

    # random
    random_thata_hat = torch.zeros((1,))
    random_thata_hats = [random_thata_hat]
    random_asked_zs = []
    random_asked_ys = []
    for i in tqdm(range(num_steps)):
        random_asked_zs.append(zs[i])
        random_asked_ys.append(ys[i])
        random_thata_hat = estimate_theta(random_thata_hat, random_asked_ys, random_asked_zs)
        random_thata_hats.append(random_thata_hat)
    
    # adaptive
    adaptive_thata_hat = torch.zeros((1,))
    adaptive_thata_hats = [adaptive_thata_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = zs.clone()
    remain_ys = ys.clone()
    for _ in tqdm(range(num_steps)):
        fisher_info = compute_fisher_info(adaptive_thata_hat, remain_zs)
        next_item = torch.argmax(fisher_info)
        adaptive_asked_zs.append(remain_zs[next_item])
        adaptive_asked_ys.append(remain_ys[next_item])
        adaptive_thata_hat = estimate_theta(adaptive_thata_hat, adaptive_asked_ys, adaptive_asked_zs)
        adaptive_thata_hats.append(adaptive_thata_hat)
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(num_steps+1), (np.array(random_thata_hats) - theta_true) ** 2, label="random")
    plt.plot(np.arange(num_steps+1), (np.array(adaptive_thata_hats) - theta_true) ** 2, label="adaptive")
    plt.ylabel("MSE")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()