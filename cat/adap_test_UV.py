from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import AUROC
torch.manual_seed(0)

def estimate_theta(theta, asked_ys, asked_zs):
    def closure():
        optim.zero_grad()
        # theta: taker * fac
        # asked_zs: step * taker * fac
        # y.shape: item
        # probs = torch.sigmoid(theta @ asked_zs.T) # test_taker * step
        # print("theta.shape",theta.shape)
        # print("asked_zs.shape",asked_zs.shape)
        probs = torch.sigmoid(torch.einsum('nk,ink->in',theta,asked_zs)) # step * testtaker
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss
    # list of shape= test_taker
    # want: step * test_taker
    asked_ys = torch.stack(asked_ys)
    
    # step * test_taker * k
    asked_zs = torch.stack(asked_zs)
    # test_taker * k
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

def pred_mean_mse(U_preds,V, Y):
    #U_preds: step * rank
    # V: num_item * rank
    # (U_preds @ V.T): step* item
    prob = torch.sigmoid(U_preds @ V.T)
    pred_mean = prob.mean(axis = 1)
    # pred_mean: step
    # Y items
    mse = ((pred_mean - Y.mean())**2)
    return mse

def auc_for_list_Uhat(U_hat_list,V,Y):
    
    auroc = AUROC(task="binary")
    aucs = []
    for uhat in U_hat_list:
        prob = torch.sigmoid(uhat @ V.T)
        aucs.append(auroc(prob,Y))
        
    return aucs


def compute_fisher_info(U, V):
    p = torch.sigmoid(U @ V.T)
    return p * (1 - p)



if __name__ == "__main__":

    num_item_pool = 1000
    # num_steps = 50
    num_steps = 50
    k = 2
    num_test_taker, num_item_pool, K_true, K_fit = 100, 20000, k, k

    # Create a small synthetic dataset
    # theta
    U_true = torch.randn(num_test_taker, K_true)
    # zs
    V_true = torch.randn(num_item_pool, K_true)  
    # num_taker * 2 @ 2*  num_item_pool 
    logits = U_true @ V_true.T
    P = torch.sigmoid(logits)
    # for that unknown test taker, its true Y
    Y = torch.bernoulli(P)
    
    
    # zs = torch.randn(num_item_pool)
    # ys = Bernoulli(probs=torch.sigmoid(theta_true + zs)).sample()

    # random
    random_thata_hat = torch.zeros((num_test_taker,K_fit))
    random_thata_hats = [random_thata_hat]
    random_asked_zs = []
    random_asked_ys = []
    # breakpoint()
    # for i in tqdm(range(num_steps)):
    #     random_asked_zs.append(V_true[i].repeat(num_test_taker, 1))
    #     random_asked_ys.append(Y[:,i])
    #     # random_thata_hat: K_fit
    #     # random_asked_ys: num_item_pool
    #     # random_asked_zs: 
        
    #     random_thata_hat = estimate_theta(random_thata_hat, random_asked_ys, random_asked_zs)
    #     random_thata_hats.append(random_thata_hat)

    
    # adaptive
    
    adaptive_thata_hat = torch.ones((K_fit))
    adaptive_thata_hats = [adaptive_thata_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = V_true.clone()
    remain_ys = Y.clone()
    breakpoint()
    for i in tqdm(range(num_steps)):
        if i == 40:
            breakpoint()
        fisher_info = compute_fisher_info(adaptive_thata_hat, remain_zs)
        next_item = torch.argmax(fisher_info,axis=1)
        
        adaptive_asked_zs.append(remain_zs[next_item])
        adaptive_asked_ys.append(remain_ys[:,next_item]) 
        adaptive_thata_hat = estimate_theta(adaptive_thata_hat, adaptive_asked_ys, adaptive_asked_zs)
        adaptive_thata_hats.append(adaptive_thata_hat)
        remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
        remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
    
    
    
    
    plt.figure(figsize=(6, 5))
    
    # plt.plot(np.arange(num_steps+1), ((np.array(random_thata_hats) - U_true.numpy()) ** 2).sum(axis=1), label="random")
    # plt.plot(np.arange(num_steps+1), ((np.array(adaptive_thata_hats) - U_true.numpy()) ** 2).sum(axis=1), label="adaptive")

    # plt.plot(np.arange(num_steps+1), pred_mean_mse(torch.stack(random_thata_hats),V_true,Y), label="random")
    # plt.plot(np.arange(num_steps+1), pred_mean_mse(torch.stack(adaptive_thata_hats),V_true,Y), label="adaptive")
    
    plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(random_thata_hats,V_true,Y), label="random")
    plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(adaptive_thata_hats,V_true,Y), label="adaptive")
    
    
    plt.ylabel("auc")
    plt.title("auc")
    # plt.ylim(0, 1)
    plt.legend()

    
    plt.show()
    plt.savefig("plot/MSE_adap_5.png", dpi=200)
    
    
    