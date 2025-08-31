from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from simpler_model import fit_logistic_mf
from util import get_new_benchmark
import argparse
from torchmetrics import AUROC
torch.manual_seed(0)

def estimate_theta(theta, asked_ys, asked_zs):
    def closure():
        optim.zero_grad()

        probs = torch.sigmoid(theta @ asked_zs.T)
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss
    
    asked_ys = torch.tensor(asked_ys)
    asked_zs = torch.stack(asked_zs)
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
    p = torch.sigmoid(theta @ remain_zs.T)
    return p * (1 - p)

def auc_for_list_Uhat(U_hat_list,V,Y):

    auroc = AUROC(task="binary")
    aucs = []
    for uhat in U_hat_list:
        prob = torch.sigmoid(uhat @ V.T)
        aucs.append(auroc(prob,Y))
        
    return aucs


def grab_V_y(k):
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:1"
    
    
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_new_benchmark(seed)
    idx_split = int(data_withneg1.shape[0] * 0.8)
    train_data = data_withneg1[:idx_split,:]
    data_idtor_train = data_idtor[:idx_split,:]
    
    train_data_missing = train_data.clone().float()
    train_data_missing[~data_idtor_train] = float("nan")
    
    test_data = data_withneg1[idx_split:,:]
    data_idtor_test = data_idtor[idx_split:,:]
    
    model = fit_logistic_mf(train_data_missing, K=k, mask=data_idtor_train, steps=10, lr=5e-3, device=device)
    
    P_hat = torch.sigmoid(model.forward())
    auroc = AUROC(task="binary")
    train_auc = auroc(P_hat[data_idtor_train].cpu(), train_data_missing[data_idtor_train].cpu())
    print(f"factor {k} train auc: {train_auc}")
    V_true = model.V.clone().cpu()
    return V_true, test_data, data_idtor_test

if __name__ == "__main__":
    k = 2
    # V_true , test_data, data_idtor_test = grab_V_y(k)
    # torch.save(V_true, "data/cat/V_true.pt")
    # torch.save(test_data, "data/cat/test_data.pt")
    # torch.save(data_idtor_test, "data/cat/data_idtor_test.pt")
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test_taker_id", type=int, default=0, help="which test taker to simulate")
    
    V_true = torch.load("data/cat/V_true.pt")
    test_data = torch.load("data/cat/test_data.pt")
    data_idtor_test = torch.load("data/cat/data_idtor_test.pt")
    breakpoint()
    exit(0)
    #------------ test taker
    V_true = V_true.detach()
    
    # theta_true = 1.5
    num_item_pool = V_true.shape[0]
    num_steps = 50
    # zs = torch.randn(num_item_pool)
    # ys = Bernoulli(probs=torch.sigmoid(theta_true + zs)).sample()
    test_taker_id = 0
    idtor = data_idtor_test[test_taker_id]
    ys = test_data[test_taker_id][idtor].cpu()
    V_true = V_true[idtor]
    
    # random
    random_thata_hat = torch.zeros((k,))
    random_thata_hats = [random_thata_hat]
    random_asked_zs = []
    random_asked_ys = []

    for i in tqdm(range(num_steps)):
        random_asked_zs.append(V_true[i])
        random_asked_ys.append(ys[i])
        random_thata_hat = estimate_theta(random_thata_hat, random_asked_ys, random_asked_zs)
        random_thata_hats.append(random_thata_hat)
    
    # adaptive
    adaptive_thata_hat = torch.zeros((k,))
    adaptive_thata_hats = [adaptive_thata_hat]
    adaptive_asked_zs = []
    adaptive_asked_ys = []
    remain_zs = V_true.clone()
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
    # plt.plot(np.arange(num_steps+1), (np.array(random_thata_hats) - theta_true) ** 2, label="random")
    # plt.plot(np.arange(num_steps+1), (np.array(adaptive_thata_hats) - theta_true) ** 2, label="adaptive")
    
    plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(random_thata_hats,V_true,ys), label="random")
    plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(adaptive_thata_hats,V_true,ys), label="adaptive")
    
    plt.ylabel("auc")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    plt.savefig(f"plot/auc_adap_testing_{num_steps}.png", dpi=600)
