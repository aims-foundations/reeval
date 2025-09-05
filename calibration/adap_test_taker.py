from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from simpler_model import fit_logistic_mf
from util import get_official_provider_benchmark, get_everything_benchmark, get_helm_benchmark
import argparse
from rasch_model import rasch
import os
import multiprocessing as mp
from functools import partial
from torchmetrics import AUROC
torch.manual_seed(0)

def estimate_theta(theta, asked_ys, asked_zs, device="cuda:0"):
    def closure():
        optim.zero_grad()

        probs = torch.sigmoid(theta @ asked_zs.T)

        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    asked_ys = torch.tensor(asked_ys)
    asked_zs = torch.stack(asked_zs)
    theta = theta.clone().requires_grad_(True)
    asked_ys = asked_ys.to(device)
    asked_zs = asked_zs.to(device)
    theta = theta.to(device)
    
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


def estimate_theta_rasch(theta, asked_ys, asked_zs, device="cuda:0"):
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + asked_zs[None, :])
        loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    asked_ys = torch.tensor(asked_ys)
    asked_zs = torch.tensor(asked_zs)
    theta = theta.clone().requires_grad_(True)
    asked_ys = asked_ys.to(device)
    asked_zs = asked_zs.to(device)
    theta = theta.to(device)
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

def compute_fisher_info_rasch(theta, remain_zs):
    p = torch.sigmoid(theta[:, None] + remain_zs[None, :])
    return p * (1 - p)

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


def grab_V_y(dataset,k,seed=0):
    torch.manual_seed(seed)
    device = "cuda:0"
    
    if dataset == "HELM":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_helm_benchmark(seed, "random_mask")
    elif dataset == "everything2":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_everything_benchmark(seed, "random_mask")
    elif dataset == "official_provider":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_official_provider_benchmark(seed, "random_mask")
    else:
        assert False
        
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

def grab_V_y_rasch(dataset,seed=0):

    torch.manual_seed(seed)
    
    
    if dataset == "HELM":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_helm_benchmark(seed, "random_mask")
    elif dataset == "everything2":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_everything_benchmark(seed, "random_mask")
    elif dataset == "official_provider":
        data_withneg1, data_with0, data_idtor, train_idtor, test_idtor, _ = get_official_provider_benchmark(seed, "random_mask")
    else:
        assert False
    

    train_row_idtor = torch.bernoulli(data_idtor.max(axis=1).values * 0.8).bool()
    
    train_data = data_with0[train_row_idtor] 
    test_data = data_with0[~train_row_idtor] 
    data_idtor_train = data_idtor[train_row_idtor]
    data_idtor_test = data_idtor[~train_row_idtor]
    train_data_missing = train_data.clone().float()

    # P_hat, _, zs = rasch(data_with0, train_idtor=train_idtor, B=5_000, device="cuda:0")
    batch = 5000
    if dataset == 'everything2':
        batch = 500
    P_hat, _, zs = rasch(train_data_missing, train_idtor=data_idtor_train, B=batch, device="cuda:0")
    auroc = AUROC(task="binary")
    train_auc = auroc(P_hat[data_idtor_train].cpu(), train_data_missing[data_idtor_train].cpu())
    print(f"sanity check: train auc{train_auc}")
    return zs, test_data, data_idtor_test



def test_taker(factor, V, test_data, data_idtor_test ,test_taker_id, data_method_name):
    # try:
        device = "cuda:0"
        run_name = f"{data_method_name}_test_taker_id{test_taker_id}"
        res_path = f"results/cat/adaptive_thata_hats_{run_name}.pt"
        if os.path.exists(res_path):      
            if torch.load(res_path) is not None:
                print(f"[Skip] {res_path} already exist. Exiting.")
                return


        num_item_pool = V.shape[0]
        num_steps = 50
        # zs = torch.randn(num_item_pool)
        # ys = Bernoulli(probs=torch.sigmoid(theta_true + zs)).sample()

        
        # for test_taker_id in range(test_data.shape[0]):
        print("test_taker_id",test_taker_id)

        idtor = data_idtor_test[test_taker_id]
        ys = test_data[test_taker_id][idtor].cpu()
        V_true = V.clone().detach()
        V_true = V_true[idtor]

        # random

        random_thata_hat = torch.zeros((max(1,factor),), device=device)

        random_thata_hats = [random_thata_hat]
        random_asked_zs = []
        random_asked_ys = []

        for i in tqdm(range(num_steps)):
            random_asked_zs.append(V_true[i])
            random_asked_ys.append(ys[i])
            # try:
            if factor == 0:
                random_thata_hat = estimate_theta_rasch(random_thata_hat, random_asked_ys, random_asked_zs)
            else:
                random_thata_hat = estimate_theta(random_thata_hat, random_asked_ys, random_asked_zs)
            # except:
            #     breakpoint()
            #     print("err at random")

            
            random_thata_hats.append(random_thata_hat)

        # adaptive
        adaptive_thata_hat = torch.zeros((max(1,factor),), device=device)
        adaptive_thata_hats = [adaptive_thata_hat]
        adaptive_asked_zs = []
        adaptive_asked_ys = []
        remain_zs = V_true.clone()
        remain_ys = ys.clone()
        for _ in tqdm(range(num_steps)):

            if factor == 0:
                fisher_info = compute_fisher_info_rasch(adaptive_thata_hat, remain_zs.to(device))
            else:
                fisher_info = compute_fisher_info(adaptive_thata_hat, remain_zs.to(device))

            next_item = torch.argmax(fisher_info)
            adaptive_asked_zs.append(remain_zs[next_item])
            adaptive_asked_ys.append(remain_ys[next_item])
            # try:
            if factor == 0:
                adaptive_thata_hat = estimate_theta_rasch(adaptive_thata_hat, adaptive_asked_ys, adaptive_asked_zs)
            else:
                adaptive_thata_hat = estimate_theta(adaptive_thata_hat, adaptive_asked_ys, adaptive_asked_zs)
            # except:
            #     print("err at adaptive testing")
            adaptive_thata_hats.append(adaptive_thata_hat)
            remain_zs = torch.cat([remain_zs[:next_item], remain_zs[next_item + 1:]])
            remain_ys = torch.cat([remain_ys[:next_item], remain_ys[next_item + 1:]])
    
    
            torch.save(random_thata_hats, f"results/cat/random_thata_hats_{run_name}.pt")
            torch.save(adaptive_thata_hats, f"results/cat/adaptive_thata_hats_{run_name}.pt")
    # except:
    #     print(f"crashed{run_name}")
    
    # plt.figure(figsize=(6, 5))
    # # plt.plot(np.arange(num_steps+1), (np.array(random_thata_hats) - theta_true) ** 2, label="random")
    # # plt.plot(np.arange(num_steps+1), (np.array(adaptive_thata_hats) - theta_true) ** 2, label="adaptive")
    
    # plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(random_thata_hats,V_true,ys), label="random")
    # plt.plot(np.arange(num_steps+1), auc_for_list_Uhat(adaptive_thata_hats,V_true,ys), label="adaptive")
    
    # plt.ylabel("auc")
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()
    # plt.savefig(f"plot/auc_adap_testing_{num_steps}.png", dpi=600)






def run_single_test_taker(dataset,factor):

    data_method_name = f"factor_{factor}_dataset{dataset}_v2"
    


    data_idtor_test_path = f"data/cat/data_idtor_test_{data_method_name}.pt"
    if not os.path.exists(data_idtor_test_path):  
        if factor == 0:
            V_true , test_data, data_idtor_test = grab_V_y_rasch(dataset)
        else:
            V_true , test_data, data_idtor_test = grab_V_y(dataset, factor)
        
        torch.save(V_true, f"data/cat/V_true_{data_method_name}.pt")
        torch.save(test_data, f"data/cat/test_data_{data_method_name}.pt")
        torch.save(data_idtor_test, f"data/cat/data_idtor_test_{data_method_name}.pt")
        return
        
    V = torch.load(f"data/cat/V_true_{data_method_name}.pt").cpu()
    test_data = torch.load(f"data/cat/test_data_{data_method_name}.pt").cpu()
    data_idtor_test = torch.load(f"data/cat/data_idtor_test_{data_method_name}.pt").cpu()
    print("loaded V")
    #------------ test taker

    # Create a partial function with fixed arguments
    test_taker_partial = partial(test_taker, factor, V, test_data, data_idtor_test)
    
    # Get list of test_taker_ids
    test_taker_ids = list(range(test_data.shape[0]))
    
    # for i in test_taker_ids:
    #     # test_taker_partial(i,data_method_name)
    #     test_taker(factor, V, test_data, data_idtor_test, i,data_method_name)
    # Parallelize the loop
    with mp.Pool(processes=50) as pool:
        pool.starmap(test_taker_partial, [(test_taker_id, data_method_name) for test_taker_id in test_taker_ids])
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="everything2", help="which test taker to simulate")
    parser.add_argument("--factor", type=int, default=2, help="which test taker to simulate")
    # parser.add_argument("--test_taker_id", type=int, default=0, help="which test taker to simulate")
    args = parser.parse_args()
    # V, test_data, data_idtor_test = grab_V_y_rasch()
    
    device = "cuda:0"
    os.makedirs("data/cat", exist_ok=True)
    os.makedirs("results/cat", exist_ok=True)
    
    
    run_single_test_taker(args.dataset, args.factor)