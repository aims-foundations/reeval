from tqdm import tqdm
import torch
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from simpler_model import fit_logistic_mf
from grab_data import get_new_benchmark
from torchmetrics import AUROC
torch.manual_seed(0)



def estimate_theta_adap(theta, asked_ys, asked_zs, device=None, idtor=None):
    def closure():
        optim.zero_grad()

        # print(asked_zs)
        # step * testtaker
        probs = torch.sigmoid(torch.einsum('nk,nik->ni',theta,asked_zs)) 
        if idtor is not None:

            loss = -Bernoulli(probs=probs[idtor]).log_prob(asked_ys[idtor]).mean()        
        else:
            loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss

    # test_taker * obs

    asked_ys = [torch.stack(test_taker_y) for test_taker_y in asked_ys]
    asked_ys = torch.stack(asked_ys)
    
    
    # test_taker * obs * fac
    asked_zs = [torch.stack(test_taker_z) for test_taker_z in asked_zs]
    asked_zs = torch.stack(asked_zs)
    # test_taker * fac
    theta = theta.clone().requires_grad_(True)

    if device:
        asked_ys = asked_ys.to(device)
        asked_zs = asked_zs.to(device)
        theta = theta.to(device)
    
    
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    for iteration in range(100):
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        print(f"training at iteration {iteration}")
        
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
    seed = 0
    torch.manual_seed(seed)
    num_item_pool = 1000
    # num_steps = 50
    num_steps = 10
    k = 2
    device = "cuda:1"
    data_withneg1, data_with0, data_idtor, train_idtor, test_idtor = get_new_benchmark(seed)
    idx_split = int(data_withneg1.shape[0] * 0.99)
    train_data = data_withneg1[:idx_split,:]
    data_idtor_train = data_idtor[:idx_split,:]
    
    train_data_missing = train_data.clone().float()
    train_data_missing[~data_idtor_train] = float("nan")
    
    test_data = data_withneg1[idx_split:,:]
    data_idtor_test = data_idtor[idx_split:,:]
    num_test_taker, num_item_pool, K_fit = test_data.shape[0], test_data.shape[1], k
    
    model = fit_logistic_mf(train_data_missing, K=K_fit, mask=data_idtor_train, steps=10, lr=5e-3, device=device)
    
    P_hat = torch.sigmoid(model.forward())
    auroc = AUROC(task="binary")
    train_auc = auroc(P_hat[data_idtor_train].cpu(), train_data_missing[data_idtor_train].cpu())
    print(f"factor {K_fit} train auc: {train_auc}")
    # Create a small synthetic dataset
    # theta
    
    V_true = model.V
    Y = test_data.clone()

    # ceiling_asked_ys = [list(Y[i].clone()) for i in  range(num_test_taker)]
    # ceiling_asked_zs = [list(V_true.clone()) for _ in  range(num_test_taker)]
    # ceiling_thata_hat = torch.ones((num_test_taker,K_fit),device = device)

    # random_thata_hat = estimate_theta_adap(ceiling_thata_hat, ceiling_asked_ys, ceiling_asked_zs,device = device, idtor=data_idtor_test)
    # auc_ceiling  = auc_for_list_Uhat([random_thata_hat],V_true,Y)
    # print("ceiling auc:",auc_ceiling)

    breakpoint()
    random_thata_hat = torch.ones((num_test_taker,K_fit),device = device)
    random_thata_hats = [random_thata_hat]
    random_asked_zs = [[] for _ in range(num_test_taker)]
    random_asked_ys = [[] for _ in range(num_test_taker)]
    random_remain_zs = [V_true.clone() for _ in  range(num_test_taker)]
    random_remain_ys = [Y[i].clone() for i in  range(num_test_taker)]

    for i in tqdm(range(num_steps)):
        # if i == 40:
        #     breakpoint()

        next_items = []


        for test_taker in range(num_test_taker):
            fisher_info = compute_fisher_info(random_thata_hat[test_taker], random_remain_zs[test_taker])
            next_item = torch.randint(0, len(fisher_info), (1,)).squeeze()
            next_items.append(next_item)
            # adaptive_asked_zs: test_taker * obs * fac

            random_asked_zs[test_taker].append(random_remain_zs[test_taker][next_item])
            random_asked_ys[test_taker].append(random_remain_ys[test_taker][next_item]) 
            # remain_zs: test taker * items * fac
            random_remain_zs[test_taker] = torch.cat([random_remain_zs[test_taker][:next_item], random_remain_zs[test_taker][next_item + 1:]])
            # remain_zs: test taker * items 
            random_remain_ys[test_taker] = torch.cat([random_remain_ys[test_taker][:next_item], random_remain_ys[test_taker][next_item + 1:]])
            
        
        random_thata_hat = estimate_theta_adap(random_thata_hat, random_asked_ys, random_asked_zs)
        random_thata_hats.append(random_thata_hat)
    
    
    
    adaptive_thata_hat = torch.ones((num_test_taker,K_fit))
    adaptive_thata_hats = [adaptive_thata_hat]
    adaptive_asked_zs = [[] for _ in range(num_test_taker)]
    adaptive_asked_ys = [[] for _ in range(num_test_taker)]
    remain_zs = [V_true.clone() for _ in  range(num_test_taker)]
    remain_ys = [Y[i].clone() for i in  range(num_test_taker)]

    for i in tqdm(range(num_steps)):
        # if i == 40:
        #     breakpoint()
        
        next_items = []
        # for i in range(num_item_pool):
        for test_taker in range(num_test_taker):
            fisher_info = compute_fisher_info(adaptive_thata_hat[test_taker], remain_zs[test_taker])
            next_item = torch.argmax(fisher_info)

            next_items.append(next_item)
            # adaptive_asked_zs: test_taker * obs * fac
            adaptive_asked_zs[test_taker].append(remain_zs[test_taker][next_item])
            adaptive_asked_ys[test_taker].append(remain_ys[test_taker][next_item]) 
            # remain_zs: test taker * items * fac
            remain_zs[test_taker] = torch.cat([remain_zs[test_taker][:next_item], remain_zs[test_taker][next_item + 1:]])
            # remain_zs: test taker * items 
            remain_ys[test_taker] = torch.cat([remain_ys[test_taker][:next_item], remain_ys[test_taker][next_item + 1:]])
        
        
        adaptive_thata_hat = estimate_theta_adap(adaptive_thata_hat, adaptive_asked_ys, adaptive_asked_zs)
        adaptive_thata_hats.append(adaptive_thata_hat)
        
    
    
    
    
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
    plt.savefig(f"plot/MSE_adap_step{num_steps}.png", dpi=200)
    
    
    